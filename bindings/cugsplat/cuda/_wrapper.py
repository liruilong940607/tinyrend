import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [n_primitives, 2]
        conics: Tensor,  # [n_primitives, 3]
        colors: Tensor,  # [n_primitives, channels]
        opacities: Tensor,  # [n_primitives]
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [n_images, n_tiles_y, n_tiles_x]
        flatten_ids: Tensor,  # [n_isects]
        enable_fused_jvp: bool = False,
        v_means2d: Optional[Tensor] = None,
        v_conics: Optional[Tensor] = None,
        v_colors: Optional[Tensor] = None,
        v_opacities: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        n_primitives = means2d.size(0)
        channels = colors.size(-1)
        assert means2d.shape == (n_primitives, 2), f"Got {means2d.shape=}"
        assert conics.shape == (n_primitives, 3), f"Got {conics.shape=}"
        assert colors.shape == (n_primitives, channels), f"Got {colors.shape=}"
        assert opacities.shape == (n_primitives,), f"Got {opacities.shape=}"

        assert isect_offsets.ndim == 3, f"Got {isect_offsets.ndim=}"
        assert flatten_ids.ndim == 1, f"Got {flatten_ids.ndim=}"

        n_images, n_tiles_y, n_tiles_x = isect_offsets.shape
        assert (
            math.ceil(width / float(tile_size)) == n_tiles_x
        ), f"Got {width=}, {tile_size=}, {n_tiles_x=}"
        assert (
            math.ceil(height / float(tile_size)) == n_tiles_y
        ), f"Got {height=}, {tile_size=}, {n_tiles_y=}"

        # make sure tensors are contiguous
        opacities = opacities.contiguous()
        means2d = means2d.contiguous()
        conics = conics.contiguous()
        colors = colors.contiguous()
        isect_offsets = isect_offsets.contiguous()
        isect_primitive_ids = flatten_ids.to(torch.uint32).contiguous()

        # Mark non-differentiable tensors
        ctx.mark_non_differentiable(isect_offsets)
        ctx.mark_non_differentiable(isect_primitive_ids)

        # Note: new API requires inclusive sum (i.e. prefix sum) while in old gsplat API
        # the `isect_offsets` stores exclusive sum. So we convert it here.
        # TODO: update the intersection API to produce inclusive sum.
        isect_offsets = isect_offsets.reshape(-1)
        isect_prefix_sum_per_tile = torch.empty_like(isect_offsets)
        isect_prefix_sum_per_tile[:-1] = isect_offsets[1:]
        isect_prefix_sum_per_tile[-1] = isect_primitive_ids.numel()
        isect_prefix_sum_per_tile = isect_prefix_sum_per_tile.to(
            torch.uint32
        ).contiguous()

        ctx.mark_non_differentiable(isect_offsets)
        ctx.mark_non_differentiable(isect_prefix_sum_per_tile)

        if enable_fused_jvp:
            # In fused JVP, we pass in both the primal and tangent values of the primitives to
            # the kernel and return both the primal and tangent values of the render output.
            kernel_name = "image_gaussian_rasterize_jvp"
            assert v_means2d is not None, "v_means2d is required for jvp"
            assert v_conics is not None, "v_conics is required for jvp"
            assert v_colors is not None, "v_colors is required for jvp"
            assert v_opacities is not None, "v_opacities is required for jvp"
            opacities = torch.stack([opacities, v_opacities], dim=-1).contiguous()
            means2d = torch.stack([means2d, v_means2d], dim=-1).contiguous()
            conics = torch.stack([conics, v_conics], dim=-1).contiguous()
            colors = torch.stack([colors, v_colors], dim=-1).contiguous()
        else:
            kernel_name = "image_gaussian_rasterize_forward"

        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(kernel_name)(
            # Primitives
            opacities,
            means2d,
            conics,
            colors,
            # Images
            n_images,
            width,  # image_width
            height,  # image_height
            tile_size,  # tile_width
            tile_size,  # tile_height
            # Intersections
            isect_primitive_ids,
            isect_prefix_sum_per_tile,
        )

        if enable_fused_jvp:
            # In fused JVP, the output contains both the primal and tangent values.
            # We save the tangents in the ctx, so that jvp() can directly return them.
            render_colors, render_colors_tangent = torch.unbind(render_colors, dim=-1)
            render_alphas, render_alphas_tangent = torch.unbind(render_alphas, dim=-1)
            render_colors = render_colors.contiguous()
            render_alphas = render_alphas.contiguous()
            render_colors_tangent = render_colors_tangent.contiguous()
            render_alphas_tangent = render_alphas_tangent.contiguous()
            ctx.save_for_forward(render_colors_tangent, render_alphas_tangent)

        else:
            ctx.save_for_forward(
                # Primitives
                opacities,
                means2d,
                conics,
                colors,
                # Intersections
                isect_primitive_ids,
                isect_prefix_sum_per_tile,
            )

        ctx.save_for_backward(
            # Primitives
            opacities,
            means2d,
            conics,
            colors,
            # Intersections
            isect_primitive_ids,
            isect_prefix_sum_per_tile,
            # Forward Outputs
            render_alphas,
            last_ids,
        )
        ctx.n_images = n_images
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.enable_fused_jvp = enable_fused_jvp

        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [..., H, W, 3]
        v_render_alphas: Tensor,  # [..., H, W, 1]
    ):
        # make tensors contiguous
        v_render_colors = v_render_colors.contiguous()
        v_render_alphas = v_render_alphas.contiguous()

        (
            # Primitives
            opacities,
            means2d,
            conics,
            colors,
            # Intersections
            isect_primitive_ids,
            isect_prefix_sum_per_tile,
            # Forward Outputs
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        n_images = ctx.n_images
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size

        (v_opacities, v_means2d, v_conics, v_colors,) = _make_lazy_cuda_func(
            "image_gaussian_rasterize_backward"
        )(
            # Primitives
            opacities,
            means2d,
            conics,
            colors,
            # Images
            n_images,
            width,  # image_width
            height,  # image_height
            tile_size,  # tile_width
            tile_size,  # tile_height
            # Intersections
            isect_primitive_ids,
            isect_prefix_sum_per_tile,
            # Forward Outputs
            last_ids,  # render_last_ids
            render_alphas,
            # Gradients for forward output
            v_render_alphas,
            v_render_colors,
        )

        return (
            v_means2d,  # v_means2d
            v_conics,  # v_conics
            v_colors,  # v_colors
            v_opacities,  # v_opacities
            None,  # v_width
            None,  # v_height
            None,  # v_tile_size
            None,  # v_isect_offsets
            None,  # v_flatten_ids
            None,  # v_enable_fused_jvp
            None,  # v_means2d_tangent
            None,  # v_conics_tangent
            None,  # v_colors_tangent
            None,  # v_opacities_tangent
        )

    @staticmethod
    def jvp(
        ctx,
        v_means2d: Tensor,
        v_conics: Tensor,
        v_colors: Tensor,
        v_opacities: Tensor,
        unused_v_width: int,
        unused_v_height: int,
        unused_v_tile_size: int,
        unused_v_isect_offsets: Tensor,
        unused_v_flatten_ids: Tensor,
        unused_enable_fused_jvp: bool = False,
        unused_v_means2d: Optional[Tensor] = None,
        unused_v_conics: Optional[Tensor] = None,
        unused_v_colors: Optional[Tensor] = None,
        unused_v_opacities: Optional[Tensor] = None,
    ):
        if ctx.enable_fused_jvp:
            # In fused JVP, we expect the tangents have been computed in the forward pass.
            # so we could just grab and return.
            v_render_colors, v_render_alphas = ctx.saved_tensors
            assert v_render_colors is not None, "forward should have saved the tangents"
            assert v_render_alphas is not None, "forward should have saved the tangents"
            return v_render_colors, v_render_alphas

        (
            # Primitives
            opacities,
            means2d,
            conics,
            colors,
            # Intersections
            isect_primitive_ids,
            isect_prefix_sum_per_tile,
        ) = ctx.saved_tensors
        n_images = ctx.n_images
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size

        assert (
            v_opacities.shape == opacities.shape
        ), f"Got {v_opacities.shape=}, {opacities.shape=}"
        assert (
            v_means2d.shape == means2d.shape
        ), f"Got {v_means2d.shape=}, {means2d.shape=}"
        assert v_conics.shape == conics.shape, f"Got {v_conics.shape=}, {conics.shape=}"
        assert v_colors.shape == colors.shape, f"Got {v_colors.shape=}, {colors.shape=}"

        render_colors, render_alphas, _ = _make_lazy_cuda_func(
            "image_gaussian_rasterize_jvp"
        )(
            # Primitives
            torch.stack([opacities, v_opacities], dim=-1),
            torch.stack([means2d, v_means2d], dim=-1),
            torch.stack([conics, v_conics], dim=-1),
            torch.stack([colors, v_colors], dim=-1),
            # Images
            n_images,
            width,  # image_width
            height,  # image_height
            tile_size,  # tile_width
            tile_size,  # tile_height
            # Intersections
            isect_primitive_ids,
            isect_prefix_sum_per_tile,
        )

        v_render_colors = render_colors[..., 1]
        v_render_alphas = render_alphas[..., 1]

        return v_render_colors, v_render_alphas
