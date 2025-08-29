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
        assert math.ceil(width / float(tile_size)) == n_tiles_x, f"Got {width=}, {tile_size=}, {n_tiles_x=}"
        assert math.ceil(height / float(tile_size)) == n_tiles_y, f"Got {height=}, {tile_size=}, {n_tiles_y=}"
        
        # make sure tensors are contiguous
        opacities = opacities.contiguous()
        means2d = means2d.contiguous()
        conics = conics.contiguous()
        colors = colors.contiguous()
        isect_offsets = isect_offsets.contiguous()
        isect_primitive_ids = flatten_ids.to(torch.uint32).contiguous()

        # Note: new API requires inclusive sum (i.e. prefix sum) while in old gsplat API
        # the `isect_offsets` stores exclusive sum. So we convert it here. 
        # TODO: update the intersection API to produce inclusive sum.
        isect_offsets = isect_offsets.reshape(-1)
        isect_prefix_sum_per_tile = torch.empty_like(isect_offsets)
        isect_prefix_sum_per_tile[:-1] = isect_offsets[1:]
        isect_prefix_sum_per_tile[-1] = isect_primitive_ids.numel()
        isect_prefix_sum_per_tile = isect_prefix_sum_per_tile.to(torch.uint32).contiguous()

        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "image_gaussian_rasterize_forward"
        )(
            # Primitives
            opacities,
            means2d,
            conics,
            colors,
            # Images
            n_images,
            width, # image_width
            height, # image_height
            tile_size, # tile_width
            tile_size, # tile_height
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

        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [..., H, W, 3]
        v_render_alphas: Tensor,  # [..., H, W, 1]
    ):
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
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size

        raise NotImplementedError("Not implemented")

        # (
        #     v_means2d_abs,
        #     v_means2d,
        #     v_conics,
        #     v_colors,
        #     v_opacities,
        # ) = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
        #     means2d,
        #     conics,
        #     colors,
        #     opacities,
        #     backgrounds,
        #     masks,
        #     width,
        #     height,
        #     tile_size,
        #     isect_offsets,
        #     flatten_ids,
        #     render_alphas,
        #     last_ids,
        #     v_render_colors.contiguous(),
        #     v_render_alphas.contiguous(),
        #     absgrad,
        # )

        # if absgrad:
        #     means2d.absgrad = v_means2d_abs

        # if ctx.needs_input_grad[4]:
        #     v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
        #         dim=(-3, -2)
        #     )
        # else:
        #     v_backgrounds = None

        return (
            None, # v_means2d
            None, # v_conics
            None, # v_colors
            None, # v_opacities
            None,
            None,
            None,
            None,
            None,
        )