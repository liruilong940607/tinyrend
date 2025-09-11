import math
from typing import Optional, Tuple
from collections import namedtuple

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor
from typing_extensions import Literal

from .common import has_tangent, _make_lazy_cuda_func


ProjectionFusedEWA3DGSOutput = namedtuple("ProjectionFusedEWA3DGSOutput", ["radii", "means2d", "depth", "conic"])


def projection_fused_ewa_3dgs(
    resolution: Tuple[int, int],
    principal_point: Tuple[float, float],
    focal_length: Tuple[float, float],

    viewmat_R: Tensor,  # [C, 3, 3]
    viewmat_t: Tensor,  # [C, 3]

    near_plane: float,
    far_plane: float,
    eps2d: float,

    opacities: Tensor,  # [N]
    means: Tensor,     # [N, 3]
    quats: Tensor,     # [N, 4]
    scales: Tensor,    # [N, 3]
) -> ProjectionFusedEWA3DGSOutput:
    """Projection fused EWA 3DGS"""
    radii, means2d, depth, conic = _ProjectionFusedEWA3DGS.apply(
        resolution,
        principal_point,
        focal_length,
        viewmat_R,
        viewmat_t,
        near_plane,
        far_plane,
        eps2d,
        opacities,
        means,
        quats,
        scales,
    )
    return ProjectionFusedEWA3DGSOutput(radii=radii, means2d=means2d, depth=depth, conic=conic)


class _ProjectionFusedEWA3DGS(torch.autograd.Function):
    """Projection fused EWA 3DGS"""

    @staticmethod
    def forward(
        ctx,
        resolution: Tuple[int, int],
        principal_point: Tuple[float, float],
        focal_length: Tuple[float, float],

        viewmat_R: Tensor,  # [C, 3, 3]
        viewmat_t: Tensor,  # [C, 3]

        near_plane: float,
        far_plane: float,
        eps2d: float,

        opacities: Tensor,  # [N]
        means: Tensor,     # [N, 3]
        quats: Tensor,     # [N, 4]
        scales: Tensor,    # [N, 3]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        N = means.size(0)
        C = viewmat_R.size(0)
        assert means.shape == (N, 3), f"Got {means.shape=}"
        assert quats.shape == (N, 4), f"Got {quats.shape=}"
        assert scales.shape == (N, 3), f"Got {scales.shape=}"
        assert opacities.shape == (N,), f"Got {opacities.shape=}"

        assert viewmat_R.shape == (C, 3, 3), f"Got {viewmat_R.shape=}"
        assert viewmat_t.shape == (C, 3), f"Got {viewmat_t.shape=}"

        # make sure tensors are contiguous
        viewmat_R = viewmat_R.contiguous()
        viewmat_t = viewmat_t.contiguous()
        opacities = opacities.contiguous()
        means = means.contiguous()
        quats = quats.contiguous()
        scales = scales.contiguous()

        # Mark non-differentiable tensors
        ctx.mark_non_differentiable(viewmat_R)
        ctx.mark_non_differentiable(viewmat_t)

        kernel_name = "projection_fused_ewa_3dgs_fwd"

        radii, means2d, depth, conic = _make_lazy_cuda_func(kernel_name)(
            # Perfect Pinhole Camera: All cameras share the same intrinsic.
            resolution[0], # image_width
            resolution[1], # image_height
            principal_point[0], # principal_point_x
            principal_point[1], # principal_point_y
            focal_length[0], # focal_length_x
            focal_length[1], # focal_length_y

            viewmat_R,
            viewmat_t,

            near_plane,
            far_plane,
            eps2d,

            opacities,
            means,
            quats,
            scales,
        )

        return radii, means2d, depth, conic

    @staticmethod
    def backward(
        ctx,
        v_radii: Tensor,  # [C, N, 2]
        v_means2d: Tensor,  # [C, N, 2]
        v_depth: Tensor,  # [C, N]
        v_conic: Tensor,  # [C, N, 3]
    ):
        # make tensors contiguous
        v_radii = v_radii.contiguous()
        v_means2d = v_means2d.contiguous()
        v_depth = v_depth.contiguous()
        v_conic = v_conic.contiguous()

        raise NotImplementedError("Backward pass is not implemented")

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
