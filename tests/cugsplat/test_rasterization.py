import os
import sys
import pytest
import torch
from typing import Tuple

import math
import imageio
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BINDINGS_DIR = os.path.join(CURRENT_DIR, "..", "..", "bindings")
sys.path.append(BINDINGS_DIR)
from cugsplat.cuda._wrapper import _RasterizeToPixels
import torch.autograd.forward_ad as fwAD
import tqdm

device = torch.device("cuda")


def create_test_data():
    torch.manual_seed(42)

    from gsplat._helper import load_test_data

    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(
            os.path.dirname(__file__), "../../assets/test_garden.npz"
        ),
    )
    return {
        "means": means,  # [N, 3]
        "quats": quats,  # [N, 4]
        "scales": scales,  # [N, 3]
        "opacities": opacities,  # [N]
        "viewmats": viewmats,  # [C, 4, 4]
        "Ks": Ks,  # [C, 3, 3]
        "width": width,
        "height": height,
    }


def create_rasterizer_test_data():
    test_data = create_test_data()
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
    )

    N = test_data["means"][::10].shape[0]
    C = test_data["viewmats"].shape[0]

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"][::10]
    scales = test_data["scales"][::10]
    means = test_data["means"][::10]
    opacities = test_data["opacities"][::10]
    colors = torch.rand(C, N, 3, device=device)
    backgrounds = torch.zeros((C, 3), device=device)

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    opacities = torch.broadcast_to(opacities[None, :], (C, N))

    # Identify intersecting tiles
    tile_size = 16
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    return {
        "means2d": means2d,
        "conics": conics,
        "colors": colors,
        "opacities": opacities,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "isect_offsets": isect_offsets,
        "flatten_ids": flatten_ids,
        "backgrounds": backgrounds,
    }


@pytest.fixture
def test_data():
    return create_test_data()


@pytest.fixture
def rasterizer_test_data():
    return create_rasterizer_test_data()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_to_pixels(rasterizer_test_data: dict):
    from gsplat.cuda._wrapper import rasterize_to_pixels

    means2d = rasterizer_test_data["means2d"]
    conics = rasterizer_test_data["conics"]
    colors = rasterizer_test_data["colors"]
    opacities = rasterizer_test_data["opacities"]
    width = rasterizer_test_data["width"]
    height = rasterizer_test_data["height"]
    tile_size = rasterizer_test_data["tile_size"]
    isect_offsets = rasterizer_test_data["isect_offsets"]
    flatten_ids = rasterizer_test_data["flatten_ids"]
    backgrounds = rasterizer_test_data["backgrounds"]  # not supported yet

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    # forward
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )
    render_colors_, render_alphas_ = _RasterizeToPixels.apply(
        means2d.reshape(-1, 2),
        conics.reshape(-1, 3),
        colors.reshape(-1, 3),
        opacities.reshape(-1),
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    torch.testing.assert_close(render_colors, render_colors_)
    torch.testing.assert_close(render_alphas, render_alphas_)

    # backward
    v_render_colors = torch.rand_like(render_colors)
    v_render_alphas = torch.rand_like(render_alphas)

    v_means2d, v_conics, v_colors, v_opacities = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities),
    )
    v_means2d_, v_conics_, v_colors_, v_opacities_ = torch.autograd.grad(
        (render_colors_ * v_render_colors).sum()
        + (render_alphas_ * v_render_alphas).sum(),
        (means2d, conics, colors, opacities),
    )
    torch.testing.assert_close(v_means2d, v_means2d_)
    torch.testing.assert_close(v_conics, v_conics_, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_colors, v_colors_)
    torch.testing.assert_close(v_opacities, v_opacities_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_to_pixels_jvp(rasterizer_test_data: dict):
    means2d = rasterizer_test_data["means2d"].reshape(-1, 2)
    conics = rasterizer_test_data["conics"].reshape(-1, 3)
    colors = rasterizer_test_data["colors"].reshape(-1, 3)
    opacities = rasterizer_test_data["opacities"].reshape(-1)
    width = rasterizer_test_data["width"]
    height = rasterizer_test_data["height"]
    tile_size = rasterizer_test_data["tile_size"]
    isect_offsets = rasterizer_test_data["isect_offsets"]
    flatten_ids = rasterizer_test_data["flatten_ids"]
    backgrounds = rasterizer_test_data["backgrounds"]  # not supported yet

    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    with fwAD.dual_level():
        # Normal JVP
        dual_means2d = fwAD.make_dual(means2d, torch.rand_like(means2d))
        dual_conics = fwAD.make_dual(conics, torch.rand_like(conics))
        dual_colors = fwAD.make_dual(colors, torch.rand_like(colors))
        dual_opacities = fwAD.make_dual(opacities, torch.rand_like(opacities))

        dual_render_colors, dual_render_alphas = _RasterizeToPixels.apply(
            dual_means2d,
            dual_conics,
            dual_colors,
            dual_opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )
        torch.testing.assert_close(
            render_colors, fwAD.unpack_dual(dual_render_colors).primal
        )
        assert render_colors.shape == fwAD.unpack_dual(dual_render_colors).tangent.shape
        torch.testing.assert_close(
            render_alphas, fwAD.unpack_dual(dual_render_alphas).primal
        )
        assert render_alphas.shape == fwAD.unpack_dual(dual_render_alphas).tangent.shape

        # Fused JVP
        dual_render_colors_, dual_render_alphas_ = _RasterizeToPixels.apply(
            dual_means2d,
            dual_conics,
            dual_colors,
            dual_opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            True,  # enable_fused_jvp
            fwAD.unpack_dual(dual_means2d).tangent,  # means2d_tangent
            fwAD.unpack_dual(dual_conics).tangent,  # conics_tangent
            fwAD.unpack_dual(dual_colors).tangent,  # colors_tangent
            fwAD.unpack_dual(dual_opacities).tangent,  # opacities_tangent
        )
        torch.testing.assert_close(
            render_colors, fwAD.unpack_dual(dual_render_colors_).primal
        )
        assert (
            render_colors.shape == fwAD.unpack_dual(dual_render_colors_).tangent.shape
        )
        torch.testing.assert_close(
            render_alphas, fwAD.unpack_dual(dual_render_alphas_).primal
        )
        assert (
            render_alphas.shape == fwAD.unpack_dual(dual_render_alphas_).tangent.shape
        )


if __name__ == "__main__":
    rasterizer_test_data = create_rasterizer_test_data()

    test_rasterize_to_pixels(rasterizer_test_data=rasterizer_test_data)
    test_rasterize_to_pixels_jvp(rasterizer_test_data=rasterizer_test_data)
