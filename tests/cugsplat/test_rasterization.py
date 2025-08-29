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
        data_path=os.path.join(os.path.dirname(__file__), "../../assets/test_garden.npz"),
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
    
    N = test_data["means"].shape[0]
    C = test_data["viewmats"].shape[0]

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    opacities = test_data["opacities"]
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
    backgrounds = rasterizer_test_data["backgrounds"]

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
        # backgrounds=backgrounds, #TODO not supported yet
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

    


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
# # @pytest.mark.parametrize("channels", [3, 32, 128])
# # @pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
# @pytest.mark.parametrize("channels", [3])
# @pytest.mark.parametrize("batch_dims", [()])
# @pytest.mark.parametrize("benchmark", [False])
# def test_rasterize_to_pixels(test_data, channels: int, batch_dims: Tuple[int, ...], benchmark: bool):
#     from gsplat.cuda._wrapper import (
#         fully_fused_projection,
#         isect_offset_encode,
#         isect_tiles,
#         quat_scale_to_covar_preci,
#         rasterize_to_pixels,
#     )

#     N = test_data["means"].shape[-2]
#     C = test_data["viewmats"].shape[-3]
#     I = math.prod(batch_dims) * C
#     test_data.update(
#         {
#             "colors": torch.rand(C, N, channels, device=device),
#             "backgrounds": torch.rand((C, channels), device=device) * 0.0,
#         }
#     )
#     # test_data = expand(test_data, batch_dims)
#     Ks = test_data["Ks"]
#     viewmats = test_data["viewmats"]
#     height = test_data["height"]
#     width = test_data["width"]
#     quats = test_data["quats"]
#     scales = test_data["scales"]
#     means = test_data["means"]
#     opacities = test_data["opacities"]
#     colors = test_data["colors"]
#     backgrounds = test_data["backgrounds"]

#     covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

#     # Project Gaussians to 2D
#     radii, means2d, depths, conics, compensations = fully_fused_projection(
#         means, covars, None, None, viewmats, Ks, width, height
#     )
#     opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))

#     # Identify intersecting tiles
#     tile_size = 16 if channels <= 32 else 4
#     tile_width = math.ceil(width / float(tile_size))
#     tile_height = math.ceil(height / float(tile_size))
#     tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
#         means2d, radii, depths, tile_size, tile_width, tile_height
#     )
#     isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
#     isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

#     means2d.requires_grad = True
#     conics.requires_grad = True
#     colors.requires_grad = True
#     opacities.requires_grad = True
#     backgrounds.requires_grad = True

#     # forward
#     render_colors, render_alphas = rasterize_to_pixels(
#         means2d,
#         conics,
#         colors,
#         opacities,
#         width,
#         height,
#         tile_size,
#         isect_offsets,
#         flatten_ids,
#         # backgrounds=backgrounds,
#     )

#     isect_offsets = isect_offsets.reshape(-1)
#     isect_prefix_sum = torch.empty_like(isect_offsets)
#     isect_prefix_sum[:-1] = isect_offsets[1:]
#     isect_prefix_sum[-1] = flatten_ids.numel()

#     for _ in tqdm.trange(10000 if benchmark else 1, disable=not benchmark, desc="Forward"):
#         render_colors_, render_alphas_, render_last_ids_ = _C.image_gaussian_rasterize_forward(
#             opacities.reshape(-1).contiguous(),
#             means2d.reshape(-1, 2).contiguous(),
#             conics.reshape(-1, 3).contiguous(),
#             colors.reshape(-1, channels).contiguous(),
#             I,
#             width,
#             height,
#             tile_size, # tile_width
#             tile_size, # tile_height
#             flatten_ids.reshape(-1).to(torch.uint32).contiguous(),   
#             isect_prefix_sum.reshape(-1).to(torch.uint32).contiguous(),
#         )

#     torch.testing.assert_close(render_colors, render_colors_)
#     torch.testing.assert_close(render_alphas, render_alphas_)

#     # jvp
#     opacities_with_grad = torch.stack([opacities, torch.rand_like(opacities)], dim=-1)
#     means2d_with_grad = torch.stack([means2d, torch.rand_like(means2d)], dim=-1)
#     conics_with_grad = torch.stack([conics, torch.rand_like(conics)], dim=-1)
#     colors_with_grad = torch.stack([colors, torch.rand_like(colors)], dim=-1)
#     for _ in tqdm.trange(10000 if benchmark else 1, disable=not benchmark, desc="JVP"):
#         render_colors_, render_alphas_, render_last_ids_ = _C.image_gaussian_rasterize_jvp(
#             opacities_with_grad.reshape(-1, 2).contiguous(),
#             means2d_with_grad.reshape(-1, 2, 2).contiguous(),
#             conics_with_grad.reshape(-1, 3, 2).contiguous(),
#             colors_with_grad.reshape(-1, channels, 2).contiguous(),
#             I,
#             width,
#             height,
#             tile_size, # tile_width
#             tile_size, # tile_height
#             flatten_ids.reshape(-1).to(torch.uint32).contiguous(),   
#             isect_prefix_sum.reshape(-1).to(torch.uint32).contiguous(),
#         )
#     # render_colors_ shape: [n_images, image_height, image_width, channels, 2]
#     # render_alphas_ shape: [n_images, image_height, image_width, 1, 2]

#     torch.testing.assert_close(render_colors, render_colors_[..., 0])
#     torch.testing.assert_close(render_alphas, render_alphas_[..., 0])

#     # # backward
#     # v_render_colors = torch.randn_like(render_colors)
#     # v_render_alphas = torch.randn_like(render_alphas)

#     # v_means2d, v_conics, v_colors, v_opacities = torch.autograd.grad(
#     #     (render_colors * v_render_colors).sum()
#     #     + (render_alphas * v_render_alphas).sum(),
#     #     (means2d, conics, colors, opacities),
#     # )

#     # v_opacities_, v_means2d_, v_conics_, v_colors_ = _C.image_gaussian_rasterize_backward(
#     #     opacities.reshape(-1).contiguous(),
#     #     means2d.reshape(-1, 2).contiguous(),
#     #     conics.reshape(-1, 3).contiguous(),
#     #     colors.reshape(-1, channels).contiguous(),
#     #     I,
#     #     width,
#     #     height,
#     #     tile_size, # tile_width
#     #     tile_size, # tile_height
#     #     flatten_ids.reshape(-1).to(torch.uint32).contiguous(),   
#     #     isect_prefix_sum.reshape(-1).to(torch.uint32).contiguous(),
#     #     render_last_ids_.contiguous(),
#     #     render_alphas_.contiguous(),
#     #     v_render_alphas,
#     #     v_render_colors,
#     # )
#     # torch.cuda.synchronize()

#     # torch.testing.assert_close(v_opacities, v_opacities_)
#     # torch.testing.assert_close(v_means2d, v_means2d_)
#     # torch.testing.assert_close(v_conics, v_conics_)
#     # torch.testing.assert_close(v_colors, v_colors_)


#     # torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
#     # torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
#     # torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
#     # torch.testing.assert_close(v_opacities, _v_opacities, rtol=8e-3, atol=6e-3)
#     # torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    rasterizer_test_data = create_rasterizer_test_data()
    test_rasterize_to_pixels(rasterizer_test_data=rasterizer_test_data)

    # test_data = create_test_data()
    # test_rasterize_to_pixels(test_data=test_data, channels=3, batch_dims=(), benchmark=False)
