import os
import sys
import pytest
import math

import torch
import torch.autograd.forward_ad as fwAD


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BINDINGS_DIR = os.path.join(CURRENT_DIR, "..", "..", "bindings")
sys.path.append(BINDINGS_DIR)
from cugsplat.cuda.warppers.projection import projection_fused_ewa_3dgs
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
        "colors": colors,  # [N, 3]
        "viewmats": viewmats,  # [C, 4, 4]
        "Ks": Ks,  # [C, 3, 3]
        "width": width,
        "height": height,
    }



@pytest.fixture
def test_data():
    return create_test_data()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("benchmark", [False])
def test_projection_fused_ewa_3dgs_fwd(test_data: dict, benchmark: bool):
    from gsplat.cuda._wrapper import fully_fused_projection as fully_fused_projection_gsplat

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    viewmats_R = viewmats[:, :3, :3]
    viewmats_t = viewmats[:, :3, 3]
    resolution = (test_data["width"], test_data["height"])
    focal_length = (Ks[0, 0, 0].item(), Ks[0, 1, 1].item())
    principal_point = (Ks[0, 0, 2].item(), Ks[0, 1, 2].item())
    near_plane = 0.1
    far_plane = 10000.0
    eps2d = 0.3

    means = test_data["means"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    opacities = test_data["opacities"]

    means.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    opacities.requires_grad = True

    # forward
    for _ in tqdm.trange(
        500000 if benchmark else 1, desc="gsplat:forward", disable=not benchmark
    ):
        radii, means2d, depths, conics, _ = fully_fused_projection_gsplat(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            resolution[0], # image_width
            resolution[1], # image_height
            near_plane=near_plane,
            far_plane=far_plane,
            eps2d=eps2d,
            opacities=opacities,
        )

    for _ in tqdm.trange(
        500000 if benchmark else 1, desc="cugsplat:forward", disable=not benchmark
    ):
        radii_, means2d_, depths_, conics_ = projection_fused_ewa_3dgs(
            resolution,
            principal_point,
            focal_length,
            viewmats_R,
            viewmats_t,
            near_plane,
            far_plane,
            eps2d,
            opacities,
            means,
            quats,
            scales,
        )
    
    valid = (radii > 0).all(dim=-1) & (radii_ > 0).all(dim=-1)

    torch.testing.assert_close(radii, radii_, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], means2d_[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], depths_[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], conics_[valid], rtol=1e-4, atol=1e-4)

    # # backward
    # v_render_colors = torch.rand_like(render_colors)
    # v_render_alphas = torch.rand_like(render_alphas)

    # for _ in tqdm.trange(
    #     5000 if benchmark else 1, desc="gsplat:backward", disable=not benchmark
    # ):
    #     v_means2d, v_conics, v_colors, v_opacities = torch.autograd.grad(
    #         (render_colors * v_render_colors).sum()
    #         + (render_alphas * v_render_alphas).sum(),
    #         (means2d, conics, colors, opacities),
    #         create_graph=True,
    #     )

    # for _ in tqdm.trange(
    #     5000 if benchmark else 1, desc="cugsplat:backward", disable=not benchmark
    # ):
    #     v_means2d_, v_conics_, v_colors_, v_opacities_ = torch.autograd.grad(
    #         (render_output.colors * v_render_colors).sum()
    #         + (render_output.alphas * v_render_alphas).sum(),
    #         (means2d, conics, colors, opacities),
    #         create_graph=True,
    #     )

    # torch.testing.assert_close(v_means2d, v_means2d_)
    # torch.testing.assert_close(v_conics, v_conics_, rtol=1e-4, atol=1e-4)
    # torch.testing.assert_close(v_colors, v_colors_)
    # torch.testing.assert_close(v_opacities, v_opacities_)




if __name__ == "__main__":
    test_data = create_test_data()

    test_projection_fused_ewa_3dgs_fwd(test_data=test_data, benchmark=True)
