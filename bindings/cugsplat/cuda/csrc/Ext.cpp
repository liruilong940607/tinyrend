#include <torch/extension.h>

#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "image_gaussian_rasterize_forward", &cugsplat::image_gaussian_rasterize_forward
    );
    m.def("image_gaussian_rasterize_jvp", &cugsplat::image_gaussian_rasterize_jvp);
    m.def(
        "image_gaussian_rasterize_backward",
        &cugsplat::image_gaussian_rasterize_backward
    );

    m.def("projection_fused_ewa_3dgs_fwd", &cugsplat::projection_fused_ewa_3dgs_fwd);
    m.def("projection_fused_ewa_3dgs_bwd", &cugsplat::projection_fused_ewa_3dgs_bwd);
}
