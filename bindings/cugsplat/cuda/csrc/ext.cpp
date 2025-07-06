#include <torch/extension.h>

#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("image_gaussian_rasterize_forward", &cugsplat::image_gaussian_rasterize_forward);
}
