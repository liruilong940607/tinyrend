#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include "Common.h"
#include "Ops.h"
#include "RasterizeToPixels3DGS.h"

namespace cugsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor> image_gaussian_rasterize_forward(
    // Primitives
    const at::Tensor opacities, // [n_primitives]
    const at::Tensor means2d,   // [n_primitives, 2]
    const at::Tensor conics,    // [n_primitives, 3]
    const at::Tensor features,  // [n_primitives, channels]

    // Images
    const int64_t n_images,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_width,
    const int64_t tile_height,

    // Intersections
    const at::Tensor isect_primitive_ids,      // [n_isects]
    const at::Tensor isect_prefix_sum_per_tile // [n_tiles]
) {
    DEVICE_GUARD(means2d);

    // Check inputs
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(features);
    CHECK_INPUT(opacities);
    CHECK_INPUT(isect_primitive_ids);
    CHECK_INPUT(isect_prefix_sum_per_tile);

    const int64_t n_primitives = means2d.size(0);
    const int64_t channels = features.size(-1);

    // Create outputs
    at::Tensor render_last_index = at::empty(
        {n_images, image_height, image_width, 1}, means2d.options().dtype(at::kInt)
    );
    at::Tensor render_alpha =
        at::empty({n_images, image_height, image_width, 1}, means2d.options());
    at::Tensor render_feature =
        at::empty({n_images, image_height, image_width, channels}, means2d.options());

    // Launch kernel
#define __LAUNCH_KERNEL__(DIM)                                                         \
    case DIM:                                                                          \
        image_gaussian_rasterize_kernel_forward<DIM>(                                  \
            n_primitives,                                                              \
            opacities.data_ptr<float>(),                                               \
            reinterpret_cast<fvec2 *>(means2d.data_ptr<float>()),                      \
            reinterpret_cast<fvec3 *>(conics.data_ptr<float>()),                       \
            reinterpret_cast<fvec<DIM> *>(features.data_ptr<float>()),                 \
            n_images,                                                                  \
            image_height,                                                              \
            image_width,                                                               \
            tile_width,                                                                \
            tile_height,                                                               \
            isect_primitive_ids.data_ptr<uint32_t>(),                                  \
            isect_prefix_sum_per_tile.data_ptr<uint32_t>(),                            \
            render_last_index.data_ptr<int32_t>(),                                     \
            render_alpha.data_ptr<float>(),                                            \
            reinterpret_cast<fvec<DIM> *>(render_feature.data_ptr<float>())            \
        );                                                                             \
        break;

    switch (channels) {
        __LAUNCH_KERNEL__(3)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef __LAUNCH_KERNEL__

    return {render_feature, render_alpha, render_last_index};
}

} // namespace cugsplat