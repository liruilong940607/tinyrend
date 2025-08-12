#pragma once

#include <ATen/core/Tensor.h>

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
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> image_gaussian_rasterize_jvp(
    // Primitives
    const at::Tensor opacities, // [n_primitives, 2]
    const at::Tensor means2d,   // [n_primitives, 2, 2]
    const at::Tensor conics,    // [n_primitives, 3, 2]
    const at::Tensor features,  // [n_primitives, channels, 2]

    // Images
    const int64_t n_images,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_width,
    const int64_t tile_height,

    // Intersections
    const at::Tensor isect_primitive_ids,      // [n_isects]
    const at::Tensor isect_prefix_sum_per_tile // [n_tiles]
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_gaussian_rasterize_backward(
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
    const at::Tensor isect_primitive_ids,       // [n_isects]
    const at::Tensor isect_prefix_sum_per_tile, // [n_tiles]

    // Forward Outputs
    const at::Tensor render_last_index, // [n_images, image_height, image_width, 1]
    const at::Tensor render_alpha,      // [n_images, image_height, image_width, 1]

    // Gradients for Forward Outputs
    const at::Tensor v_render_alpha,  // [n_images, image_height, image_width, 1]
    const at::Tensor v_render_feature // [n_images, image_height, image_width, channels]
);

} // namespace cugsplat