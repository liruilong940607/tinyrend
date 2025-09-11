#pragma once

#include <ATen/core/Tensor.h>

namespace cugsplat {

/********************* Image Gaussian Rasterize *********************/

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

/********************* Projection Fused EWA 3DGS *********************/

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_fused_ewa_3dgs_fwd(
    // Perfect Pinhole Camera: All cameras share the same intrinsic.
    const uint32_t image_width,
    const uint32_t image_height,
    const float principal_point_x,
    const float principal_point_y,
    const float focal_length_x,
    const float focal_length_y,

    const at::Tensor viewmat_R, // [C, 3, 3]
    const at::Tensor viewmat_t, // [C, 3]

    const float near_plane,
    const float far_plane,
    const float eps2d,

    // Gaussians
    const at::Tensor opacities, // [N]
    const at::Tensor means,     // [N, 3]
    const at::Tensor quats,     // [N, 4]
    const at::Tensor scales     // [N, 3]
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> projection_fused_ewa_3dgs_bwd(
    // Perfect Pinhole Camera: All cameras share the same intrinsic.
    const uint32_t image_width,
    const uint32_t image_height,
    const float principal_point_x,
    const float principal_point_y,
    const float focal_length_x,
    const float focal_length_y,

    const at::Tensor viewmat_R, // [C, 3, 3]
    const at::Tensor viewmat_t, // [C, 3]

    // Gaussians
    const at::Tensor means,  // [N, 3]
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]

    // Forward Outputs
    const at::Tensor radii, // [C, N, 2]
    const at::Tensor conic, // [C, N, 3]

    // Gradient of Forward Outputs
    const at::Tensor v_means2d, // [C, N, 2]
    const at::Tensor v_depth,   // [C, N]
    const at::Tensor v_conic    // [C, N, 3]
);

} // namespace cugsplat