#pragma once

#include <cstdint>

#include "tinyrend/common/vec.h"

namespace cugsplat {

using namespace tinyrend;

template <size_t FEATURE_DIM>
void image_gaussian_rasterize_kernel_forward(
    // Primitives
    const size_t n_primitives,
    const float *__restrict__ opacity_ptr, // [n_primitives]
    fvec2 *__restrict__ mean_ptr, // [n_primitives, 2]
    fvec3 *__restrict__ conic_ptr, // [n_primitives, 3]
    fvec<FEATURE_DIM> *__restrict__ feature_ptr, // [n_primitives, FEATURE_DIM]

    // Images
    const size_t n_images,
    const size_t image_height,
    const size_t image_width,
    const size_t tile_width,
    const size_t tile_height,

    // Isect info
    const uint32_t *__restrict__ isect_primitive_ids,       // [n_isects]
    const uint32_t *__restrict__ isect_prefix_sum_per_tile, // [n_tiles]

    // Outputs
    int32_t *__restrict__ render_last_index_ptr, // [n_images, image_height, image_width, 1]
    float *__restrict__ render_alpha_ptr, // [n_images, image_height, image_width, 1]
    fvec<FEATURE_DIM> *__restrict__ render_feature_ptr // [n_images, image_height, image_width, FEATURE_DIM]
);

} // namespace cugsplat
