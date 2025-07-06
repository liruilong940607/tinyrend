#pragma once

#include <cmath>
#include <cstdint>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/vec.h"

namespace cugsplat {

using namespace tinyrend;

#ifdef __CUDA_ARCH__ // Shared between forward and backward cuda kernels

struct EvaluateLightAttenuationContext {
    float alpha;
    float vis;
    fvec3 conic;
    float dx;
    float dy;
    float maximum_alpha;
};

inline TREND_HOST_DEVICE auto evaluate_light_attenuation_forward(
    const float opacity,
    const fvec2 mean,
    const fvec3 conic,
    const float pixel_x,
    const float pixel_y,
    const float maximum_alpha
) -> std::pair<float, EvaluateLightAttenuationContext> {
    // TODO(ruilong): do we really need to add 0.5f here?
    auto const dx = pixel_x + 0.5f - mean[0];
    auto const dy = pixel_y + 0.5f - mean[1];
    auto const sigma =
        0.5f * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy;
    auto const vis = __expf(-sigma); // Device code: use fast CUDA intrinsic
    auto const alpha = opacity * vis;
    auto const output = min(alpha, maximum_alpha);
    return {
        output,
        EvaluateLightAttenuationContext{alpha, vis, conic, dx, dy, maximum_alpha}
    };
}

inline TREND_HOST_DEVICE auto evaluate_light_attenuation_backward(
    // context from forward pass
    EvaluateLightAttenuationContext ctx,
    // gradient of outputs
    const float v_alpha,
    // gradients of inputs
    float &v_opacity,
    fvec2 &v_mean,
    fvec3 &v_conic
) -> void {
    if (ctx.alpha >= ctx.maximum_alpha) {
        return; // clip happens so no gradient
    }

    auto const v_sigma = -ctx.alpha * v_alpha;
    v_opacity += ctx.vis * v_alpha;
    v_mean += v_sigma * fvec2{
                            ctx.conic[0] * ctx.dx + ctx.conic[1] * ctx.dy,
                            ctx.conic[1] * ctx.dx + ctx.conic[2] * ctx.dy
                        };
    v_conic += v_sigma *
               fvec3{0.5f * ctx.dx * ctx.dx, ctx.dx * ctx.dy, 0.5f * ctx.dy * ctx.dy};
}

#endif

template <size_t FEATURE_DIM>
void image_gaussian_rasterize_kernel_forward(
    // Primitives
    const size_t n_primitives,
    const float *__restrict__ opacity_ptr,       // [n_primitives]
    fvec2 *__restrict__ mean_ptr,                // [n_primitives, 2]
    fvec3 *__restrict__ conic_ptr,               // [n_primitives, 3]
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
    int32_t
        *__restrict__ render_last_index_ptr, // [n_images, image_height, image_width, 1]
    float *__restrict__ render_alpha_ptr,    // [n_images, image_height, image_width, 1]
    fvec<FEATURE_DIM> *__restrict__ render_feature_ptr // [n_images, image_height,
                                                       // image_width, FEATURE_DIM]
);

template <size_t FEATURE_DIM>
void image_gaussian_rasterize_kernel_backward(
    // Forward Inputs
    const size_t n_primitives,
    const float *__restrict__ opacity_ptr,             // [n_primitives]
    const fvec2 *__restrict__ mean_ptr,                // [n_primitives, 2]
    const fvec3 *__restrict__ conic_ptr,               // [n_primitives, 3]
    const fvec<FEATURE_DIM> *__restrict__ feature_ptr, // [n_primitives, FEATURE_DIM]

    // Images
    const size_t n_images,
    const size_t image_height,
    const size_t image_width,
    const size_t tile_width,
    const size_t tile_height,

    // Isect info
    const uint32_t *__restrict__ isect_primitive_ids,       // [n_isects]
    const uint32_t *__restrict__ isect_prefix_sum_per_tile, // [n_tiles]

    // Forward Outputs
    const int32_t
        *__restrict__ render_last_index_ptr, // [n_images, image_height, image_width, 1]
    const float
        *__restrict__ render_alpha_ptr, // [n_images, image_height, image_width, 1]

    // Gradients for Forward Outputs
    const float
        *__restrict__ v_render_alpha_ptr, // [n_images, image_height, image_width, 1]
    const fvec<FEATURE_DIM>
        *__restrict__ v_render_feature_ptr, // [n_images, image_height, image_width,
                                            // FEATURE_DIM]

    // Gradients for Forward Inputs
    float *__restrict__ v_opacity_ptr,            // [N, 1]
    fvec2 *__restrict__ v_mean_ptr,               // [N, 2]
    fvec3 *__restrict__ v_conic_ptr,              // [N, 3]
    fvec<FEATURE_DIM> *__restrict__ v_feature_ptr // [N, FEATURE_DIM]
);

} // namespace cugsplat
