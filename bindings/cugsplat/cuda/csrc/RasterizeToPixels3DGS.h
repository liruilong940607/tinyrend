#pragma once

#include <cmath>
#include <cstdint>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/scalar_grad.h"
#include "tinyrend/common/vec.h"

namespace cugsplat {

using namespace tinyrend;

#ifdef __CUDA_ARCH__ // Shared between forward and backward cuda kernels

template <typename ScalarType, typename Vec3Type>
struct EvaluateLightAttenuationContext {
    ScalarType alpha;
    ScalarType vis;
    Vec3Type conic;
    ScalarType dx;
    ScalarType dy;
    float maximum_alpha;
};

template <typename ScalarType, typename Vec2Type, typename Vec3Type>
inline TREND_HOST_DEVICE auto evaluate_light_attenuation_forward(
    const ScalarType opacity,
    const Vec2Type mean,
    const Vec3Type conic,
    const float pixel_x,
    const float pixel_y,
    const float maximum_alpha
) -> std::pair<ScalarType, EvaluateLightAttenuationContext<ScalarType, Vec3Type>> {
    // TODO(ruilong): do we really need to add 0.5f here?
    auto const dx = pixel_x + 0.5f - mean[0];
    auto const dy = pixel_y + 0.5f - mean[1];
    auto const sigma =
        0.5f * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy;
    ScalarType vis;
    if constexpr (std::is_same_v<ScalarType, float>) {
        vis = __expf(-sigma); // Device code: use fast CUDA intrinsic
    } else {
        vis = exp(-sigma); // Use standard exp
    }
    auto const alpha = opacity * vis;
    auto const output = fmin(alpha, maximum_alpha);
    return {
        output,
        EvaluateLightAttenuationContext<ScalarType, Vec3Type>{
            alpha, vis, conic, dx, dy, maximum_alpha
        }
    };
}

template <typename ScalarType, typename Vec2Type, typename Vec3Type>
inline TREND_HOST_DEVICE auto evaluate_light_attenuation_backward(
    // context from forward pass
    EvaluateLightAttenuationContext<ScalarType, Vec3Type> ctx,
    // gradient of outputs
    const ScalarType v_alpha,
    // gradients of inputs
    ScalarType &v_opacity,
    Vec2Type &v_mean,
    Vec3Type &v_conic
) -> void {
    if (ctx.alpha >= ctx.maximum_alpha) {
        return; // clip happens so no gradient
    }

    auto const v_sigma = -ctx.alpha * v_alpha;
    v_opacity += ctx.vis * v_alpha;
    v_mean += v_sigma * Vec2Type{
                            ctx.conic[0] * ctx.dx + ctx.conic[1] * ctx.dy,
                            ctx.conic[1] * ctx.dx + ctx.conic[2] * ctx.dy
                        };
    v_conic +=
        v_sigma *
        Vec3Type{0.5f * ctx.dx * ctx.dx, ctx.dx * ctx.dy, 0.5f * ctx.dy * ctx.dy};
}

#endif

template <size_t FEATURE_DIM, typename ScalarType, typename Vec2Type, typename Vec3Type>
void image_gaussian_rasterize_kernel_forward(
    // Primitives
    const size_t n_primitives,
    const ScalarType *__restrict__ opacity_ptr, // [n_primitives, 2]
    const Vec2Type *__restrict__ mean_ptr,      // [n_primitives, 2, 2]
    const Vec3Type *__restrict__ conic_ptr,     // [n_primitives, 3, 2]
    const vec<ScalarType, FEATURE_DIM>
        *__restrict__ feature_ptr, // [n_primitives, FEATURE_DIM, 2]

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
    ScalarType
        *__restrict__ render_alpha_ptr, // [n_images, image_height, image_width, 1, 2]
    vec<ScalarType, FEATURE_DIM>
        *__restrict__ render_feature_ptr // [n_images, image_height,
                                         // image_width, FEATURE_DIM, 2]
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
