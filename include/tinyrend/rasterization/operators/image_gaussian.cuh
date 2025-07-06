#pragma once

#include <cooperative_groups.h>
#include <cstdint>

#include "tinyrend/common/vec.h"
#include "tinyrend/rasterization/base.cuh"
#include "tinyrend/util/warp.cuh"

namespace tinyrend::rasterization {

namespace cg = cooperative_groups;

struct EvaluateLightAttenuationContext {
    float alpha;
    float vis;
    fvec3 conic;
    float dx;
    float dy;
    float maximum_alpha;
};

inline __device__ auto evaluate_light_attenuation_forward(
    const float opacity,
    const fvec2 mean,
    const fvec3 conic,
    const float pixel_x,
    const float pixel_y,
    const float maximum_alpha
) -> std::pair<float, EvaluateLightAttenuationContext> {
    auto const dx = pixel_x - mean[0];
    auto const dy = pixel_y - mean[1];
    auto const sigma =
        0.5f * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy;
    auto const vis = __expf(-sigma);
    auto const alpha = opacity * vis;
    auto const output = min(alpha, maximum_alpha);
    return {
        output,
        EvaluateLightAttenuationContext{alpha, vis, conic, dx, dy, maximum_alpha}
    };
}

inline __device__ auto evaluate_light_attenuation_backward(
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

template <size_t FEATURE_DIM>
struct ImageGaussianRasterizeKernelForwardOperator
    : BaseRasterizeKernelOperator<
          ImageGaussianRasterizeKernelForwardOperator<FEATURE_DIM>> {

    using FeatureType = fvec<FEATURE_DIM>;

    // Inputs
    float *opacity_ptr; // [N, 1]
    fvec2 *mean_ptr;    // [N, 2]
    fvec3 *conic_ptr;   // [N, 3]
    FeatureType
        *feature_ptr; // [N, FEATURE_DIM] (e.g., 3 for RGB or 256 for neural features)

    // Outputs
    int32_t *render_last_index_ptr; // [n_images, image_height, image_width, 1]
    float *render_alpha_ptr;        // [n_images, image_height, image_width, 1]
    FeatureType
        *render_feature_ptr; // [n_images, image_height, image_width, FEATURE_DIM]

    // Internal variables
    FeatureType _expected_feature = {0.0f}; // buffer for feature accumulation
    float _T = 1.0f;                        // current transmittance
    int32_t _last_index = -1; // the index of intersections ([n_isects]) for the last
                              // one being rasterized. -1 means no intersection.

    // Configs
    const float skip_if_alpha_smaller_than = 1.0f / 255.0f;
    const float maximum_alpha = 0.999f; // For backward numerical stability.
    const float stop_if_next_trans_smaller_than =
        1e-4f; // For backward numerical stability.

    static inline __host__ auto sm_size_per_primitive_impl() -> uint32_t {
        // cache the opacity, mean, conic, and primitive_id
        return sizeof(float) + sizeof(fvec2) + sizeof(fvec3) + sizeof(uint32_t);
    }

    inline __device__ auto initialize_impl() -> bool { return true; }

    inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // cache data to shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_mean_ptr =
            reinterpret_cast<fvec2 *>(&sm_opacity_ptr[this->n_threads_per_block]);
        auto const sm_conic_ptr =
            reinterpret_cast<fvec3 *>(&sm_mean_ptr[this->n_threads_per_block]);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_conic_ptr[this->n_threads_per_block]);
        sm_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
        sm_mean_ptr[this->thread_rank] = this->mean_ptr[primitive_id];
        sm_conic_ptr[this->thread_rank] = this->conic_ptr[primitive_id];
        sm_primitive_id_ptr[this->thread_rank] = primitive_id;
    }

    template <class WarpT>
    inline __device__ auto
    rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
        // load data from shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_mean_ptr =
            reinterpret_cast<fvec2 *>(&sm_opacity_ptr[this->n_threads_per_block]);
        auto const sm_conic_ptr =
            reinterpret_cast<fvec3 *>(&sm_mean_ptr[this->n_threads_per_block]);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_conic_ptr[this->n_threads_per_block]);
        auto const opacity = sm_opacity_ptr[t];
        auto const mean = sm_mean_ptr[t];
        auto const conic = sm_conic_ptr[t];

        // compute the light attenuation
        auto const &[alpha, _ctx] = evaluate_light_attenuation_forward(
            opacity, mean, conic, this->pixel_x, this->pixel_y, this->maximum_alpha
        );
        // skip if the alpha is smaller than the threshold
        if (alpha < this->skip_if_alpha_smaller_than) {
            return false; // continue
        }

        // check if I should stop
        auto const next_T = this->_T * (1.0f - alpha);
        if (next_T < this->stop_if_next_trans_smaller_than) {
            return true; // terminate
        }

        // weights for expectation calculation
        auto const weight = alpha * this->_T;

        // accumulate the expectation of the feature
        // Note(ruilong): we directly load feature from global memory here. Not sure if
        // this is better than prefetching it to shared memory and loading it from
        // there.
        auto const primitive_id = sm_primitive_id_ptr[t];
        this->_expected_feature += weight * this->feature_ptr[primitive_id];

        // update the transmittance
        this->_T = next_T;

        // the global index in all intersections ([n_isects]).
        this->_last_index = batch_start + t;

        // Return whether we want to terminate the rasterization process.
        return false;
    }

    inline __device__ auto pixel_postprocess_impl() -> void {
        // write to the output buffer
        auto const offset_pixel =
            this->image_id * this->image_height * this->image_width + this->pixel_id;
        this->render_alpha_ptr[offset_pixel] = 1.0f - this->_T;
        this->render_last_index_ptr[offset_pixel] = this->_last_index;
        this->render_feature_ptr[offset_pixel] = this->_expected_feature;
    }
};

template <size_t FEATURE_DIM>
struct ImageGaussianRasterizeKernelBackwardOperator
    : BaseRasterizeKernelOperator<
          ImageGaussianRasterizeKernelBackwardOperator<FEATURE_DIM>> {

    using FeatureType = fvec<FEATURE_DIM>;

    // Forward Inputs
    float *opacity_ptr; // [N, 1]
    fvec2 *mean_ptr;    // [N, 2]
    fvec3 *conic_ptr;   // [N, 3]
    FeatureType
        *feature_ptr; // [N, FEATURE_DIM] (e.g., 3 for RGB or 256 for neural features)

    // Forward Outputs
    int32_t *render_last_index_ptr; // [n_images, image_height, image_width, 1]
    float *render_alpha_ptr;        // [n_images, image_height, image_width, 1]

    // Gradients for Forward Outputs
    float *v_render_alpha_ptr; // [n_images, image_height, image_width, 1]
    FeatureType
        *v_render_feature_ptr; // [n_images, image_height, image_width, FEATURE_DIM]

    // Gradients for Forward Inputs
    float *v_opacity_ptr;       // [N, 1]
    fvec2 *v_mean_ptr;          // [N, 2]
    fvec3 *v_conic_ptr;         // [N, 3]
    FeatureType *v_feature_ptr; // [N, FEATURE_DIM]

    // Internal variables
    float _T_final;                // final transmittance
    float _T;                      // current transmittance (from back to front)
    float _v_render_alpha;         // dl/d_render_alpha for this pixel
    FeatureType _v_render_feature; // dl/d_render_feature for this pixel
    FeatureType _expected_feature = {0.0f}; // buffer for feature accumulation

    // Configs
    const float skip_if_alpha_smaller_than = 1.0f / 255.0f;
    const float maximum_alpha = 0.999f; // For backward numerical stability.
    const float stop_if_next_trans_smaller_than =
        1e-4f; // For backward numerical stability.

    static inline __host__ auto sm_size_per_primitive_impl() -> uint32_t {
        // cache the opacity, mean, conic, primitive_id, and feature
        return sizeof(float) + sizeof(fvec2) + sizeof(fvec3) + sizeof(uint32_t) +
               sizeof(FeatureType);
    }

    inline __device__ auto initialize_impl() -> bool {
        // load the gradient for this pixel
        auto const offset_pixel =
            this->image_id * this->image_height * this->image_width + this->pixel_id;
        this->_v_render_alpha = this->v_render_alpha_ptr[offset_pixel];
        this->_v_render_feature = this->v_render_feature_ptr[offset_pixel];

        // load the initial transmittance as remaining transmittance
        this->_T_final = 1.0f - this->render_alpha_ptr[offset_pixel];
        this->_T = this->_T_final;
        return true;
    }

    inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // cache data to shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_mean_ptr =
            reinterpret_cast<fvec2 *>(&sm_opacity_ptr[this->n_threads_per_block]);
        auto const sm_conic_ptr =
            reinterpret_cast<fvec3 *>(&sm_mean_ptr[this->n_threads_per_block]);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_conic_ptr[this->n_threads_per_block]);
        auto const sm_feature_ptr = reinterpret_cast<FeatureType *>(
            &sm_primitive_id_ptr[this->n_threads_per_block]
        );
        sm_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
        sm_mean_ptr[this->thread_rank] = this->mean_ptr[primitive_id];
        sm_conic_ptr[this->thread_rank] = this->conic_ptr[primitive_id];
        sm_primitive_id_ptr[this->thread_rank] = primitive_id;
        sm_feature_ptr[this->thread_rank] = this->feature_ptr[primitive_id];
    }

    template <class WarpT>
    inline __device__ auto
    rasterize_impl(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
        // load data from shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_mean_ptr =
            reinterpret_cast<fvec2 *>(&sm_opacity_ptr[this->n_threads_per_block]);
        auto const sm_conic_ptr =
            reinterpret_cast<fvec3 *>(&sm_mean_ptr[this->n_threads_per_block]);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_conic_ptr[this->n_threads_per_block]);
        auto const sm_feature_ptr = reinterpret_cast<FeatureType *>(
            &sm_primitive_id_ptr[this->n_threads_per_block]
        );
        auto const opacity = sm_opacity_ptr[t];
        auto const mean = sm_mean_ptr[t];
        auto const conic = sm_conic_ptr[t];

        // compute the light attenuation
        auto const &[alpha, ela_ctx] = evaluate_light_attenuation_forward(
            opacity, mean, conic, this->pixel_x, this->pixel_y, this->maximum_alpha
        );

        // skip if the alpha is smaller than the threshold
        if (alpha < this->skip_if_alpha_smaller_than) {
            return false; // continue
        }

        // check if I should stop
        auto const next_T = this->_T * (1.0f - alpha);
        if (next_T < this->stop_if_next_trans_smaller_than) {
            return true; // terminate
        }

        // weights for expectation calculation
        auto const weight = alpha * this->_T;

        // compute the gradient
        auto const ra = 1.0f / (1.0f - alpha);
        this->_T *= ra;
        auto v_alpha = this->_T_final * ra * this->_v_render_alpha;

        // accumulate the expectation of the feature
        auto const feature = sm_feature_ptr[t];
        FeatureType v_feature = weight * this->_v_render_feature;
        this->_expected_feature += weight * feature;
        v_alpha += ((feature * this->_T - this->_expected_feature * ra) *
                    this->_v_render_feature)
                       .sum();

        // compute the gradient of the `evaluate_light_attenuation`
        auto v_mean = fvec2{};
        auto v_conic = fvec3{};
        auto v_opacity = 0.0f;
        evaluate_light_attenuation_backward(
            ela_ctx, v_alpha, v_opacity, v_mean, v_conic
        );

        // reduce the gradient over the warp [faster than atomicAdd to global memory]
        tinyrend::warp::warpSum(v_opacity, warp);
        tinyrend::warp::warpSum(v_mean, warp);
        tinyrend::warp::warpSum(v_conic, warp);
        tinyrend::warp::warpSum(v_feature, warp);

        // first thread in the warp writes the gradient to global memory.
        if (warp.thread_rank() == 0) {
            auto const primitive_id = sm_primitive_id_ptr[t];
            float *v_opacity_ptr = (float *)this->v_opacity_ptr;
            atomicAdd(v_opacity_ptr + primitive_id, v_alpha);

            float *v_mean_ptr = (float *)this->v_mean_ptr;
            atomicAdd(v_mean_ptr + primitive_id * 2, v_mean[0]);
            atomicAdd(v_mean_ptr + primitive_id * 2 + 1, v_mean[1]);

            float *v_conic_ptr = (float *)this->v_conic_ptr;
            atomicAdd(v_conic_ptr + primitive_id * 3, v_conic[0]);
            atomicAdd(v_conic_ptr + primitive_id * 3 + 1, v_conic[1]);
            atomicAdd(v_conic_ptr + primitive_id * 3 + 2, v_conic[2]);

            float *v_feature_ptr = (float *)this->v_feature_ptr;
#pragma unroll
            for (size_t i = 0; i < FEATURE_DIM; i++) {
                atomicAdd(v_feature_ptr + primitive_id * FEATURE_DIM + i, v_feature[i]);
            }
        }

        // Return whether we want to terminate the rasterization process.
        return false;
    }

    inline __device__ auto pixel_postprocess_impl() -> void {
        // Do nothing
    }
};

} // namespace tinyrend::rasterization