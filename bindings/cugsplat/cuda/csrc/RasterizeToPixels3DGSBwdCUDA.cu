#include <cooperative_groups.h>
#include <cstdint>

#include "RasterizeToPixels3DGS.h"
#include "tinyrend/common/vec.h"
#include "tinyrend/rasterization/base.cuh"
#include "tinyrend/util/warp.cuh"

namespace cugsplat {

using namespace tinyrend;

namespace cg = cooperative_groups;

template <size_t FEATURE_DIM>
struct ImageGaussianRasterizeKernelBackwardOperator
    : tinyrend::rasterization::BaseRasterizeKernelOperator<
          ImageGaussianRasterizeKernelBackwardOperator<FEATURE_DIM>> {

    using FeatureType = fvec<FEATURE_DIM>;

    // Forward Inputs
    const float *__restrict__ opacity_ptr;       // [N, 1]
    const fvec2 *__restrict__ mean_ptr;          // [N, 2]
    const fvec3 *__restrict__ conic_ptr;         // [N, 3]
    const FeatureType *__restrict__ feature_ptr; // [N, FEATURE_DIM] (e.g., 3 for RGB or
                                                 // 256 for neural features)

    // Forward Outputs
    const int32_t
        *__restrict__ render_last_index_ptr; // [n_images, image_height, image_width, 1]
    const float
        *__restrict__ render_alpha_ptr; // [n_images, image_height, image_width, 1]

    // Gradients for Forward Outputs
    const float
        *__restrict__ v_render_alpha_ptr; // [n_images, image_height, image_width, 1]
    const FeatureType *__restrict__ v_render_feature_ptr; // [n_images, image_height,
                                                          // image_width, FEATURE_DIM]

    // Gradients for Forward Inputs
    float *__restrict__ v_opacity_ptr;       // [N, 1]
    fvec2 *__restrict__ v_mean_ptr;          // [N, 2]
    fvec3 *__restrict__ v_conic_ptr;         // [N, 3]
    FeatureType *__restrict__ v_feature_ptr; // [N, FEATURE_DIM]

    // Internal variables
    float _T_final;           // final transmittance
    float _T;                 // current transmittance (from back to front)
    int32_t _last_index;      // the index of intersections ([n_isects]) for the last
                              // one being rasterized. -1 means no intersection.
    int32_t _warp_last_index; // the maximum _last_index in all threads in the warp.
    float _v_render_alpha;    // dl/d_render_alpha for this pixel
    FeatureType _v_render_feature; // dl/d_render_feature for this pixel
    FeatureType _expected_feature =
        FeatureType::zero(); // buffer for feature accumulation

    // Configs
    const float skip_if_alpha_smaller_than = 1.0f / 255.0f;
    const float maximum_alpha = 0.999f; // For backward numerical stability.

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
        this->_last_index = this->render_last_index_ptr[offset_pixel];

        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        this->_warp_last_index =
            cg::reduce(warp, this->_last_index, cg::greater<int>());
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
    inline __device__ auto rasterize_impl(
        uint32_t batch_start, uint32_t t, WarpT &warp, bool &terminated
    ) -> bool {
        // If this GS is behind the maximum last rendered GS in the warp, then we know
        // this GS is not rendered by any pixel in the warp. So we can early return.
        if (batch_start + t > this->_warp_last_index) {
            return terminated;
        }

        // Flag to indicate if this GS is rendered to this pixel. If not we will skip
        // the gradient computation. Note: we do not do early return like we do in
        // forward pass because we need to call warpSum later so the thread needs to be
        // alive.
        auto maybe_rendered = true;

        if (terminated) {
            maybe_rendered = false;
        }

        // If this GS is behind the last rendered GS of this pixel, then we know it is
        // not rendered. But we still need to keep this thread alive as this GS might
        // contribute to other pixels in the warp.
        if (batch_start + t > this->_last_index) {
            maybe_rendered = false;
        }

        // Prepare gradients that will be accumulated over the warp.
        auto v_mean = fvec2{};
        auto v_conic = fvec3{};
        auto v_opacity = 0.0f;
        auto v_feature = FeatureType::zero();

        // Prepare pointers to shared memory.
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_mean_ptr =
            reinterpret_cast<fvec2 *>(&sm_opacity_ptr[this->n_threads_per_block]);
        auto const sm_conic_ptr =
            reinterpret_cast<fvec3 *>(&sm_mean_ptr[this->n_threads_per_block]);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_conic_ptr[this->n_threads_per_block]);

        if (maybe_rendered) {
            // load data from shared memory
            auto const opacity = sm_opacity_ptr[t];
            auto const mean = sm_mean_ptr[t];
            auto const conic = sm_conic_ptr[t];

            // compute the light attenuation
            auto const &[alpha, ela_ctx] = evaluate_light_attenuation_forward(
                opacity, mean, conic, this->pixel_x, this->pixel_y, this->maximum_alpha
            );

            // skip if the alpha is smaller than the threshold
            if (alpha < this->skip_if_alpha_smaller_than) {
                maybe_rendered = false;
            }

            if (maybe_rendered) {
                auto const sm_feature_ptr = reinterpret_cast<FeatureType *>(
                    &sm_primitive_id_ptr[this->n_threads_per_block]
                );
                auto const feature = sm_feature_ptr[t];

                // compute the gradient
                auto const ra = 1.0f / (1.0f - alpha);
                this->_T *= ra;
                auto v_alpha = this->_T_final * ra * this->_v_render_alpha;

                // weights for expectation calculation
                auto const weight = alpha * this->_T;

                // accumulate the expectation of the feature
                v_feature = weight * this->_v_render_feature;

                // The contribution of the feature to alpha:
                v_alpha += ((feature * this->_T - this->_expected_feature * ra) *
                            this->_v_render_feature)
                               .sum();
                this->_expected_feature += weight * feature;

                // compute the gradient of the `evaluate_light_attenuation`
                evaluate_light_attenuation_backward(
                    ela_ctx, v_alpha, v_opacity, v_mean, v_conic
                );
            }
        }

        // If this GS is not rendered to any pixel in the warp, we can early return.
        if (!warp.any(maybe_rendered)) {
            return terminated;
        }

        // reduce the gradient over the warp [faster than atomicAdd to global memory]
        tinyrend::warp::warpSum(v_opacity, warp);
        tinyrend::warp::warpSum(v_mean, warp);
        tinyrend::warp::warpSum(v_conic, warp);
        tinyrend::warp::warpSum(v_feature, warp);

        // first thread in the warp writes the gradient to global memory.
        if (warp.thread_rank() == 0) {
            // TODO(ruilong): make atomicAdd work for vec
            auto const primitive_id = sm_primitive_id_ptr[t];
            float *v_opacity_ptr = (float *)this->v_opacity_ptr;
            atomicAdd(v_opacity_ptr + primitive_id, v_opacity);

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
        // In backward pass we don't do early return so we maintain the `terminated`
        // flag.
        return terminated;
    }

    inline __device__ auto pixel_postprocess_impl() -> void {
        // Do nothing
    }
};

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
) {
    ImageGaussianRasterizeKernelBackwardOperator<FEATURE_DIM> op{};
    op.opacity_ptr = opacity_ptr;
    op.mean_ptr = mean_ptr;
    op.conic_ptr = conic_ptr;
    op.feature_ptr = feature_ptr;
    op.render_last_index_ptr = render_last_index_ptr;
    op.render_alpha_ptr = render_alpha_ptr;
    op.v_render_alpha_ptr = v_render_alpha_ptr;
    op.v_render_feature_ptr = v_render_feature_ptr;
    op.v_opacity_ptr = v_opacity_ptr;
    op.v_mean_ptr = v_mean_ptr;
    op.v_conic_ptr = v_conic_ptr;
    op.v_feature_ptr = v_feature_ptr;

    // Each block of threads cover a tile of the image. In total,
    // there are n_tiles_x * n_tiles_y * n_images blocks.
    auto const n_tiles_x = (image_width + tile_width - 1) / tile_width;
    auto const n_tiles_y = (image_height + tile_height - 1) / tile_height;

    dim3 threads(tile_width, tile_height, 1);
    dim3 grid(n_tiles_x, n_tiles_y, n_images);
    size_t sm_size = decltype(op)::sm_size_per_primitive() * threads.x * threads.y;

    if (cudaFuncSetAttribute(
            tinyrend::rasterization::rasterize_kernel<decltype(op)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sm_size
        ) != cudaSuccess) {
        throw std::runtime_error(
            "Failed to set maximum shared memory size (requested " +
            std::to_string(sm_size) + " bytes), try lowering tile_width or tile_height."
        );
    }

    tinyrend::rasterization::rasterize_kernel<<<grid, threads, sm_size>>>(
        op,
        image_height,
        image_width,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        true // reverse order
    );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(DIM)                                                                   \
    template void image_gaussian_rasterize_kernel_backward<DIM>(                       \
        const size_t n_primitives,                                                     \
        const float *__restrict__ opacity_ptr,                                         \
        const fvec2 *__restrict__ mean_ptr,                                            \
        const fvec3 *__restrict__ conic_ptr,                                           \
        const fvec<DIM> *__restrict__ feature_ptr,                                     \
        const size_t n_images,                                                         \
        const size_t image_height,                                                     \
        const size_t image_width,                                                      \
        const size_t tile_width,                                                       \
        const size_t tile_height,                                                      \
        const uint32_t *__restrict__ isect_primitive_ids,                              \
        const uint32_t *__restrict__ isect_prefix_sum_per_tile,                        \
        const int32_t *__restrict__ render_last_index_ptr,                             \
        const float *__restrict__ render_alpha_ptr,                                    \
        const float *__restrict__ v_render_alpha_ptr,                                  \
        const fvec<DIM> *__restrict__ v_render_feature_ptr,                            \
        float *__restrict__ v_opacity_ptr,                                             \
        fvec2 *__restrict__ v_mean_ptr,                                                \
        fvec3 *__restrict__ v_conic_ptr,                                               \
        fvec<DIM> *__restrict__ v_feature_ptr                                          \
    );

__INS__(3)
#undef __INS__

} // namespace cugsplat