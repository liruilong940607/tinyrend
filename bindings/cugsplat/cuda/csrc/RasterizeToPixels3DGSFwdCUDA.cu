#include <cooperative_groups.h>
#include <cstdint>

#include "RasterizeToPixels3DGS.h"
#include "tinyrend/common/vec.h"
#include "tinyrend/rasterization/base.cuh"
#include "tinyrend/util/warp.cuh"

namespace cugsplat {

using namespace tinyrend;

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
    // TODO(ruilong): do we really need to add 0.5f here?
    auto const dx = pixel_x + 0.5f - mean[0];
    auto const dy = pixel_y + 0.5f - mean[1];
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
    : tinyrend::rasterization::BaseRasterizeKernelOperator<
          ImageGaussianRasterizeKernelForwardOperator<FEATURE_DIM>> {

    using FeatureType = fvec<FEATURE_DIM>;

    // Inputs
    const float *__restrict__ opacity_ptr;       // [N, 1]
    const fvec2 *__restrict__ mean_ptr;          // [N, 2]
    const fvec3 *__restrict__ conic_ptr;         // [N, 3]
    const FeatureType *__restrict__ feature_ptr; // [N, FEATURE_DIM] (e.g., 3 for RGB or
                                                 // 256 for neural features)

    // Outputs
    int32_t
        *__restrict__ render_last_index_ptr; // [n_images, image_height, image_width, 1]
    float *__restrict__ render_alpha_ptr;    // [n_images, image_height, image_width, 1]
    FeatureType *__restrict__ render_feature_ptr; // [n_images, image_height,
                                                  // image_width, FEATURE_DIM]

    // Internal variables
    FeatureType _expected_feature =
        FeatureType::zero();  // buffer for feature accumulation
    float _T = 1.0f;          // current transmittance
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

) {
    ImageGaussianRasterizeKernelForwardOperator<FEATURE_DIM> op{};
    op.opacity_ptr = opacity_ptr;
    op.mean_ptr = mean_ptr;
    op.conic_ptr = conic_ptr;
    op.feature_ptr = feature_ptr;
    op.render_last_index_ptr = render_last_index_ptr;
    op.render_alpha_ptr = render_alpha_ptr;
    op.render_feature_ptr = render_feature_ptr;

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
        op, image_height, image_width, isect_primitive_ids, isect_prefix_sum_per_tile
    );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(DIM)                                                                   \
    template void image_gaussian_rasterize_kernel_forward<DIM>(                        \
        const size_t n_primitives,                                                     \
        const float *__restrict__ opacity_ptr,                                         \
        fvec2 *__restrict__ mean_ptr,                                                  \
        fvec3 *__restrict__ conic_ptr,                                                 \
        fvec<DIM> *__restrict__ feature_ptr,                                           \
        const size_t n_images,                                                         \
        const size_t image_height,                                                     \
        const size_t image_width,                                                      \
        const size_t tile_width,                                                       \
        const size_t tile_height,                                                      \
        const uint32_t *__restrict__ isect_primitive_ids,                              \
        const uint32_t *__restrict__ isect_prefix_sum_per_tile,                        \
        int32_t *__restrict__ render_last_index_ptr,                                   \
        float *__restrict__ render_alpha_ptr,                                          \
        fvec<DIM> *__restrict__ render_feature_ptr                                     \
    );

__INS__(3)
#undef __INS__

} // namespace cugsplat