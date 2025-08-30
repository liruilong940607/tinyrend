// This file implements a very simple RasterizeKernelForwardOperator /
// RasterizeKernelBackwardOperator, to demonstrate the basic usage of the rasterization
// kernel.
//
// The Primitive here is an infinite planer parallel to the image plane, defined by a
// single opacity value.

#pragma once

#include <cooperative_groups.h>
#include <cstdint>

#include "tinyrend/rasterization/base.cuh"
#include "tinyrend/util/warp.cuh"

namespace tinyrend::rasterization {

namespace cg = cooperative_groups;

struct PlanerRasterizeKernelForwardOperator
    : BaseRasterizeKernelOperator<PlanerRasterizeKernelForwardOperator> {

    // Inputs
    const float *__restrict__ opacity_ptr; // [N, 1]

    // Outputs
    float *__restrict__ render_alpha_ptr; // [n_images, image_height, image_width, 1]

    // Internal variables
    float _T = 1.0f; // current transmittance

    static inline __host__ auto sm_size_per_primitive_impl() -> uint32_t {
        return sizeof(float);
    }

    template <class WarpT> inline __device__ auto initialize_impl(WarpT &warp) -> bool {
        return true;
    }

    inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // cache data to shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        sm_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
    }

    template <class WarpT>
    inline __device__ auto rasterize_impl(
        uint32_t batch_start, uint32_t t, WarpT &warp, bool &terminated
    ) -> bool {
        if (terminated) {
            return true;
        }

        // load data from shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const alpha = sm_opacity_ptr[t];

        // update the transmittance
        auto const next_T = this->_T * (1.0f - alpha);
        this->_T = next_T;

        // Return whether we want to terminate the rasterization process.
        return false;
    }

    inline __device__ auto pixel_postprocess_impl() -> void {
        // write to the output buffer
        if (this->render_alpha_ptr != nullptr) {
            auto const offset_pixel =
                this->image_id * this->image_height * this->image_width +
                this->pixel_id;
            this->render_alpha_ptr[offset_pixel] = 1.0f - this->_T;
        }
    }
};

void planer_rasterize_kernel_forward(
    // Primitives
    const size_t n_primitives,
    const float *__restrict__ opacity_ptr, // [n_primitives]

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
    float *__restrict__ render_alpha_ptr // [n_images, image_height, image_width, 1]
) {
    PlanerRasterizeKernelForwardOperator op{};
    op.opacity_ptr = opacity_ptr;
    op.render_alpha_ptr = render_alpha_ptr;

    dim3 threads(tile_width, tile_height, 1);
    dim3 grid(1, 1, 1);
    size_t sm_size = decltype(op)::sm_size_per_primitive() * threads.x * threads.y;
    rasterize_kernel<<<grid, threads, sm_size>>>(
        op, image_height, image_width, isect_primitive_ids, isect_prefix_sum_per_tile
    );
}

struct PlanerRasterizeKernelBackwardOperator
    : BaseRasterizeKernelOperator<PlanerRasterizeKernelBackwardOperator> {

    // Forward Inputs
    const float *__restrict__ opacity_ptr; // [N, 1]

    // Forward Outputs
    const float
        *__restrict__ render_alpha_ptr; // [n_images, image_height, image_width, 1]

    // Gradients for Forward Outputs
    const float
        *__restrict__ v_render_alpha_ptr; // [n_images, image_height, image_width, 1]

    // Gradients for Forward Inputs
    float *__restrict__ v_opacity_ptr; // [N, 1]

    // Internal variables
    float _T_final;        // final transmittance
    float _T;              // current transmittance (from back to front)
    float _v_render_alpha; // dl/d_render_alpha for this pixel

    static inline __host__ auto sm_size_per_primitive_impl() -> uint32_t {
        // since we will cache the opacity [float] and primitive_id [uint32_t] in the
        // shared memory, the total shared memory size per primitive is:
        return sizeof(float) + sizeof(uint32_t);
    }

    template <class WarpT> inline __device__ auto initialize_impl(WarpT &warp) -> bool {
        // load the gradient for this pixel
        auto const offset_pixel =
            this->image_id * this->image_height * this->image_width + this->pixel_id;
        this->_v_render_alpha = this->v_render_alpha_ptr[offset_pixel];

        // load the initial transmittance as remaining transmittance
        this->_T_final = 1.0f - this->render_alpha_ptr[offset_pixel];
        this->_T = this->_T_final;
        return true;
    }

    inline __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // cache data to shared memory
        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_opacity_ptr[this->n_threads_per_block]);
        sm_opacity_ptr[this->thread_rank] = this->opacity_ptr[primitive_id];
        sm_primitive_id_ptr[this->thread_rank] = primitive_id;
    }

    template <class WarpT>
    inline __device__ auto rasterize_impl(
        uint32_t batch_start, uint32_t t, WarpT &warp, bool &terminated
    ) -> bool {
        // Flag to indicate if this primitive is rendered to this pixel. If not we will
        // skip the gradient computation. Note: we do not do early return like we do in
        // forward pass because we need to call warpSum later so the thread needs to be
        // alive.
        auto maybe_rendered = true;

        if (terminated) {
            maybe_rendered = false;
        }

        auto const sm_opacity_ptr = reinterpret_cast<float *>(this->sm_ptr);
        auto const sm_primitive_id_ptr =
            reinterpret_cast<uint32_t *>(&sm_opacity_ptr[this->n_threads_per_block]);

        auto v_alpha = 0.0f;
        if (maybe_rendered) {
            // load data from shared memory
            auto const alpha = sm_opacity_ptr[t];

            // compute the gradient
            auto const ra = 1.0f / (1.0f - alpha);
            this->_T *= ra;
            v_alpha = this->_T_final * ra * this->_v_render_alpha;
        }

        // reduce the gradient over the warp [faster than atomicAdd to global memory]
        tinyrend::warp::warpSum(v_alpha, warp);

        // first thread in the warp writes the gradient to global memory.
        if (warp.thread_rank() == 0) {
            auto const primitive_id = sm_primitive_id_ptr[t];

            float *v_opacity_ptr = (float *)this->v_opacity_ptr;
            atomicAdd(v_opacity_ptr + primitive_id, v_alpha);
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

void planer_rasterize_kernel_backward(
    // Primitives
    const size_t n_primitives,
    const float *__restrict__ opacity_ptr, // [n_primitives]

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
    const float
        *__restrict__ render_alpha_ptr, // [n_images, image_height, image_width, 1]

    // Gradient for outputs
    const float
        *__restrict__ v_render_alpha_ptr, // [n_images, image_height, image_width, 1]

    // Gradient for inputs
    float *__restrict__ v_opacity_ptr // [n_primitives]
) {
    PlanerRasterizeKernelBackwardOperator op{};
    op.opacity_ptr = opacity_ptr;
    op.render_alpha_ptr = render_alpha_ptr;
    op.v_render_alpha_ptr = v_render_alpha_ptr;
    op.v_opacity_ptr = v_opacity_ptr;

    dim3 threads(tile_width, tile_height, 1);
    dim3 grid(1, 1, 1);
    size_t sm_size = decltype(op)::sm_size_per_primitive() * tile_width * tile_height;
    rasterize_kernel<<<grid, threads, sm_size>>>(
        op,
        image_height,
        image_width,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        true // reverse order
    );
}

} // namespace tinyrend::rasterization