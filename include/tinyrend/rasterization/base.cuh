#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace tinyrend::rasterization {

namespace cg = cooperative_groups;

/*
    A CRTP base class for all rasterize kernel operators.
    All rasterize kernel operators must inherit from this class.
*/
template <typename Derived> struct BaseRasterizeKernelOperator {
  public:
    static inline __host__ auto sm_size_per_primitive() -> uint32_t {
        return Derived::sm_size_per_primitive_impl();
    }

    inline __device__ auto initialize(
        uint32_t image_id,
        uint32_t pixel_x,
        uint32_t pixel_y,
        uint32_t image_width,
        uint32_t image_height,
        char *sm_ptr,
        uint32_t thread_rank,
        uint32_t n_threads_per_block
    ) -> bool {
        this->image_id = image_id;
        this->pixel_x = pixel_x;
        this->pixel_y = pixel_y;
        this->image_width = image_width;
        this->image_height = image_height;
        this->sm_ptr = sm_ptr;
        this->thread_rank = thread_rank;
        this->pixel_id = pixel_y * image_width + pixel_x;
        this->n_threads_per_block = n_threads_per_block;
        return static_cast<Derived *>(this)->initialize_impl();
    }

    inline __device__ auto primitive_preprocess(uint32_t primitive_id) -> void {
        static_cast<Derived *>(this)->primitive_preprocess_impl(primitive_id);
    }

    template <class WarpT>
    inline __device__ auto
    rasterize(uint32_t batch_start, uint32_t t, WarpT &warp) -> bool {
        return static_cast<Derived *>(this)->rasterize_impl(batch_start, t, warp);
    }

    inline __device__ auto pixel_postprocess() -> void {
        static_cast<Derived *>(this)->pixel_postprocess_impl();
    }

  protected:
    uint32_t image_id;
    uint32_t pixel_x;
    uint32_t pixel_y;
    uint32_t pixel_id;
    uint32_t image_width;
    uint32_t image_height;
    char *sm_ptr;
    uint32_t thread_rank;
    uint32_t n_threads_per_block;
};

// A helper struct to check if a type is a valid rasterize kernel operator.
template <typename T>
struct is_rasterize_kernel_operator
    : std::is_base_of<BaseRasterizeKernelOperator<T>, T> {};

/*
    The main rasterization kernel.

    We expect to launch this kernel with this pattern:
    - dim3 threads = {tile_width, tile_height, 1};
    - dim3 grid = {n_tiles_x, n_tiles_y, n_images};

    The kernel will rasterize the primitives to the output image.

    The input isect_primitive_ids and isect_prefix_sum_per_tile are pre-computed
    information of the primitive-tile intersections.

    The RasterizeKernelOperator should be a class that inherits from
    BaseRasterizeKernelOperator. See the example in
    tinyrend/rasterization/operators/simple_planer.cuh.

    The RasterizeKernelOperator should implement the following methods:
    - sm_size_per_primitive_impl: Return the size of the shared memory per primitive.
    - initialize_impl: Initialize the operator.
    - primitive_preprocess_impl: Each thread processes one primitive.
    - rasterize_impl: Each thread rasterize a batch of primitives to the current pixel.
    - pixel_postprocess_impl: Postprocess the rasterized pixel (e.g., write to buffer.)
*/
template <typename RasterizeKernelOperator>
__global__ void rasterize_kernel(
    RasterizeKernelOperator op,

    // The output image size
    const uint32_t image_height,
    const uint32_t image_width,

    // Primitive-Tile intersection information
    // - isect_primitive_ids: Store the primitive ids for all the intersections.
    // [n_isects]
    // - isect_prefix_sum_per_tile: Store the prefix sum of the number of intersections
    // per tile. [n_tiles]
    const uint32_t *isect_primitive_ids,
    const uint32_t *isect_prefix_sum_per_tile,

    // For each tile, scan the primitives (defined in isect_primitive_ids)
    // in the reverse order or not.
    const bool reverse_order = false
) {
    static_assert(
        is_rasterize_kernel_operator<RasterizeKernelOperator>::value,
        "RasterizeKernelOperator must inherit from BaseRasterizeKernelOperator"
    );

    // The size of each tile.
    auto const tile_width = blockDim.x;
    auto const tile_height = blockDim.y;

    // How many tiles are there in the x and y direction?
    auto const n_tiles_x = gridDim.x;
    auto const n_tiles_y = gridDim.y;
    auto const n_tiles_per_image = n_tiles_x * n_tiles_y;

    // Which image am I focusing on?
    auto const image_id = blockIdx.z;

    // Which tile am I focusing on?
    auto const tile_x = blockIdx.x;
    auto const tile_y = blockIdx.y;
    auto const tile_id = tile_y * n_tiles_x + tile_x + image_id * n_tiles_per_image;

    // Which pixel am I focusing on?
    auto const pixel_x = tile_x * tile_width + threadIdx.x;
    auto const pixel_y = tile_y * tile_height + threadIdx.y;
    // auto const pixel_id = pixel_y * image_width + pixel_x; // not used

    // How many threads are there in the block?
    auto const n_threads_per_block = blockDim.x * blockDim.y;

    // Which thread am I in the block?
    auto const thread_rank = threadIdx.x + threadIdx.y * blockDim.x;

    // warp for reduction
    auto const warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Prepare the shared memory for the operator
    extern __shared__ char sm[];

    // Initialize the operator
    auto const init_success = op.initialize(
        image_id,
        pixel_x,
        pixel_y,
        image_width,
        image_height,
        sm,
        thread_rank,
        n_threads_per_block
    );

    // Check if the pixel is inside the image. If not, we still keep this thread
    // alive to help with preprocessing primitives.
    auto const inside = pixel_x < image_width && pixel_y < image_height;
    auto done = !(inside && init_success);

    // First, figure out which primitives intersect with the current tile.
    // If reverse_order is true, we scan the primitives from end -> start.
    // Otherwise, we scan the primitives from start -> end.
    auto const start = tile_id == 0 ? 0 : isect_prefix_sum_per_tile[tile_id - 1];
    auto const end = isect_prefix_sum_per_tile[tile_id];

    // Since each thread is responsible for loading one primitive into shared memory,
    // we can load at most `n_threads_per_block` primitives at a time as a batch. So
    // here we figure out how many batches do we need to load all the primitives.
    auto const n_batches =
        (end - start + n_threads_per_block - 1) / n_threads_per_block;

    // Now start the rasterization process.
    for (int32_t b = reverse_order ? n_batches - 1 : 0;
         reverse_order ? b >= 0 : b < n_batches;
         reverse_order ? --b : ++b) {
        // resync all threads before beginning next batch and early stop if entire
        // tile is done
        if (__syncthreads_count(done) >= n_threads_per_block) {
            break;
        }

        // Preprocess the next batch of primitives (e.g., load to shared memory)
        auto const batch_start = start + b * n_threads_per_block;
        auto const batch_end = min(end, batch_start + n_threads_per_block);
        auto const batch_size = batch_end - batch_start;
        if (thread_rank < batch_size) {
            auto const primitive_id = isect_primitive_ids[batch_start + thread_rank];
            op.primitive_preprocess(primitive_id);
        }

        // Wait for other threads to preprocess the primitives in the batch
        __syncthreads();

        // Now, the job of this thread is to rasterize this batch of primitives
        // to the current pixel.
        for (int32_t t = reverse_order ? batch_size - 1 : 0;
             reverse_order ? t >= 0 : t < batch_size;
             reverse_order ? --t : ++t) {
            if (done)
                break;
            // `t` is the local index of the primitive in the batch.
            bool terminate = op.rasterize(batch_start, t, warp);
            done = done || terminate;
        }
    }

    // After the rasterization process, we could do some pixel-level postprocessing.
    if (inside) {
        op.pixel_postprocess();
    }
}

} // namespace tinyrend::rasterization
