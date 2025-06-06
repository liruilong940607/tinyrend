#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace tinyrend::rasterization {

/*
    A CRTP base class for all rasterize kernel operators.
    All rasterize kernel operators must inherit from this class.
*/
template <typename Derived> struct BaseRasterizeKernelOperator {
  public:
    static __host__ auto smem_size_per_primitive() -> uint32_t {
        return Derived::smem_size_per_primitive_impl();
    }

    __device__ auto initialize(
        uint32_t image_id,
        uint32_t pixel_x,
        uint32_t pixel_y,
        uint32_t image_width,
        uint32_t image_height,
        char *smem,
        uint32_t thread_rank,
        uint32_t n_threads_per_block
    ) -> bool {
        this->image_id = image_id;
        this->pixel_x = pixel_x;
        this->pixel_y = pixel_y;
        this->image_width = image_width;
        this->image_height = image_height;
        this->smem = smem;
        this->thread_rank = thread_rank;
        this->n_threads_per_block = n_threads_per_block;
        return static_cast<Derived *>(this)->initialize_impl();
    }

    __device__ auto primitive_preprocess(uint32_t primitive_id) -> void {
        static_cast<Derived *>(this)->primitive_preprocess_impl(primitive_id);
    }

    __device__ auto rasterize(uint32_t batch_start, uint32_t t) -> bool {
        return static_cast<Derived *>(this)->rasterize_impl(batch_start, t);
    }

    __device__ auto pixel_postprocess() -> void {
        static_cast<Derived *>(this)->pixel_postprocess_impl();
    }

  protected:
    uint32_t image_id;
    uint32_t pixel_x;
    uint32_t pixel_y;
    uint32_t image_width;
    uint32_t image_height;
    char *smem;
    uint32_t thread_rank;
    uint32_t n_threads_per_block;
};

/*
    A null operator that does nothing.

    This demonstrates the what a operator should look like. And also used for testing.
*/
struct NullRasterizeKernelOperator
    : BaseRasterizeKernelOperator<NullRasterizeKernelOperator> {
    static __host__ auto smem_size_per_primitive_impl() -> uint32_t { return 0; }

    __device__ auto initialize_impl() -> bool { return true; }

    __device__ auto primitive_preprocess_impl(uint32_t primitive_id) -> void {
        // Do nothing
    }

    __device__ auto rasterize_impl(uint32_t batch_start, uint32_t t) -> bool {
        // Do nothing
        return false; // Return whether we want to terminate the rasterization process.
    }

    __device__ auto pixel_postprocess_impl() -> void {
        // Do nothing
    }
};

// A helper struct to check if a type is a valid rasterize kernel operator.
template <typename T>
struct is_rasterize_kernel_operator
    : std::is_base_of<BaseRasterizeKernelOperator<T>, T> {};

/*
    The main rasterization kernel.
*/
template <typename RasterizeKernelOperator>
__global__ void rasterizer_kernel_forward(
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
    const uint32_t *isect_prefix_sum_per_tile
) {
    static_assert(
        is_rasterize_kernel_operator<RasterizeKernelOperator>::value,
        "RasterizeKernelOperator must inherit from BaseRasterizeKernelOperator"
    );

    // We expect to launch this kernel with this pattern:
    // - dim3 threads = {tile_width, tile_height, 1};
    // - dim3 grid = {n_tiles_x, n_tiles_y, n_images};

    // The size of each tile.
    auto const tile_width = blockDim.x;
    auto const tile_height = blockDim.y;

    // How many tiles are there in the x and y direction?
    auto const n_tiles_x = gridDim.x;
    auto const n_tiles_y = gridDim.y;

    // Which tile am I focusing on?
    auto const tile_x = blockIdx.x;
    auto const tile_y = blockIdx.y;
    auto const tile_id = tile_y * n_tiles_x + tile_x;

    // Which pixel am I focusing on?
    auto const pixel_x = tile_x * tile_width + threadIdx.x;
    auto const pixel_y = tile_y * tile_height + threadIdx.y;
    auto const pixel_id = pixel_y * image_width + pixel_x;

    // Which image am I focusing on?
    auto const image_id = blockIdx.z;

    // How many threads are there in the block?
    auto const n_threads_per_block = blockDim.x * blockDim.y;

    // Which thread am I in the block?
    auto const thread_rank = threadIdx.x + threadIdx.y * blockDim.x;

    // Prepare the shared memory for the operator
    extern __shared__ char smem[];

    // Initialize the operator
    auto const init_success = op.initialize(
        image_id,
        pixel_x,
        pixel_y,
        image_width,
        image_height,
        smem,
        thread_rank,
        n_threads_per_block
    );

    // Check if the pixel is inside the image. If not, we still keep this thread
    // alive to help with preprocessing primitives.
    auto const inside = pixel_x < image_width && pixel_y < image_height;
    auto done = !(inside && init_success);

    // First, figure out which primitives intersect with the current tile.
    auto const start = tile_id == 0 ? 0 : isect_prefix_sum_per_tile[tile_id - 1];
    auto const end = isect_prefix_sum_per_tile[tile_id];

    // Since each thread is responsible for loading one primitive into shared memory,
    // we can load at most `n_threads_per_block` primitives at a time as a batch. So
    // here we figure out how many batches do we need to load all the primitives.
    auto const n_batches =
        (end - start + n_threads_per_block - 1) / n_threads_per_block;

    // Now start the rasterization process.
    for (uint32_t b = 0; b < n_batches; ++b) {
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
        for (uint32_t t = 0; (t < batch_size) && (!done); ++t) {
            // `t` is the local index of the primitive in the batch.
            bool terminate = op.rasterize(batch_start, t);
            done = done || terminate;
        }
    }

    // After the rasterization process, we could do some pixel-level postprocessing.
    if (inside) {
        op.pixel_postprocess();
    }
}

} // namespace tinyrend::rasterization
