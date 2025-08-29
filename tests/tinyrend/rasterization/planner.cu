#include <cuda_runtime.h>
#include <iostream>

#include "../helpers.cuh"
#include "../helpers.h"
#include "tinyrend/common/vec.h"
#include "tinyrend/rasterization/operators/planer.cuh"

using namespace tinyrend;
using namespace tinyrend::rasterization;

auto test_rasterization_planer(bool save_img = false) -> int {
    check_cuda_set_device();

    // Configurations
    const int n_primitives = 2;
    const uint32_t image_height = 22;
    const uint32_t image_width = 18;
    const uint32_t tile_width = 8;
    const uint32_t tile_height = 16;

    // Create primitive data:
    auto const opacity_ptr = create_device_ptr<float>({0.5f, 0.7f});
    // Create isect info: all two primitives are intersected with the first tile
    auto const isect_primitive_ids = create_device_ptr<uint32_t>({0, 1});
    auto const isect_prefix_sum_per_tile = create_device_ptr<uint32_t>({2});

    // Prepare forward outputs
    auto render_alpha_ptr =
        create_device_ptr<float>(image_height * image_width); // only alloc mem, no init

    planer_rasterize_kernel_forward(
        n_primitives,
        opacity_ptr,
        1, // n_images
        image_height,
        image_width,
        tile_width,
        tile_height,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        render_alpha_ptr
    );
    check_cuda_device_synchronize();
    check_cuda_get_last_error();

    // Copy data back to host and check the result
    auto const h_render_alpha_ptr =
        device_ptr_to_host_ptr<float>(render_alpha_ptr, image_height * image_width);
    for (int x = 0; x < tile_width; x++) {
        for (int y = 0; y < tile_height; y++) {
            int i = x + y * image_width;
            assert(is_close(h_render_alpha_ptr[i], 0.5f + (1 - 0.5f) * 0.7f));
        }
    }
    if (save_img) {
        save_png(
            h_render_alpha_ptr, image_width, image_height, 1, "results/render_alpha.png"
        );
    }

    // Prepare backward gradients
    auto const v_render_alpha_ptr =
        create_device_ptr<float>(image_height * image_width, 0.3f);
    auto v_opacity_ptr = create_device_ptr<float>(n_primitives, 0.0f); // zero init

    planer_rasterize_kernel_backward(
        n_primitives,
        opacity_ptr,
        1, // n_images
        image_height,
        image_width,
        tile_width,
        tile_height,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        render_alpha_ptr,
        v_render_alpha_ptr,
        v_opacity_ptr
    );
    check_cuda_device_synchronize();
    check_cuda_get_last_error();

    // o = a + (1 - a) * b
    // o = 0.5f + (1 - 0.5f) * 0.7f
    // dl/da = dl/do * do/da = 0.3f * (1 - 0.7f) = 0.09f
    // dl/db = dl/do * do/db = 0.3f * 0.5f = 0.15f
    auto const h_v_opacity_ptr =
        device_ptr_to_host_ptr<float>(v_opacity_ptr, n_primitives);
    assert(is_close(h_v_opacity_ptr[0], 0.09f * tile_width * tile_height));
    assert(is_close(h_v_opacity_ptr[1], 0.15f * tile_width * tile_height));

    return 0;
}

auto main() -> int {
    int fails = 0;
    fails += test_rasterization_planer();

    if (fails == 0) {
        printf("[rasterization/planer.cpp] All tests passed!\n");
    } else {
        printf("[rasterization/planer.cpp] %d tests failed!\n", fails);
    }

    return fails;
}