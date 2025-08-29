#pragma once

#include <array>
#include <cmath>

#include "tinyrend/common/mat.h"
#include "tinyrend/common/math.h"
#include "tinyrend/common/vec.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Helper macro: Returns 1 if the condition is false, 0 otherwise
#define CHECK(condition, message)                                                      \
    ((condition)                                                                       \
         ? 0                                                                           \
         : (printf("[FAIL] condition: %s, message: %s\n", #condition, message), 1))

// Helper function to check if two vectors are close within absolute and relative
// tolerances
bool is_close(const float &a, const float &b, float atol = 1e-4f, float rtol = 1e-4f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}
bool is_close(
    const double &a, const double &b, double atol = 1e-4, double rtol = 1e-4
) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

template <typename T, size_t N>
bool is_close(
    const tinyrend::vec<T, N> &a,
    const tinyrend::vec<T, N> &b,
    float atol = 1e-4f,
    float rtol = 1e-4f
) {
    return a.is_close(b, atol, rtol);
}

template <typename T, size_t Cols, size_t Rows>
bool is_close(
    const tinyrend::mat<T, Cols, Rows> &a,
    const tinyrend::mat<T, Cols, Rows> &b,
    float atol = 1e-4f,
    float rtol = 1e-4f
) {
    return a.is_close(b, atol, rtol);
}

template <typename T, size_t N>
bool is_close(
    const std::array<T, N> &a,
    const std::array<T, N> &b,
    float atol = 1e-4f,
    float rtol = 1e-4f
) {
    for (size_t i = 0; i < N; ++i) {
        if (!is_close(a[i], b[i], atol, rtol)) {
            return false;
        }
    }
    return true;
}

template <typename T, size_t N>
bool is_close(
    const std::initializer_list<T> &a,
    const std::initializer_list<T> &b,
    float atol = 1e-4f,
    float rtol = 1e-4f
) {
    return is_close(std::array<T, N>(a), std::array<T, N>(b), atol, rtol);
}

template <typename T, size_t N>
bool is_close(const T *a, const T *b, float atol = 1e-4f, float rtol = 1e-4f) {
    std::array<T, N> arr_a, arr_b;
    std::copy(a, a + N, arr_a.begin());
    std::copy(b, b + N, arr_b.begin());
    return is_close(arr_a, arr_b, atol, rtol);
}

void save_png(
    float *buffer, int width, int height, int channels, const char *filename
) {
    // Convert float buffer to unsigned char buffer
    unsigned char *image_data = new unsigned char[width * height * channels];

    // Normalize and convert float values to 0-255 range
    for (int i = 0; i < width * height * channels; i++) {
        // Clamp values between 0 and 1
        float value = std::max(0.0f, std::min(1.0f, buffer[i]));
        // Convert to 0-255 range
        image_data[i] = static_cast<unsigned char>(value * 255.0f);
    }

    // Save as PNG
    stbi_write_png(filename, width, height, channels, image_data, width * channels);

    // Clean up
    delete[] image_data;
}