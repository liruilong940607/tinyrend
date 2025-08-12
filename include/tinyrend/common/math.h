#pragma once

#include <cmath>
#include <cstdint>

#include "tinyrend/common/macros.h"

namespace tinyrend {

template <typename T> inline TREND_HOST_DEVICE T rsqrt(const T x) {
    // #ifdef __CUDACC__
    //     return ::rsqrt(x); // use CUDA's fast rsqrt()
    // #else
    //     return 1.0 / sqrt(x); // use standard sqrt on CPU
    // #endif

    // Always use standard sqrt as T might be a scalar_grad
    return 1.0 / sqrt(x); // use standard sqrt on CPU
}

} // namespace tinyrend