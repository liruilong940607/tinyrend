#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"
#include "tinyrend/common/vec.h"

namespace tinyrend::warp {

namespace cg = cooperative_groups;

// For scalar types (constrain to float for now)
template <class WarpT> inline __device__ void warpSum(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

// For vector types
template <class WarpT, typename T, size_t N>
inline __device__ void warpSum(vec<T, N> &val, WarpT &warp) {
#pragma unroll
    for (size_t i = 0; i < N; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<T>());
    }
}

// For quat types
template <class WarpT, typename T>
inline __device__ void warpSum(quat<T> &val, WarpT &warp) {
    warpSum(val.w, warp);
    warpSum(val.x, warp);
    warpSum(val.y, warp);
    warpSum(val.z, warp);
}

// For matrix types
template <class WarpT, typename T, size_t Cols, size_t Rows>
inline __device__ void warpSum(mat<T, Cols, Rows> &val, WarpT &warp) {
#pragma unroll
    for (size_t i = 0; i < Cols; i++) { // column major
        warpSum(val[i], warp);
    }
}

} // namespace tinyrend::warp