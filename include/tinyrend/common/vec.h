#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/math.h"

namespace tinyrend {

template <typename T, size_t N> struct alignas(T) vec {
    T data[N];

    // Default constructor
    constexpr vec() = default;

    // Initialize from values
    template <typename... Args>
    TREND_HOST_DEVICE constexpr vec(Args... args) : data{static_cast<T>(args)...} {
        static_assert(sizeof...(args) == N, "Invalid number of arguments");
    }

    // Initialize from pointer
    TREND_HOST_DEVICE static vec from_ptr(const T *ptr) {
        vec result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = ptr[i];
        }
        return result;
    }

    // Zero initialization
    TREND_HOST_DEVICE static vec zero() {
        vec result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = T(0);
        }
        return result;
    }

    // Ones initialization
    TREND_HOST_DEVICE static vec ones() {
        vec result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = T(1);
        }
        return result;
    }

    // Unary minus operator
    TREND_HOST_DEVICE vec operator-() const {
        vec result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = -data[i];
        }
        return result;
    }

    // Sum all elements
    TREND_HOST_DEVICE T sum() const {
        T result = T(0);
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result += data[i];
        }
        return result;
    }

    // Access operators
    TREND_HOST_DEVICE constexpr T &operator[](size_t i) { return data[i]; }
    TREND_HOST_DEVICE constexpr const T &operator[](size_t i) const { return data[i]; }

    // Pointer casting operators
    TREND_HOST_DEVICE operator T *() { return data; }
    TREND_HOST_DEVICE operator const T *() const { return data; }

    // Vector-Vector operations
    TREND_HOST_DEVICE vec<T, N> operator+(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    TREND_HOST_DEVICE vec<T, N> operator-(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    TREND_HOST_DEVICE vec<T, N> operator*(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] * other[i];
        }
        return result;
    }

    TREND_HOST_DEVICE vec<T, N> operator/(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] / other[i];
        }
        return result;
    }

    // Vector-Scalar operations
    TREND_HOST_DEVICE vec<T, N> operator+(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] + scalar;
        }
        return result;
    }

    TREND_HOST_DEVICE vec<T, N> operator-(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] - scalar;
        }
        return result;
    }

    TREND_HOST_DEVICE vec<T, N> operator*(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    TREND_HOST_DEVICE vec<T, N> operator/(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] / scalar;
        }
        return result;
    }

    // Scalar-Vector operations (friend functions)
    TREND_HOST_DEVICE friend vec<T, N> operator+(T scalar, const vec<T, N> &v) {
        return v + scalar;
    }

    TREND_HOST_DEVICE friend vec<T, N> operator-(T scalar, const vec<T, N> &v) {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = scalar - v[i];
        }
        return result;
    }

    TREND_HOST_DEVICE friend vec<T, N> operator*(T scalar, const vec<T, N> &v) {
        return v * scalar;
    }

    TREND_HOST_DEVICE friend vec<T, N> operator/(T scalar, const vec<T, N> &v) {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = scalar / v[i];
        }
        return result;
    }

    // Compound assignment operators
    TREND_HOST_DEVICE vec<T, N> &operator+=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] += other[i];
        }
        return *this;
    }

    TREND_HOST_DEVICE vec<T, N> &operator-=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] -= other[i];
        }
        return *this;
    }

    TREND_HOST_DEVICE vec<T, N> &operator*=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] *= other[i];
        }
        return *this;
    }

    TREND_HOST_DEVICE vec<T, N> &operator/=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] /= other[i];
        }
        return *this;
    }

    TREND_HOST_DEVICE vec<T, N> &operator+=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] += scalar;
        }
        return *this;
    }

    TREND_HOST_DEVICE vec<T, N> &operator-=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] -= scalar;
        }
        return *this;
    }

    TREND_HOST_DEVICE vec<T, N> &operator*=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] *= scalar;
        }
        return *this;
    }

    TREND_HOST_DEVICE vec<T, N> &operator/=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] /= scalar;
        }
        return *this;
    }

    // Comparison operators
    TREND_HOST_DEVICE bool operator==(const vec<T, N> &other) const {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            if (data[i] != other[i]) {
                return false;
            }
        }
        return true;
    }

    TREND_HOST_DEVICE bool operator!=(const vec<T, N> &other) const {
        return !(*this == other);
    }

    // Is close
    TREND_HOST_DEVICE bool
    is_close(const vec<T, N> &other, T atol = 1e-5f, T rtol = 1e-5f) const {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            if (abs(data[i] - other[i]) > atol + rtol * abs(other[i])) {
                return false;
            }
        }
        return true;
    }

    // To string
    std::string to_string() const {
        std::stringstream ss;
        ss << "vec" << N << "(";
        for (size_t i = 0; i < N; ++i) {
            ss << data[i];
            if (i < N - 1) {
                ss << ", ";
            }
        }
        ss << ")";
        return ss.str();
    }
};

// Common type aliases
template <size_t N> using fvec = vec<float, N>;
using fvec2 = fvec<2>;
using fvec3 = fvec<3>;
using fvec4 = fvec<4>;

template <size_t N> using dvec = vec<double, N>;
using dvec2 = dvec<2>;
using dvec3 = dvec<3>;
using dvec4 = dvec<4>;

template <size_t N> using ivec = vec<int, N>;
using ivec2 = ivec<2>;
using ivec3 = ivec<3>;
using ivec4 = ivec<4>;

// Functions
template <typename T, size_t N>
inline TREND_HOST_DEVICE T dot(const vec<T, N> &v1, const vec<T, N> &v2) {
    return (v1 * v2).sum();
}

template <typename T, size_t N>
inline TREND_HOST_DEVICE vec<T, N> cross(const vec<T, N> &v1, const vec<T, N> &v2) {
    return vec<T, N>(
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    );
}

template <typename T, size_t N> inline TREND_HOST_DEVICE T length(const vec<T, N> &v) {
    return sqrt(dot(v, v));
}

template <typename T, size_t N> inline TREND_HOST_DEVICE T length2(const vec<T, N> &v) {
    return dot(v, v);
}

template <typename T, size_t N>
inline TREND_HOST_DEVICE T safe_length(const vec<T, N> &v) {
    // Find the maximum absolute value
    T max_abs = abs(v[0]);
#pragma unroll
    for (size_t i = 1; i < N; ++i) {
        max_abs = fmax(max_abs, abs(v[i]));
    }

    if (max_abs <= T(0))
        return T(0);

    // Compute sum of squares normalized by max_abs
    T inv_max_abs = T(1) / max_abs;
    T sum_squares = T(0);
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        T ratio = v[i] * inv_max_abs;
        sum_squares += ratio * ratio;
    }

    return max_abs * sqrt(sum_squares);
}

template <typename T, size_t N>
inline TREND_HOST_DEVICE vec<T, N> normalize(const vec<T, N> &v) {
    return v * rsqrt(length2(v));
}

template <typename T, size_t N>
inline TREND_HOST_DEVICE vec<T, N> safe_normalize(const vec<T, N> &v) {
    const T l = safe_length(v);
    return (l > 0.0f) ? (v / l) : v;
}

template <typename T, size_t N>
inline TREND_HOST_DEVICE vec<T, N>
safe_normalize_vjp(const vec<T, N> &v, const vec<T, N> &dl_dout) {
    const T l = safe_length(v);
    if (l > 0.0f) {
        const T il = T(1) / l;
        const T il3 = il * il * il;
        return il * dl_dout - il3 * dot(dl_dout, v) * v;
    }
    return dl_dout;
}

} // namespace tinyrend
