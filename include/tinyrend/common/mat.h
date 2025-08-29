#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/vec.h"

namespace tinyrend {

template <typename T, size_t Cols, size_t Rows> struct alignas(T) mat {
    vec<T, Rows> data[Cols]; // Column vectors for access

    // Default constructor
    constexpr mat() = default;

    // Initialize from values
    template <typename... Args> TREND_HOST_DEVICE constexpr mat(Args... args) {
        static_assert(sizeof...(args) == Rows * Cols, "Invalid number of arguments");
        T arr[] = {static_cast<T>(args)...};
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] = arr[i + j * Rows];
            }
        }
    }

    // Initialize from pointer (column-major)
    TREND_HOST_DEVICE static mat from_ptr_col_major(const T *ptr) {
        mat result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result.data[j][i] = ptr[i + j * Rows];
            }
        }
        return result;
    }

    // Initialize from pointer (row-major)
    TREND_HOST_DEVICE static mat from_ptr_row_major(const T *ptr) {
        mat result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result.data[j][i] = ptr[i * Cols + j];
            }
        }
        return result;
    }

    // Zero initialization
    TREND_HOST_DEVICE static mat zero() {
        mat result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = T(0);
            }
        }
        return result;
    }

    // Ones initialization
    TREND_HOST_DEVICE static mat ones() {
        mat result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = T(1);
            }
        }
        return result;
    }

    // Identity initialization
    TREND_HOST_DEVICE static mat identity() {
        static_assert(Rows == Cols, "Identity matrix must be square");
        mat result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = i == j ? T(1) : T(0);
            }
        }
        return result;
    }

    // Unary minus operator
    TREND_HOST_DEVICE mat<T, Cols, Rows> operator-() const {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = -data[j][i];
            }
        }
        return result;
    }

    // Access operators [col] to return vec<T>
    TREND_HOST_DEVICE vec<T, Rows> &operator[](size_t col) { return data[col]; }
    TREND_HOST_DEVICE const vec<T, Rows> &operator[](size_t col) const {
        return data[col];
    }

    // Access operators (col, row)
    TREND_HOST_DEVICE T &operator()(size_t col, size_t row) { return data[col][row]; }
    TREND_HOST_DEVICE const T &operator()(size_t col, size_t row) const {
        return data[col][row];
    }

    // Matrix-Matrix element-wise operations
    TREND_HOST_DEVICE mat<T, Cols, Rows> operator+(const mat<T, Cols, Rows> &other
    ) const {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = data[j][i] + other(j, i);
            }
        }
        return result;
    }

    TREND_HOST_DEVICE mat<T, Cols, Rows> operator-(const mat<T, Cols, Rows> &other
    ) const {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = data[j][i] - other(j, i);
            }
        }
        return result;
    }

    // Matrix-Matrix product
    template <size_t OtherCols>
    TREND_HOST_DEVICE mat<T, OtherCols, Rows>
    operator*(const mat<T, OtherCols, Cols> &other) const {
        mat<T, OtherCols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < OtherCols; ++j) {
                result(j, i) = T(0);
#pragma unroll
                for (size_t k = 0; k < Cols; ++k) {
                    result(j, i) += data[k][i] * other(j, k);
                }
            }
        }
        return result;
    }

    // Matrix-Scalar operations
    TREND_HOST_DEVICE mat<T, Cols, Rows> operator+(T scalar) const {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = data[j][i] + scalar;
            }
        }
        return result;
    }

    TREND_HOST_DEVICE mat<T, Cols, Rows> operator-(T scalar) const {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = data[j][i] - scalar;
            }
        }
        return result;
    }

    TREND_HOST_DEVICE mat<T, Cols, Rows> operator*(T scalar) const {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = data[j][i] * scalar;
            }
        }
        return result;
    }

    TREND_HOST_DEVICE mat<T, Cols, Rows> operator/(T scalar) const {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = data[j][i] / scalar;
            }
        }
        return result;
    }

    // Scalar-Matrix operations (friend functions)
    TREND_HOST_DEVICE friend mat<T, Cols, Rows>
    operator+(T scalar, const mat<T, Cols, Rows> &m) {
        return m + scalar;
    }

    TREND_HOST_DEVICE friend mat<T, Cols, Rows>
    operator-(T scalar, const mat<T, Cols, Rows> &m) {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = scalar - m(j, i);
            }
        }
        return result;
    }

    TREND_HOST_DEVICE friend mat<T, Cols, Rows>
    operator*(T scalar, const mat<T, Cols, Rows> &m) {
        return m * scalar;
    }

    TREND_HOST_DEVICE friend mat<T, Cols, Rows>
    operator/(T scalar, const mat<T, Cols, Rows> &m) {
        mat<T, Cols, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = scalar / m(j, i);
            }
        }
        return result;
    }

    // Matrix-Vector multiplication
    template <size_t N>
    TREND_HOST_DEVICE vec<T, Rows> operator*(const vec<T, N> &v) const {
        static_assert(N == Cols, "Vector dimension must match matrix data");
        vec<T, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
            result[i] = T(0);
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result[i] += data[j][i] * v[j];
            }
        }
        return result;
    }

    // Vector-Matrix multiplication
    template <size_t N>
    TREND_HOST_DEVICE friend vec<T, Cols>
    operator*(const vec<T, N> &v, const mat<T, Rows, Cols> &m) {
        static_assert(N == Rows, "Vector dimension must match matrix rows");
        vec<T, Cols> result;
#pragma unroll
        for (size_t i = 0; i < Cols; ++i) {
            result[i] = T(0);
#pragma unroll
            for (size_t j = 0; j < Rows; ++j) {
                result[i] += m.data[i][j] * v[j];
            }
        }
        return result;
    }

    // Compound assignment operators
    TREND_HOST_DEVICE mat<T, Cols, Rows> &operator+=(const mat<T, Cols, Rows> &other) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] += other(j, i);
            }
        }
        return *this;
    }

    TREND_HOST_DEVICE mat<T, Cols, Rows> &operator-=(const mat<T, Cols, Rows> &other) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] -= other(j, i);
            }
        }
        return *this;
    }

    TREND_HOST_DEVICE mat<T, Cols, Rows> &operator*=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] *= scalar;
            }
        }
        return *this;
    }

    TREND_HOST_DEVICE mat<T, Cols, Rows> &operator/=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] /= scalar;
            }
        }
        return *this;
    }

    // Comparison operators
    TREND_HOST_DEVICE bool operator==(const mat<T, Cols, Rows> &other) const {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                if (data[j][i] != other(j, i)) {
                    return false;
                }
            }
        }
        return true;
    }

    TREND_HOST_DEVICE bool operator!=(const mat<T, Cols, Rows> &other) const {
        return !(*this == other);
    }

    // Is close
    TREND_HOST_DEVICE bool
    is_close(const mat<T, Cols, Rows> &other, T atol = 1e-5f, T rtol = 1e-5f) const {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                if (std::fabs(data[j][i] - other(j, i)) >
                    atol + rtol * std::fabs(other(j, i))) {
                    return false;
                }
            }
        }
        return true;
    }

    // To string
    std::string to_string() const {
        std::stringstream ss;
        ss << "mat" << Cols << "x" << Rows << "(";
        for (size_t j = 0; j < Cols; ++j) {
            if (j > 0)
                ss << ", ";
            ss << "(";
            for (size_t i = 0; i < Rows; ++i) {
                if (i > 0)
                    ss << ", ";
                ss << data[j][i];
            }
            ss << ")";
        }
        ss << ")";
        return ss.str();
    }

    // Transpose
    TREND_HOST_DEVICE mat<T, Rows, Cols> transpose() const {
        mat<T, Rows, Cols> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = data[j][i];
            }
        }
        return result;
    }
};

// Common type aliases
template <size_t Cols, size_t Rows> using fmat = mat<float, Cols, Rows>;
using fmat2x2 = fmat<2, 2>;
using fmat2x3 = fmat<2, 3>;
using fmat2x4 = fmat<2, 4>;
using fmat3x2 = fmat<3, 2>;
using fmat3x3 = fmat<3, 3>;
using fmat3x4 = fmat<3, 4>;
using fmat4x2 = fmat<4, 2>;
using fmat4x3 = fmat<4, 3>;
using fmat4x4 = fmat<4, 4>;

using fmat2 = fmat<2, 2>;
using fmat3 = fmat<3, 3>;
using fmat4 = fmat<4, 4>;

// Functions
template <typename T, size_t N1, size_t N2>
inline TREND_HOST_DEVICE mat<T, N2, N1>
outer(const vec<T, N1> &v1, const vec<T, N2> &v2) {
    mat<T, N2, N1> result;
#pragma unroll
    for (size_t r = 0; r < N1; ++r) {
#pragma unroll
        for (size_t c = 0; c < N2; ++c) {
            result(c, r) = v1[r] * v2[c];
        }
    }
    return result;
}

template <typename T>
inline TREND_HOST_DEVICE mat<T, 2, 2> inverse2x2(const mat<T, 2, 2> &m) {
    T det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    T inv_det = T(1) / det;

    T a00 = m(1, 1) * inv_det;
    T a01 = -m(0, 1) * inv_det;
    T a10 = -m(1, 0) * inv_det;
    T a11 = m(0, 0) * inv_det;
    return mat<T, 2, 2>(a00, a01, a10, a11);
}

template <typename T>
inline TREND_HOST_DEVICE mat<T, 3, 3> inverse3x3(const mat<T, 3, 3> &m) {
    T det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
            m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
            m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    T inv_det = T(1) / det;

    T a00 = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) * inv_det;
    T a01 = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * inv_det;
    T a02 = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * inv_det;
    T a10 = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * inv_det;
    T a11 = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * inv_det;
    T a12 = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * inv_det;
    T a20 = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) * inv_det;
    T a21 = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * inv_det;
    T a22 = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * inv_det;

    return mat<T, 3, 3>(a00, a01, a02, a10, a11, a12, a20, a21, a22);
}

template <typename T>
inline TREND_HOST_DEVICE mat<T, 4, 4> inverse4x4(const mat<T, 4, 4> &m) {
    // Precompute sub-determinants for efficiency
    T s0 = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    T s1 = m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0);
    T s2 = m(0, 0) * m(1, 3) - m(0, 3) * m(1, 0);
    T s3 = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
    T s4 = m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1);
    T s5 = m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2);

    T c0 = m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0);
    T c1 = m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0);
    T c2 = m(2, 0) * m(3, 3) - m(2, 3) * m(3, 0);
    T c3 = m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1);
    T c4 = m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1);
    T c5 = m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2);

    T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    T inv_det = T(1) / det;

    return mat<T, 4, 4>(
        (m(1, 1) * c5 - m(1, 2) * c4 + m(1, 3) * c3) * inv_det,
        (-m(0, 1) * c5 + m(0, 2) * c4 - m(0, 3) * c3) * inv_det,
        (m(3, 1) * s5 - m(3, 2) * s4 + m(3, 3) * s3) * inv_det,
        (-m(2, 1) * s5 + m(2, 2) * s4 - m(2, 3) * s3) * inv_det,
        (-m(1, 0) * c5 + m(1, 2) * c2 - m(1, 3) * c1) * inv_det,
        (m(0, 0) * c5 - m(0, 2) * c2 + m(0, 3) * c1) * inv_det,
        (-m(3, 0) * s5 + m(3, 2) * s2 - m(3, 3) * s1) * inv_det,
        (m(2, 0) * s5 - m(2, 2) * s2 + m(2, 3) * s1) * inv_det,
        (m(1, 0) * c4 - m(1, 1) * c2 + m(1, 3) * c0) * inv_det,
        (-m(0, 0) * c4 + m(0, 1) * c2 - m(0, 3) * c0) * inv_det,
        (m(3, 0) * s4 - m(3, 1) * s2 + m(3, 3) * s0) * inv_det,
        (-m(2, 0) * s4 + m(2, 1) * s2 - m(2, 3) * s0) * inv_det,
        (-m(1, 0) * c3 + m(1, 1) * c1 - m(1, 2) * c0) * inv_det,
        (m(0, 0) * c3 - m(0, 1) * c1 + m(0, 2) * c0) * inv_det,
        (-m(3, 0) * s3 + m(3, 1) * s1 - m(3, 2) * s0) * inv_det,
        (m(2, 0) * s3 - m(2, 1) * s1 + m(2, 2) * s0) * inv_det
    );
}

template <typename T, size_t Rows, size_t Cols>
inline TREND_HOST_DEVICE mat<T, Rows, Cols> inverse(const mat<T, Rows, Cols> &m) {
    static_assert(Rows == Cols, "Non-square matrix does not have a regular inverse");
    static_assert(
        Rows == 2 || Rows == 3 || Rows == 4,
        "Only 2x2, 3x3, and 4x4 matrices are supported"
    );

    if constexpr (Rows == 2) {
        return inverse2x2(m);
    } else if constexpr (Rows == 3) {
        return inverse3x3(m);
    } else if constexpr (Rows == 4) {
        return inverse4x4(m);
    }
}

} // namespace tinyrend