#pragma once

#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/vec.h"

namespace tinyrend {

template <typename T> struct quat {
    T w, x, y, z;

    // Default constructor
    quat() = default;

    // Constructor from components (wxyz order)
    TREND_HOST_DEVICE quat(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}

    // Constructor from vec4
    TREND_HOST_DEVICE quat(const vec<T, 4> &v) : w(v[0]), x(v[1]), y(v[2]), z(v[3]) {}

    // Constructor from wxyz order pointer
    TREND_HOST_DEVICE static quat from_wxyz_ptr(const T *ptr) {
        return quat(ptr[0], ptr[1], ptr[2], ptr[3]);
    }

    // Constructor from xyzw order pointer
    TREND_HOST_DEVICE static quat from_xyzw_ptr(const T *ptr) {
        return quat(ptr[3], ptr[0], ptr[1], ptr[2]);
    }

    // Identity quaternion
    TREND_HOST_DEVICE static quat identity() { return quat(T(1), T(0), T(0), T(0)); }

    // Unary minus operator
    TREND_HOST_DEVICE quat operator-() const { return quat(-w, -x, -y, -z); }

    // Quaternion operators
    TREND_HOST_DEVICE quat operator+(const quat &q) const {
        return quat(w + q.w, x + q.x, y + q.y, z + q.z);
    }

    TREND_HOST_DEVICE quat operator-(const quat &q) const {
        return quat(w - q.w, x - q.x, y - q.y, z - q.z);
    }

    TREND_HOST_DEVICE quat operator*(const quat &q) const {
        return quat(
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        );
    }

    // Quat-Scalar operators
    TREND_HOST_DEVICE quat operator*(T s) const {
        return quat(w * s, x * s, y * s, z * s);
    }

    TREND_HOST_DEVICE quat operator/(T s) const {
        return quat(w / s, x / s, y / s, z / s);
    }

    // Scalar-Vector operations (friend functions)
    TREND_HOST_DEVICE friend quat operator*(T s, const quat &q) { return q * s; }

    TREND_HOST_DEVICE friend quat operator/(T s, const quat &q) {
        return quat(s / q.w, s / q.x, s / q.y, s / q.z);
    }

    // Compound assignment for quaternion multiplication
    TREND_HOST_DEVICE quat &operator*=(const quat &q) {
        *this = *this * q;
        return *this;
    }

    // Comparison operators
    TREND_HOST_DEVICE bool operator==(const quat &q) const {
        vec<T, 4> v1 = vec<T, 4>(w, x, y, z);
        vec<T, 4> v2 = vec<T, 4>(q.w, q.x, q.y, q.z);
        return v1 == v2;
    }

    TREND_HOST_DEVICE bool operator!=(const quat &q) const { return !(*this == q); }

    // Is close
    TREND_HOST_DEVICE bool
    is_close(const quat &q, T atol = 1e-5f, T rtol = 1e-5f) const {
        vec<T, 4> v1 = vec<T, 4>(w, x, y, z);
        vec<T, 4> v2 = vec<T, 4>(q.w, q.x, q.y, q.z);
        return v1.is_close(v2, atol, rtol);
    }

    // To string
    std::string to_string() const {
        std::stringstream ss;
        ss << "quat(" << w << ", " << x << ", " << y << ", " << z << ")";
        return ss.str();
    }
};

// Common type aliases
using fquat = quat<float>;
using dquat = quat<double>;

// Functions
template <typename T>
inline TREND_HOST_DEVICE T dot(const quat<T> &q1, const quat<T> &q2) {
    return q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
}

template <typename T> inline TREND_HOST_DEVICE T length2(const quat<T> &q) {
    return dot(q, q);
}

template <typename T> inline TREND_HOST_DEVICE T length(const quat<T> &q) {
    return std::sqrt(length2(q));
}

template <typename T> inline TREND_HOST_DEVICE quat<T> normalize(const quat<T> &q) {
    T len = length(q);
    if (len > T(0)) {
        return q / len;
    }
    return q;
}

template <typename T>
inline TREND_HOST_DEVICE quat<T> normalize_vjp(const quat<T> &q, const quat<T> &v_o) {
    T len2 = length2(q);
    if (len2 > T(0)) {
        auto const inv_len = rsqrtf(len2);
        auto const o = q * inv_len; // normalized quat
        return (v_o - dot(v_o, o) * o) * inv_len;
    }
    return v_o;
}

template <typename T>
inline TREND_HOST_DEVICE mat<T, 3, 3> mat3_cast(const quat<T> &q) {
    T xx = q.x * q.x; // x*x
    T yy = q.y * q.y; // y*y
    T zz = q.z * q.z; // z*z
    T xy = q.x * q.y; // x*y
    T xz = q.x * q.z; // x*z
    T yz = q.y * q.z; // y*z
    T wx = q.w * q.x; // w*x
    T wy = q.w * q.y; // w*y
    T wz = q.w * q.z; // w*z

    return mat<T, 3, 3>(
        T(1) - T(2) * (yy + zz),
        T(2) * (xy + wz),
        T(2) * (xz - wy),
        T(2) * (xy - wz),
        T(1) - T(2) * (xx + zz),
        T(2) * (yz + wx),
        T(2) * (xz + wy),
        T(2) * (yz - wx),
        T(1) - T(2) * (xx + yy)
    );
}

template <typename T>
inline TREND_HOST_DEVICE quat<T>
mat3_cast_vjp(const quat<T> &q, const mat<T, 3, 3> &v_m) {
    return quat<T>(
        2.f * (q.x * (v_m[1][2] - v_m[2][1]) + q.y * (v_m[2][0] - v_m[0][2]) +
               q.z * (v_m[0][1] - v_m[1][0])),
        2.f * (-2.f * q.x * (v_m[1][1] + v_m[2][2]) + q.y * (v_m[0][1] + v_m[1][0]) +
               q.z * (v_m[0][2] + v_m[2][0]) + q.w * (v_m[1][2] - v_m[2][1])),
        2.f * (q.x * (v_m[0][1] + v_m[1][0]) - 2.f * q.y * (v_m[0][0] + v_m[2][2]) +
               q.z * (v_m[1][2] + v_m[2][1]) + q.w * (v_m[2][0] - v_m[0][2])),
        2.f * (q.x * (v_m[0][2] + v_m[2][0]) + q.y * (v_m[1][2] + v_m[2][1]) -
               2.f * q.z * (v_m[0][0] + v_m[1][1]) + q.w * (v_m[0][1] - v_m[1][0]))
    );
}

template <typename T>
inline TREND_HOST_DEVICE quat<T> quat_cast(const mat<T, 3, 3> &m) {
    T tr = m[0][0] + m[1][1] + m[2][2];

    if (tr > T(0)) {
        T S = std::sqrt(tr + T(1)) * T(2); // S=4*qw
        return quat<T>(
            T(0.25) * S,             // w
            (m[1][2] - m[2][1]) / S, // x
            (m[2][0] - m[0][2]) / S, // y
            (m[0][1] - m[1][0]) / S  // z
        );
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        T S = std::sqrt(T(1) + m[0][0] - m[1][1] - m[2][2]) * T(2); // S=4*x
        return quat<T>(
            (m[1][2] - m[2][1]) / S, // w
            T(0.25) * S,             // x
            (m[1][0] + m[0][1]) / S, // y
            (m[2][0] + m[0][2]) / S  // z
        );
    } else if (m[1][1] > m[2][2]) {
        T S = std::sqrt(T(1) + m[1][1] - m[0][0] - m[2][2]) * T(2); // S=4*y
        return quat<T>(
            (m[2][0] - m[0][2]) / S, // w
            (m[1][0] + m[0][1]) / S, // x
            T(0.25) * S,             // y
            (m[2][1] + m[1][2]) / S  // z
        );
    } else {
        T S = std::sqrt(T(1) + m[2][2] - m[0][0] - m[1][1]) * T(2); // S=4*z
        return quat<T>(
            (m[0][1] - m[1][0]) / S, // w
            (m[2][0] + m[0][2]) / S, // x
            (m[2][1] + m[1][2]) / S, // y
            T(0.25) * S              // z
        );
    }
}

template <typename T>
inline TREND_HOST_DEVICE quat<T> mix(const quat<T> &q1, const quat<T> &q2, T t) {
    return q1 * (T(1) - t) + q2 * t;
}

// Spherical linear interpolation
// Ref:
// https://github.com/NVlabs/tiny-cuda-nn/blob/d9e5274da3630ff747b67ed6188c838f4ad671ef/include/tiny-cuda-nn/vec.h#L1133
template <typename T>
inline TREND_HOST_DEVICE quat<T> slerp(const quat<T> &q1, const quat<T> &q2, T t) {
    quat<T> z = q2;

    T cos_theta = dot(q1, q2);

    // If cos_theta < 0, the interpolation will take the long way around the sphere.
    // To fix this, one quat must be negated.
    if (cos_theta < T(0)) {
        z = -q2;
        cos_theta = -cos_theta;
    }

    // Perform a linear interpolation when cos_theta is close to 1 to avoid side effect
    // of sin(angle) becoming a zero denominator
    if (cos_theta > T(1) - std::numeric_limits<T>::epsilon()) {
        return mix(q1, z, t);
    } else {
        // Essential Mathematics, page 467
        T angle = std::acos(cos_theta);
        return (std::sin((T(1) - t) * angle) * q1 + std::sin(t * angle) * z) /
               std::sin(angle);
    }
}

inline TREND_HOST_DEVICE fquat conjugate(const fquat &q) {
    return fquat(q.w, -q.x, -q.y, -q.z);
}

inline TREND_HOST_DEVICE fquat inverse(const fquat &q) {
    return conjugate(q) / length2(q);
}

} // namespace tinyrend