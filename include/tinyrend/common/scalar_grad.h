#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/math.h"

namespace tinyrend {

template <typename T> struct alignas(T) scalar_grad {
    T scalar;
    T grad;

    // Default constructor
    constexpr scalar_grad() = default;

    // Constructor from scalar and grad: TODO: should we set grad to 0 if not provided?
    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad(Type scalar, Type grad = T(0))
        : scalar(static_cast<T>(scalar)), grad(static_cast<T>(grad)) {}

    // Unary minus operator
    TREND_HOST_DEVICE scalar_grad operator-() const {
        return scalar_grad(-scalar, -grad);
    }

    // Vector-Vector operations
    TREND_HOST_DEVICE scalar_grad operator+(const scalar_grad &other) const {
        return scalar_grad(scalar + other.scalar, grad + other.grad);
    }

    TREND_HOST_DEVICE scalar_grad operator-(const scalar_grad &other) const {
        return scalar_grad(scalar - other.scalar, grad - other.grad);
    }

    TREND_HOST_DEVICE scalar_grad operator*(const scalar_grad &other) const {
        return scalar_grad(
            scalar * other.scalar, grad * other.scalar + scalar * other.grad
        );
    }

    TREND_HOST_DEVICE scalar_grad operator/(const scalar_grad &other) const {
        return scalar_grad(
            scalar / other.scalar,
            (grad * other.scalar - scalar * other.grad) / (other.scalar * other.scalar)
        );
    }

    // Vector-Scalar operations
    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad operator+(Type other) const {
        T other_scalar = static_cast<T>(other);
        return scalar_grad(scalar + other_scalar, grad);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad operator-(Type other) const {
        T other_scalar = static_cast<T>(other);
        return scalar_grad(scalar - other_scalar, grad);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad operator*(Type other) const {
        T other_scalar = static_cast<T>(other);
        return scalar_grad(scalar * other_scalar, grad * other_scalar);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad operator/(Type other) const {
        T other_scalar = static_cast<T>(other);
        return scalar_grad(scalar / other_scalar, grad / other_scalar);
    }

    // Scalar-Vector operations (friend functions)
    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE friend scalar_grad operator+(Type scalar, const scalar_grad &v) {
        T scalar_scalar = static_cast<T>(scalar);
        return scalar_grad(scalar_scalar + v.scalar, v.grad);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE friend scalar_grad operator-(Type scalar, const scalar_grad &v) {
        T scalar_scalar = static_cast<T>(scalar);
        return scalar_grad(scalar_scalar - v.scalar, -v.grad);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE friend scalar_grad operator*(Type scalar, const scalar_grad &v) {
        T scalar_scalar = static_cast<T>(scalar);
        return scalar_grad(scalar_scalar * v.scalar, scalar_scalar * v.grad);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE friend scalar_grad operator/(Type scalar, const scalar_grad &v) {
        T scalar_scalar = static_cast<T>(scalar);
        return scalar_grad(
            scalar_scalar / v.scalar, -scalar_scalar * v.grad / (v.scalar * v.scalar)
        );
    }

    // Compound assignment operators
    TREND_HOST_DEVICE scalar_grad &operator+=(const scalar_grad &other) {
        scalar += other.scalar;
        grad += other.grad;
        return *this;
    }

    TREND_HOST_DEVICE scalar_grad &operator-=(const scalar_grad &other) {
        scalar -= other.scalar;
        grad -= other.grad;
        return *this;
    }

    TREND_HOST_DEVICE scalar_grad &operator*=(const scalar_grad &other) {
        scalar *= other.scalar;
        grad = grad * other.scalar + scalar * other.grad;
        return *this;
    }

    TREND_HOST_DEVICE scalar_grad &operator/=(const scalar_grad &other) {
        scalar /= other.scalar;
        grad =
            (grad * other.scalar - scalar * other.grad) / (other.scalar * other.scalar);
        return *this;
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad &operator+=(Type other) {
        T other_scalar = static_cast<T>(other);
        scalar += other_scalar;
        return *this;
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad &operator-=(Type other) {
        T other_scalar = static_cast<T>(other);
        scalar -= other_scalar;
        return *this;
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad &operator*=(Type other) {
        T other_scalar = static_cast<T>(other);
        scalar *= other_scalar;
        grad *= other_scalar;
        return *this;
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE scalar_grad &operator/=(Type other) {
        T other_scalar = static_cast<T>(other);
        scalar /= other_scalar;
        grad /= other_scalar;
        return *this;
    }

    // Vector-Vector comparison operators
    TREND_HOST_DEVICE bool operator==(const scalar_grad &other) const {
        return scalar == other.scalar;
    }

    TREND_HOST_DEVICE bool operator!=(const scalar_grad &other) const {
        return !(*this == other);
    }

    TREND_HOST_DEVICE bool operator<(const scalar_grad &other) const {
        return scalar < other.scalar;
    }

    TREND_HOST_DEVICE bool operator>(const scalar_grad &other) const {
        return scalar > other.scalar;
    }

    TREND_HOST_DEVICE bool operator<=(const scalar_grad &other) const {
        return scalar <= other.scalar;
    }

    TREND_HOST_DEVICE bool operator>=(const scalar_grad &other) const {
        return scalar >= other.scalar;
    }

    // Vector-Scalar comparison operators
    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE bool operator==(Type other) const {
        return scalar == static_cast<T>(other);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE bool operator!=(Type other) const {
        return !(*this == other);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE bool operator<(Type other) const {
        return scalar < static_cast<T>(other);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE bool operator>(Type other) const {
        return scalar > static_cast<T>(other);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE bool operator<=(Type other) const {
        return scalar <= static_cast<T>(other);
    }

    template <typename Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
    TREND_HOST_DEVICE bool operator>=(Type other) const {
        return scalar >= static_cast<T>(other);
    }

    // to string
    std::string to_string() const {
        return "scalar_grad(" + std::to_string(scalar) + ", " + std::to_string(grad) +
               ")";
    }
};

// overload operators
template <typename T>
std::ostream &operator<<(std::ostream &os, const scalar_grad<T> &x) {
    os << x.to_string();
    return os;
}

template <typename T> TREND_HOST_DEVICE scalar_grad<T> abs(const scalar_grad<T> &a) {
    T data = std::abs(a.scalar);
    T grad = a.grad * (a.scalar >= 0 ? 1 : -1);
    return scalar_grad<T>(data, grad);
}

template <typename T> TREND_HOST_DEVICE scalar_grad<T> sqrt(const scalar_grad<T> &a) {
    T data = std::sqrt(a.scalar);
    T grad = a.grad / (2.0f * data);
    return scalar_grad<T>(data, grad);
}

template <typename T> TREND_HOST_DEVICE scalar_grad<T> exp(const scalar_grad<T> &a) {
    T data = std::exp(a.scalar);
    T grad = data * a.grad;
    return scalar_grad<T>(data, grad);
}

template <typename T>
TREND_HOST_DEVICE scalar_grad<T>
fmax(const scalar_grad<T> &a, const scalar_grad<T> &b) {
    T data = std::max(a.scalar, b.scalar);
    T grad = (a.scalar >= b.scalar ? a.grad : b.grad);
    return scalar_grad<T>(data, grad);
}

template <typename T>
TREND_HOST_DEVICE scalar_grad<T> fmax(const scalar_grad<T> &a, const T &b) {
    T data = std::max(a.scalar, b);
    T grad = (a.scalar >= b ? a.grad : 0);
    return scalar_grad<T>(data, grad);
}

template <typename T>
TREND_HOST_DEVICE scalar_grad<T> fmax(const T &a, const scalar_grad<T> &b) {
    T data = std::max(a, b.scalar);
    T grad = (a >= b.scalar ? 0 : b.grad);
    return scalar_grad<T>(data, grad);
}

template <typename T>
TREND_HOST_DEVICE scalar_grad<T>
fmin(const scalar_grad<T> &a, const scalar_grad<T> &b) {
    T data = std::min(a.scalar, b.scalar);
    T grad = (a.scalar <= b.scalar ? a.grad : b.grad);
    return scalar_grad<T>(data, grad);
}

template <typename T>
TREND_HOST_DEVICE scalar_grad<T> fmin(const scalar_grad<T> &a, const T &b) {
    T data = std::min(a.scalar, b);
    T grad = (a.scalar <= b ? a.grad : 0);
    return scalar_grad<T>(data, grad);
}

template <typename T>
TREND_HOST_DEVICE scalar_grad<T> fmin(const T &a, const scalar_grad<T> &b) {
    T data = std::min(a, b.scalar);
    T grad = (a <= b.scalar ? 0 : b.grad);
    return scalar_grad<T>(data, grad);
}

} // namespace tinyrend