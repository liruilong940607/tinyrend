#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../helpers.h"
#include "tinyrend/estimator/ut.h"

using namespace tinyrend;
using namespace tinyrend::ut;

struct AuxData {};

// Test function that performs a simple linear transformation
template <int N, int M> struct LinearTransform {
    fmat<N, M> A; // N columns, M rows for N->M transformation (column-major)
    fvec<M> b;

    LinearTransform(fmat<N, M> const &A, fvec<M> const &b) : A(A), b(b) {}

    TREND_HOST_DEVICE auto operator()(fvec<N> const &x
    ) const -> std::tuple<fvec<M>, bool, AuxData> {
        return {A * x + b, true, AuxData{}};
    }
};

// Test function that performs a quadratic transformation
template <int N, int M> struct QuadraticTransform {
    fmat<N, M> A; // N columns, M rows for N->M transformation (column-major)
    fvec<M> b;
    float scale;

    QuadraticTransform(fmat<N, M> const &A, fvec<M> const &b, float scale)
        : A(A), b(b), scale(scale) {}

    TREND_HOST_DEVICE auto operator()(fvec<N> const &x
    ) const -> std::tuple<fvec<M>, bool, AuxData> {
        auto y = A * x + b;
        for (int i = 0; i < M; i++) {
            y[i] += scale * x[i] * x[i];
        }
        return {y, true, AuxData{}};
    }
};

// Test function that sometimes fails
template <int N, int M> struct FailingTransform {
    fmat<N, M> A; // N columns, M rows for N->M transformation (column-major)
    fvec<M> b;
    float threshold;

    FailingTransform(fmat<N, M> const &A, fvec<M> const &b, float threshold)
        : A(A), b(b), threshold(threshold) {}

    TREND_HOST_DEVICE auto operator()(fvec<N> const &x
    ) const -> std::tuple<fvec<M>, bool, AuxData> {
        // Fail if any component of x is above threshold
        for (int i = 0; i < N; i++) {
            if (std::fabs(x[i]) > threshold) {
                return {fvec<M>::zero(), false, AuxData{}};
            }
        }
        return {A * x + b, true, AuxData{}};
    }
};

int test_linear_transform() {
    int fails = 0;

    // Test case 1: Simple 2D to 2D linear transform
    {
        auto const A = fmat2(1.0f, 0.0f, 0.0f, 1.0f);
        auto const b = fvec2(1.0f, 2.0f);
        auto const f = LinearTransform<2, 2>(A, b);

        auto const mu = fvec2(0.0f, 0.0f);
        auto const sqrt_covar = fmat2(1.0f, 0.0f, 0.0f, 1.0f);

        auto const [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        fails += CHECK(success, "Linear transform test 1: transform returned false");

        // For linear transform, UT should give exact results
        auto const expected_mu = A * mu + b;
        fails += CHECK(
            mu_ut.is_close(expected_mu, 1e-5f, 1e-5f),
            "Linear transform test 1: mean mismatch"
        );

        auto const expected_covar = A * fmat2::identity() * A.transpose();
        fails += CHECK(
            covar_ut.is_close(expected_covar, 1e-5f, 1e-5f),
            "Linear transform test 1: covariance mismatch"
        );
    }

    // Test case 2: 3D to 2D linear transform
    {
        auto const A = fmat<3, 2>(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
        auto const b = fvec2(1.0f, 2.0f);
        auto const f = LinearTransform<3, 2>(A, b);

        auto const mu = fvec3(0.0f, 0.0f, 0.0f);
        auto const sqrt_covar = fmat3::identity();

        auto const [mu_ut, covar_ut, success, aux] =
            transform<3, 2, AuxData>(f, mu, sqrt_covar);

        fails += CHECK(success, "Linear transform test 2: transform returned false");

        // For 3D to 2D case, we need to handle dimensions correctly
        auto const expected_mu = fvec2(
            A(0, 0) * mu[0] + A(1, 0) * mu[1] + A(2, 0) * mu[2] + b[0],
            A(0, 1) * mu[0] + A(1, 1) * mu[1] + A(2, 1) * mu[2] + b[1]
        );
        fails += CHECK(
            mu_ut.is_close(expected_mu, 1e-5f, 1e-5f),
            "Linear transform test 2: mean mismatch"
        );

        // For covariance, we need to handle the 3D to 2D transformation
        // For y = Ax + b, cov(y) = A * cov(x) * A^T
        // Since cov(x) is identity, we just need A * A^T
        auto expected_covar = fmat2::zero();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    expected_covar(i, j) += A(k, i) * A(k, j);
                }
            }
        }
        fails += CHECK(
            covar_ut.is_close(expected_covar, 1e-5f, 1e-5f),
            "Linear transform test 2: covariance mismatch"
        );
    }

    return fails;
}

int test_quadratic_transform() {
    int fails = 0;

    // Test case 1: Simple 2D to 2D quadratic transform
    {
        auto const A = fmat2(1.0f, 0.0f, 0.0f, 1.0f);
        auto const b = fvec2(1.0f, 2.0f);
        auto const f = QuadraticTransform<2, 2>(A, b, 0.1f);

        auto const mu = fvec2(0.0f, 0.0f);
        auto const sqrt_covar = fmat2(1.0f, 0.0f, 0.0f, 1.0f);

        auto const [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        fails += CHECK(success, "Quadratic transform test 1: transform returned false");

        // For quadratic transform, UT should give approximate results
        // We can't check exact values, but we can verify the structure
        float mu_length_sq = mu_ut[0] * mu_ut[0] + mu_ut[1] * mu_ut[1];
        fails +=
            CHECK(mu_length_sq >= 1.0f, "Quadratic transform test 1: mean too small");

        // Check that covariance matrix has positive determinant (valid covariance)
        float det = covar_ut(0, 0) * covar_ut(1, 1) - covar_ut(0, 1) * covar_ut(1, 0);
        fails +=
            CHECK(det >= 0.0f, "Quadratic transform test 1: invalid covariance matrix");
    }

    return fails;
}

int test_failing_transform() {
    int fails = 0;

    // Test case 1: Transform that succeeds for small inputs
    {
        auto const A = fmat2(1.0f, 0.0f, 0.0f, 1.0f);
        auto const b = fvec2(1.0f, 2.0f);
        auto const f = FailingTransform<2, 2>(A, b, 2.0f);

        auto const mu = fvec2(0.0f, 0.0f);
        auto const sqrt_covar = fmat2(1.0f, 0.0f, 0.0f, 1.0f);

        auto const [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        fails += CHECK(
            success,
            "Failing transform test 1: transform should succeed for small inputs"
        );
    }

    // Test case 2: Transform that fails for large inputs
    {
        auto const A = fmat2(1.0f, 0.0f, 0.0f, 1.0f);
        auto const b = fvec2(1.0f, 2.0f);
        auto const f = FailingTransform<2, 2>(A, b, 0.5f);

        auto const mu = fvec2(0.0f, 3.0f);
        auto const sqrt_covar = fmat2(2.0f, 0.0f, 0.0f, 2.0f); // Larger covariance

        auto const [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        fails += CHECK(
            !success, "Failing transform test 2: transform should fail for large inputs"
        );
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_linear_transform();
    fails += test_quadratic_transform();
    fails += test_failing_transform();

    if (fails > 0) {
        printf("[estimator/ut.cpp] %d tests failed!\n", fails);
    } else {
        printf("[estimator/ut.cpp] All tests passed!\n");
    }

    return fails;
}
