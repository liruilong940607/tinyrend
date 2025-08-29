#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../helpers.h"
#include "tinyrend/estimator/ghq.h"

using namespace tinyrend;
using namespace tinyrend::ghq;

// Test function: f(x) = [sin(x0) + x1^2 + x0*x1, x2*exp(-x0^2)]
auto test_function = [](fvec3 const &x) -> fvec2 {
    return {std::sin(x[0]) + x[1] * x[1] + x[0] * x[1], x[2] * std::exp(-x[0] * x[0])};
};

// Analytical Jacobian of test function
auto analytical_jacobian = [](fvec3 const &x) -> fmat2x3 {
    return {
        std::cos(x[0]) + x[1],
        2.0f * x[1] + x[0],
        0.0f,
        -2.0f * x[0] * x[2] * std::exp(-x[0] * x[0]),
        0.0f,
        std::exp(-x[0] * x[0])
    };
};

// Analytical Hessian of test function
auto analytical_hessian = [](fvec3 const &x) -> std::array<fmat3, 2> {
    std::array<fmat3, 2> H{};

    // First output dimension Hessian
    H[0] = {-std::sin(x[0]), 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Second output dimension Hessian
    float exp_term = std::exp(-x[0] * x[0]);
    H[1] = {
        2.0f * x[2] * exp_term * (2.0f * x[0] * x[0] - 1.0f),
        0.0f,
        -2.0f * x[0] * exp_term,
        0.0f,
        0.0f,
        0.0f,
        -2.0f * x[0] * exp_term,
        0.0f,
        0.0f
    };

    return H;
};

int test_ghq_jacobian() {
    int fails = 0;

    // Test case 1: Normal values
    {
        fvec3 mu{0.1f, 0.2f, -0.3f};
        fvec3 sigma_diag{0.001f, 0.001f, 0.001f};

        auto [J_est, H_est] =
            estimate_jacobian_and_hessian<3, 2>(test_function, mu, sigma_diag);
        auto J_anal = analytical_jacobian(mu);

        // Compare estimated and analytical Jacobian
        fails += CHECK(
            J_est.is_close(J_anal, 1e-3f, 1e-3f),
            "GHQ Jacobian test 1: Normal values - Jacobian mismatch"
        );
    }

    // Test case 2: Different point
    {
        fvec3 mu{0.5f, -0.1f, 0.8f};
        fvec3 sigma_diag{0.001f, 0.001f, 0.001f};

        auto [J_est, H_est] =
            estimate_jacobian_and_hessian<3, 2>(test_function, mu, sigma_diag);
        auto J_anal = analytical_jacobian(mu);

        fails += CHECK(
            J_est.is_close(J_anal, 1e-3f, 1e-3f),
            "GHQ Jacobian test 2: Different point - Jacobian mismatch"
        );
    }

    return fails;
}

int test_ghq_hessian() {
    int fails = 0;

    // Test case 1: Normal values
    {
        fvec3 mu{0.1f, 0.2f, -0.3f};
        fvec3 std_dev{1e-2f, 1e-2f, 1e-1f};

        auto [J_est, H_est] =
            estimate_jacobian_and_hessian<3, 2>(test_function, mu, std_dev);
        auto H_anal = analytical_hessian(mu);

        // Compare estimated and analytical Hessian for first output
        fails += CHECK(
            H_est[0].is_close(H_anal[0], 1e-3f, 1e-3f),
            "GHQ Hessian test 1: Normal values - First output Hessian mismatch"
        );

        // Compare estimated and analytical Hessian for second output
        fails += CHECK(
            H_est[1].is_close(H_anal[1], 1e-3f, 1e-3f),
            "GHQ Hessian test 1: Normal values - Second output Hessian mismatch"
        );
    }

    // Test case 2: Different point with different standard deviations
    {
        fvec3 mu{-0.2f, 0.5f, 0.1f};
        fvec3 std_dev{5e-3f, 1e-2f, 2e-2f};

        auto [J_est, H_est] =
            estimate_jacobian_and_hessian<3, 2>(test_function, mu, std_dev);
        auto H_anal = analytical_hessian(mu);

        fails += CHECK(
            H_est[0].is_close(H_anal[0], 1e-3f, 1e-3f),
            "GHQ Hessian test 2: Different point - First output Hessian mismatch"
        );

        fails += CHECK(
            H_est[1].is_close(H_anal[1], 1e-3f, 1e-3f),
            "GHQ Hessian test 2: Different point - Second output Hessian mismatch"
        );
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_ghq_jacobian();
    fails += test_ghq_hessian();

    if (fails > 0) {
        printf("[estimator/ghq.cpp] %d tests failed!\n", fails);
    } else {
        printf("[estimator/ghq.cpp] All tests passed!\n");
    }

    return fails;
}