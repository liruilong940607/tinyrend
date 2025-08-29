#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/common/vec.h"
#include "tinyrend/util/solver.h"

using namespace tinyrend;
using namespace tinyrend::solver;

int test_newton_1d() {
    int fails = 0;

    // Test case 1: Simple quadratic equation x^2 - 4 = 0
    {
        auto const func = [](const float &x) -> std::pair<float, float> {
            return {x * x - 4.0f, 2.0f * x};
        };
        auto const result = newton_1d<10>(func, 3.0f);
        fails += CHECK((result.converged && is_close(result.x, 2.0f)), "");
    }

    // Test case 2: No convergence
    {
        auto const func = [](const float &x) -> std::pair<float, float> {
            return {1.0f, 0.0f}; // f(x) = 1, f'(x) = 0
        };
        auto const result = newton_1d<10>(func, 1.0f);
        fails += CHECK(!result.converged, "");
    }

    return fails;
}

int test_newton_2d() {
    int fails = 0;

    // Test case 1: Simple system x^2 + y^2 = 5, x + y = 3
    {
        auto const func = [](const fvec2 &xy) -> std::pair<fvec2, fmat2> {
            auto const x = xy[0];
            auto const y = xy[1];
            return {
                {x * x + y * y - 5.0f, x + y - 3.0f}, {2.0f * x, 2.0f * y, 1.0f, 1.0f}
            };
        };
        auto const result = newton_2d<10>(func, {1.5f, 1.0f});
        fails += CHECK((result.converged && is_close(result.x, fvec2(2.0f, 1.0f))), "");
    }

    return fails;
}

int test_linear_minimal_positive() {
    int fails = 0;

    // Test case 1: Simple linear equation
    {
        auto const poly = std::array<float, 2>{-2.0f, 1.0f}; // -2 + x = 0
        auto const root = linear_minimal_positive(poly, 0.0f, -1.0f);
        fails += CHECK(is_close(root, 2.0f), "");
    }

    // Test case 2: No positive root
    {
        auto const poly = std::array<float, 2>{2.0f, 1.0f}; // 2 + x = 0
        auto const root = linear_minimal_positive(poly, 0.0f, -1.0f);
        fails += CHECK(root == -1.0f, "");
    }

    return fails;
}

int test_quadratic_minimal_positive() {
    int fails = 0;

    // Test case 1: Simple quadratic equation
    {
        auto const poly = std::array<float, 3>{-3.0f, 2.0f, 1.0f}; // -3 + 2x + x^2 = 0
        auto const root = quadratic_minimal_positive(poly, 0.0f, -1.0f);
        fails += CHECK(is_close(root, 1.0f), "");
    }

    // Test case 2: No positive root
    {
        auto const poly = std::array<float, 3>{3.0f, 2.0f, 1.0f}; // 3 + 2x + x^2 = 0
        auto const root = quadratic_minimal_positive(poly, 0.0f, -1.0f);
        fails += CHECK(root == -1.0f, "");
    }

    return fails;
}

int test_cubic_minimal_positive() {
    int fails = 0;

    // Test case 1: Simple cubic equation
    {
        auto const poly =
            std::array<float, 4>{-6.0f, 11.0f, -6.0f, 1.0f}; // (x-1)(x-2)(x-3) = 0
        auto const root = cubic_minimal_positive(poly, 0.0f, -1.0f);
        fails += CHECK(is_close(root, 1.0f), "");
    }

    // Test case 2: No positive root
    {
        auto const poly =
            std::array<float, 4>{6.0f, 11.0f, 6.0f, 1.0f}; // (x+1)(x+2)(x+3) = 0
        auto const root = cubic_minimal_positive(poly, 0.0f, -1.0f);
        fails += CHECK(root == -1.0f, "");
    }

    return fails;
}

int test_polyN_minimal_positive_newton() {
    int fails = 0;

    // Test case 1: Quartic equation
    {
        auto const poly = std::array<float, 5>{
            -24.0f, 50.0f, -35.0f, 10.0f, -1.0f
        }; // (x-1)(x-2)(x-3)(x-4) = 0
        auto const root = polyN_minimal_positive_newton<10>(poly, 0.0f, 0.5f, -1.0f);
        fails += CHECK(is_close(root, 1.0f), "");
    }

    // Test case 2: No convergence
    {
        auto const poly =
            std::array<float, 5>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // 1 + x^4 = 0
        auto const root = polyN_minimal_positive_newton<10>(poly, 0.0f, 1.0f, -1.0f);
        fails += CHECK(root == -1.0f, "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_newton_1d();
    fails += test_newton_2d();
    fails += test_linear_minimal_positive();
    fails += test_quadratic_minimal_positive();
    fails += test_cubic_minimal_positive();
    fails += test_polyN_minimal_positive_newton();

    if (fails > 0) {
        printf("[util/solver.cpp] %d tests failed!\n", fails);
    } else {
        printf("[util/solver.cpp] All tests passed!\n");
    }

    return fails;
}