#include <cassert>
#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/common/mat.h"

using namespace tinyrend;

int test() {
    int fails = 0;

    // Initialize from values and pointer
    {
        float data[4] = {1.2f, 2.0f, 3.0f, 4.0f};
        fmat2x2 m1 = fmat2x2::from_ptr_col_major(data);
        fmat2x2 m2 = fmat2x2(1.2f, 2.0f, 3.0f, 4.0f);
        fails += CHECK(m1 == m2, "");
    }

    // Initialize from pointer (row-major)
    {
        float data[4] = {1.2f, 3.0f, 2.0f, 4.0f};
        fmat2x2 m1 = fmat2x2::from_ptr_row_major(data);
        fmat2x2 m2 = fmat2x2(1.2f, 2.0f, 3.0f, 4.0f);
        fails += CHECK(m1 == m2, "");
    }

    // Zero initialization
    {
        fmat2x2 m1 = fmat2x2::zero();
        CHECK(m1.is_close(fmat2x2(0.0f, 0.0f, 0.0f, 0.0f)), "");
    }

    // Ones initialization
    {
        fmat2x2 m1 = fmat2x2::ones();
        CHECK(m1.is_close(fmat2x2(1.0f, 1.0f, 1.0f, 1.0f)), "");
    }

    // Identity initialization
    {
        fmat2x2 m1 = fmat2x2::identity();
        CHECK(m1.is_close(fmat2x2(1.0f, 0.0f, 0.0f, 1.0f)), "");
    }

    // Access operators [col]
    {
        fmat2x2 m(1.2f, 2.0f, 3.0f, 4.0f);
        fvec2 v1 = m[0];
        fvec2 v2 = m[1];
        fails += CHECK(v1.is_close(fvec2(1.2f, 2.0f)), "");
        fails += CHECK(v2.is_close(fvec2(3.0f, 4.0f)), "");
    }

    // Matrix-Matrix element-wise operations
    {
        fmat2x2 m1(1.2f, 2.0f, 3.0f, 4.0f);
        fmat2x2 m2(4.0f, 5.0f, 6.0f, 7.0f);
        fmat2x2 expected_sum(5.2f, 7.0f, 9.0f, 11.0f);
        fmat2x2 expected_diff(-2.8f, -3.0f, -3.0f, -3.0f);
        fails += CHECK((m1 + m2).is_close(expected_sum), "");
        fails += CHECK((m1 - m2).is_close(expected_diff), "");
    }

    // Matrix-Matrix product
    {
        fmat2x2 m1(1.0f, 2.0f, 3.0f, 4.0f);
        fmat3x2 m2(5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f);
        fmat3x2 expected(23.0f, 34.0f, 31.0f, 46.0f, 39.0f, 58.0f);
        fails += CHECK((m1 * m2).is_close(expected), "");
    }

    // Matrix-Scalar operations
    {
        fmat2x2 m1(1.2f, 2.0f, 3.0f, 4.0f);
        fmat2x2 expected_add(2.2f, 3.0f, 4.0f, 5.0f);
        fmat2x2 expected_sub(0.2f, 1.0f, 2.0f, 3.0f);
        fmat2x2 expected_mul(2.4f, 4.0f, 6.0f, 8.0f);
        fmat2x2 expected_div(0.6f, 1.0f, 1.5f, 2.0f);
        fails += CHECK((m1 + 1.0f).is_close(expected_add), "");
        fails += CHECK((m1 - 1.0f).is_close(expected_sub), "");
        fails += CHECK((m1 * 2.0f).is_close(expected_mul), "");
        fails += CHECK((m1 / 2.0f).is_close(expected_div), "");
    }

    // Scalar-Matrix operations
    {
        fmat2x2 m1(1.2f, 2.0f, 3.0f, 4.0f);
        fmat2x2 expected_add(2.2f, 3.0f, 4.0f, 5.0f);
        fmat2x2 expected_sub(-0.2f, -1.0f, -2.0f, -3.0f);
        fmat2x2 expected_mul(2.4f, 4.0f, 6.0f, 8.0f);
        fmat2x2 expected_div(2.0f, 1.2f, 0.8f, 0.6f);
        fails += CHECK((1.0f + m1).is_close(expected_add), "");
        fails += CHECK((1.0f - m1).is_close(expected_sub), "");
        fails += CHECK((2.0f * m1).is_close(expected_mul), "");
        fails += CHECK((2.4f / m1).is_close(expected_div), "");
    }

    // Matrix-Vector / Vector-Matrix multiplication
    {
        fmat2x2 m(1.0f, 2.0f, 3.0f, 4.0f);
        fvec2 v(5.0f, 6.0f);
        fvec2 m_v_expected(23.0f, 34.0f);
        fvec2 v_m_expected(17.0f, 39.0f);
        fails += CHECK((m * v).is_close(m_v_expected), "");
        fails += CHECK((v * m).is_close(v_m_expected), "");
    }

    // Compound assignment operators
    {
        fmat2x2 m1(1.2f, 2.0f, 3.0f, 4.0f);
        fmat2x2 m2(4.0f, 5.0f, 6.0f, 7.0f);
        fmat2x2 expected_sum(5.2f, 7.0f, 9.0f, 11.0f);
        fmat2x2 expected_diff(1.2f, 2.0f, 3.0f, 4.0f);
        fmat2x2 expected_mul(2.4f, 4.0f, 6.0f, 8.0f);
        fmat2x2 expected_div(1.2f, 2.0f, 3.0f, 4.0f);

        m1 += m2;
        fails += CHECK(m1.is_close(expected_sum), "");
        m1 -= m2;
        fails += CHECK(m1.is_close(expected_diff), "");
        m1 *= 2.0f;
        fails += CHECK(m1.is_close(expected_mul), "");
        m1 /= 2.0f;
        fails += CHECK(m1.is_close(expected_div), "");
    }

    // Comparison operators
    {
        fmat2x2 m1(1.2f, 2.0f, 3.0f, 4.0f);
        fmat2x2 m2(1.2f, 2.0f, 3.0f, 4.0f);
        fmat2x2 m3(4.0f, 5.0f, 6.0f, 7.0f);
        fails += CHECK(m1 == m2, "");
        fails += CHECK(m1 != m3, "");
    }

    // To string
    {
        fmat2x2 m(1.2f, 2.0f, 3.0f, 4.0f);
        fails += CHECK(m.to_string() == "mat2x2((1.2, 2), (3, 4))", "");
    }

    // Transpose
    {
        fmat2x3 m(1.2f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        fmat3x2 expected(1.2f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f);
        fails += CHECK(m.transpose().is_close(expected), "");
    }

    // Outer product
    {
        fvec2 v1(1.0f, 2.0f);
        fvec3 v2(3.0f, 4.0f, 5.0f);
        fmat3x2 expected(3.0f, 6.0f, 4.0f, 8.0f, 5.0f, 10.0f);
        fails += CHECK((outer(v1, v2)).is_close(expected), "");
    }

    // Inverse
    {
        fmat2x2 m(1.0f, 3.0f, 2.0f, 4.0f);
        fmat2x2 expected(-2.0f, 1.5f, 1.0f, -0.5f);
        fails += CHECK(inverse(m).is_close(expected), "");

        fmat3x3 m2(1.0f, 4.0f, 2.0f, 2.0f, 4.0f, 4.0f, 3.0f, 5.0f, 5.0f);
        fmat3x3 expected2(0.0f, -2.5f, 2.0f, 0.5f, -0.25f, 0.0f, -0.5f, 1.75f, -1.0f);
        fails += CHECK(inverse(m2).is_close(expected2), "");

        fmat4x4 m3(
            1.0f,
            4.0f,
            2.0f,
            1.0f,
            2.0f,
            4.0f,
            4.0f,
            2.0f,
            3.0f,
            5.0f,
            5.0f,
            2.0f,
            4.0f,
            2.0f,
            3.0f,
            3.0f
        );
        fmat4x4 expected3(
            0.125f,
            -1.1875f,
            0.75f,
            0.25f,
            0.5f,
            -0.25f,
            0.0f,
            0.0f,
            -0.625f,
            0.4375f,
            0.25f,
            -0.25f,
            0.125f,
            1.3125f,
            -1.25f,
            0.25f
        );
        fails += CHECK(inverse(m3).is_close(expected3), "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test();

    if (fails > 0) {
        printf("[common/mat.cpp] %d tests failed!\n", fails);
    } else {
        printf("[common/mat.cpp] All tests passed!\n");
    }

    return fails;
}