#include <cassert>
#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"

using namespace tinyrend;

int test() {
    int fails = 0;

    // Initialize from values and pointer
    {
        float data[4] = {1.2f, 2.0f, 3.0f, 4.0f};
        fquat q1 = fquat::from_wxyz_ptr(data);
        fquat q2 = fquat(1.2f, 2.0f, 3.0f, 4.0f);
        fails += CHECK(q1 == q2, "");
    }

    // Initialize from xyzw order pointer
    {
        float data[4] = {2.0f, 3.0f, 4.0f, 1.2f};
        fquat q1 = fquat::from_xyzw_ptr(data);
        fquat q2 = fquat(1.2f, 2.0f, 3.0f, 4.0f); // wxyz
        fails += CHECK(q1 == q2, "");
    }

    // Identity quaternion
    {
        fquat q1 = fquat::identity();
        fquat q2 = fquat(1.0f, 0.0f, 0.0f, 0.0f);
        fails += CHECK(q1 == q2, "");
    }

    // cast to mat3 and back
    {
        fquat q1 = normalize(fquat(1.0f, 2.0f, 3.0f, 4.0f));
        fmat3 mat1 = mat3_cast(q1);
        fquat q2 = quat_cast(mat1);
        fmat3 mat2 = mat3_cast(q2);
        fails += CHECK(mat1.is_close(mat2, 1e-5f, 1e-5f), "");
    }

    // Quaternion multiplication
    {
        fquat q1 = normalize(fquat(1.0f, 2.0f, 3.0f, 4.0f));
        fquat q2 = normalize(fquat(5.0f, 6.0f, 7.0f, 8.0f));
        fquat q3 = q1 * q2;

        fmat3 mat1 = mat3_cast(q1);
        fmat3 mat2 = mat3_cast(q2);
        fmat3 mat3 = mat1 * mat2;

        fails += CHECK(mat3_cast(q3).is_close(mat3, 1e-5f, 1e-5f), "");
    }

    // slerp
    {
        // Input quaternions (w, x, y, z format):
        fquat q1(0.7071067812f, 0.7071067812f, 0.0f, 0.0f); // 90° around X-axis
        fquat q2(0.7071067812f, 0.0f, 0.7071067812f, 0.0f); // 90° around Y-axis
        // Expected result at t = 0.5:
        fquat expected(0.8164965809f, 0.4082482905f, 0.4082482905f, 0.0f);
        // Test the SLERP function:
        fquat result = slerp(q1, q2, 0.5f);

        fails += CHECK(result.is_close(expected, 1e-5f, 1e-5f), "");
    }

    // inverse
    {
        fquat q1 = normalize(fquat(1.0f, 2.0f, 3.0f, 4.0f));
        fmat3 inv1 = mat3_cast(inverse(q1));
        fmat3 inv2 = inverse(mat3_cast(q1));
        fails += CHECK(inv1.is_close(inv2, 1e-5f, 1e-5f), "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test();

    if (fails > 0) {
        printf("[common/quat.cpp] %d tests failed!\n", fails);
    } else {
        printf("[common/quat.cpp] All tests passed!\n");
    }

    return fails;
}