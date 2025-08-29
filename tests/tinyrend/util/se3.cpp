#include <algorithm>
#include <cstdint>

#include "../helpers.h"
#include "tinyrend/util/se3.h"

using namespace tinyrend;
using namespace tinyrend::se3;

int test_interpolate() {
    int fails = 0;

    // Test case 1: Quaternion interpolation
    {
        auto const rot1 = fquat(1.0f, 0.0f, 0.0f, 0.0f); // identity
        auto const transl1 = fvec3(0.0f, 0.0f, 0.0f);
        auto const rot2 = fquat(0.0f, 1.0f, 0.0f, 0.0f); // 180° around x
        auto const transl2 = fvec3(1.0f, 1.0f, 1.0f);
        auto const ratio = 0.5f;

        auto const [rot, transl] = interpolate(ratio, rot1, transl1, rot2, transl2);
        auto const expected_rot =
            fquat(0.707106781f, 0.707106781f, 0.0f, 0.0f); // 90° around x
        auto const expected_transl = fvec3(0.5f, 0.5f, 0.5f);

        fails += CHECK(
            rot.is_close(expected_rot, 1e-5f, 1e-5f) &&
                transl.is_close(expected_transl, 1e-5f, 1e-5f),
            "Quaternion interpolation failed"
        );
    }

    // Test case 2: Matrix interpolation
    {
        // Identity matrix (column-major)
        auto const rot1 = fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f // third column
        );
        auto const transl1 = fvec3(0.0f, 0.0f, 0.0f);
        // 180° rotation around y (column-major)
        auto const rot2 = fmat3(
            -1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            -1.0f // third column
        );
        auto const transl2 = fvec3(2.0f, 2.0f, 2.0f);
        auto const ratio = 0.5f;

        auto const [rot, transl] = interpolate(ratio, rot1, transl1, rot2, transl2);
        // 90° rotation around y (column-major)
        auto const expected_rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const expected_transl = fvec3(1.0f, 1.0f, 1.0f);

        fails += CHECK(
            rot.is_close(expected_rot, 1e-5f, 1e-5f) &&
                transl.is_close(expected_transl, 1e-5f, 1e-5f),
            "Matrix interpolation failed"
        );
    }

    return fails;
}

int test_transform_point() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const point = fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_point(rot, transl, point);
        // For quaternion rotation: q * p * q^-1 + t
        // 90° around y: (x,y,z) -> (z,y,-x)
        // (1,1,1) -> (1,1,-1) + (1,2,3) = (2,3,2)
        auto const expected = fvec3(2.0f, 3.0f, 2.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Quaternion transform_point failed"
        );
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const point = fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_point(rot, transl, point);
        // For column-major matrix multiplication: R * p + t
        // [0 0 1] [1]   [1]   [2]
        // [0 1 0] [1] + [2] = [3]
        // [-1 0 0][1]   [3]   [2]
        auto const expected = fvec3(2.0f, 3.0f, 2.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Matrix transform_point failed"
        );
    }

    return fails;
}

int test_invtransform_point() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const point = fvec3(2.0f, 3.0f, 2.0f);
        auto const transformed = invtransform_point(rot, transl, point);
        // For inverse quaternion rotation: q^-1 * (p - t) * q
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        // (1,1,-1) -> (1,1,1)
        auto const expected = fvec3(1.0f, 1.0f, 1.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Quaternion invtransform_point failed"
        );
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const point = fvec3(4.0f, 3.0f, 0.0f);
        auto const transformed = invtransform_point(rot, transl, point);
        // For inverse transform: R^T * (p - t)
        // [0 0 -1] [3]   [3]
        // [0 1 0]  [1] = [1]
        // [1 0 0][1]   [3]  [3]
        auto const expected = fvec3(3.0f, 1.0f, 3.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Matrix invtransform_point failed"
        );
    }

    return fails;
}

int test_transform_dir() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const dir = fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_dir(rot, dir);
        // For quaternion rotation: q * d * q^-1
        // 90° around y: (x,y,z) -> (z,y,-x)
        auto const expected = fvec3(1.0f, 1.0f, -1.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Quaternion transform_dir failed"
        );
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const dir = fvec3(1.0f, 1.0f, 1.0f);
        auto const transformed = transform_dir(rot, dir);
        auto const expected = fvec3(1.0f, 1.0f, -1.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f), "Matrix transform_dir failed"
        );
    }

    return fails;
}

int test_invtransform_dir() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const dir = fvec3(1.0f, 1.0f, -1.0f);
        auto const transformed = invtransform_dir(rot, dir);
        // For inverse quaternion rotation: q^-1 * d * q
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        auto const expected = fvec3(1.0f, 1.0f, 1.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Quaternion invtransform_dir failed"
        );
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const dir = fvec3(1.0f, 1.0f, -1.0f);
        auto const transformed = invtransform_dir(rot, dir);
        auto const expected = fvec3(1.0f, 1.0f, 1.0f);

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Matrix invtransform_dir failed"
        );
    }

    return fails;
}

int test_transform_ray() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = fvec3(1.0f, 1.0f, 1.0f);
        auto const ray_d = fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            transform_ray(rot, transl, ray_o, ray_d);
        // For quaternion rotation: q * p * q^-1 + t and q * d * q^-1
        // 90° around y: (x,y,z) -> (z,y,-x)
        auto const expected_o = fvec3(2.0f, 3.0f, 2.0f);
        auto const expected_d = fvec3(0.0f, 1.0f, 0.0f);

        fails += CHECK(
            transformed_o.is_close(expected_o, 1e-5f, 1e-5f) &&
                transformed_d.is_close(expected_d, 1e-5f, 1e-5f),
            "Quaternion transform_ray failed"
        );
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = fvec3(1.0f, 1.0f, 1.0f);
        auto const ray_d = fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            transform_ray(rot, transl, ray_o, ray_d);
        auto const expected_o = fvec3(2.0f, 3.0f, 2.0f);
        auto const expected_d = fvec3(0.0f, 1.0f, 0.0f);

        fails += CHECK(
            transformed_o.is_close(expected_o, 1e-5f, 1e-5f) &&
                transformed_d.is_close(expected_d, 1e-5f, 1e-5f),
            "Matrix transform_ray failed"
        );
    }

    return fails;
}

int test_invtransform_ray() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = fvec3(2.0f, 3.0f, 2.0f);
        auto const ray_d = fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            invtransform_ray(rot, transl, ray_o, ray_d);
        // For inverse quaternion rotation: q^-1 * (p - t) * q and q^-1 * d * q
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        auto const expected_o = fvec3(1.0f, 1.0f, 1.0f);
        auto const expected_d = fvec3(0.0f, 1.0f, 0.0f);

        fails += CHECK(
            transformed_o.is_close(expected_o, 1e-5f, 1e-5f) &&
                transformed_d.is_close(expected_d, 1e-5f, 1e-5f),
            "Quaternion invtransform_ray failed"
        );
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const transl = fvec3(1.0f, 2.0f, 3.0f);
        auto const ray_o = fvec3(4.0f, 3.0f, 0.0f);
        auto const ray_d = fvec3(0.0f, 1.0f, 0.0f);
        auto const [transformed_o, transformed_d] =
            invtransform_ray(rot, transl, ray_o, ray_d);
        auto const expected_o = fvec3(3.0f, 1.0f, 3.0f);
        auto const expected_d = fvec3(0.0f, 1.0f, 0.0f);

        fails += CHECK(
            transformed_o.is_close(expected_o, 1e-5f, 1e-5f) &&
                transformed_d.is_close(expected_d, 1e-5f, 1e-5f),
            "Matrix invtransform_ray failed"
        );
    }

    return fails;
}

int test_transform_covar() {
    int fails = 0;

    // Test case 1: Quaternion transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const covar = fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f // third column
        );
        auto const transformed = transform_covar(rot, covar);
        // For quaternion rotation: R * covar * R^T
        // 90° around y: (x,y,z) -> (z,y,-x)
        auto const expected = fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f // third column
        );

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Quaternion transform_covar failed"
        );
    }

    // Test case 2: Matrix transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const covar = fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f // third column
        );
        auto const transformed = transform_covar(rot, covar);
        auto const expected = fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f // third column
        );

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Matrix transform_covar failed"
        );
    }

    return fails;
}

int test_invtransform_covar() {
    int fails = 0;

    // Test case 1: Quaternion inverse transform
    {
        // 90° rotation around y (w, x, y, z)
        auto const rot = fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f);
        auto const covar = fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f // third column
        );
        auto const transformed = invtransform_covar(rot, covar);
        // For inverse quaternion rotation: R^T * covar * R
        // 90° around y inverse: (x,y,z) -> (-z,y,x)
        auto const expected = fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f // third column
        );

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Quaternion invtransform_covar failed"
        );
    }

    // Test case 2: Matrix inverse transform
    {
        // 90° rotation around y (column-major)
        auto const rot = fmat3(
            0.0f,
            0.0f,
            -1.0f, // first column
            0.0f,
            1.0f,
            0.0f, // second column
            1.0f,
            0.0f,
            0.0f // third column
        );
        auto const covar = fmat3(
            3.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            1.0f // third column
        );
        auto const transformed = invtransform_covar(rot, covar);
        auto const expected = fmat3(
            1.0f,
            0.0f,
            0.0f, // first column
            0.0f,
            2.0f,
            0.0f, // second column
            0.0f,
            0.0f,
            3.0f // third column
        );

        fails += CHECK(
            transformed.is_close(expected, 1e-5f, 1e-5f),
            "Matrix invtransform_covar failed"
        );
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_interpolate();
    fails += test_transform_point();
    fails += test_invtransform_point();
    fails += test_transform_dir();
    fails += test_invtransform_dir();
    fails += test_transform_ray();
    fails += test_invtransform_ray();
    fails += test_transform_covar();
    fails += test_invtransform_covar();

    if (fails > 0) {
        printf("[util/se3.cpp] %d tests failed!\n", fails);
    } else {
        printf("[util/se3.cpp] All tests passed!\n");
    }

    return fails;
}