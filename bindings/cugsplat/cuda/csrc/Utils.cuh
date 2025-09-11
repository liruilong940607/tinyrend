#pragma once

#include "tinyrend/common/mat.h"
#include "tinyrend/common/vec.h"

namespace cugsplat {

using namespace tinyrend;

// Solve the tight axis-aligned bounding box radius for a Gaussian defined as
//      y = prefactor * exp(-1/2 * xᵀ * covar⁻¹ * x)
// at the given y value.
inline __device__ auto
solve_tight_radius(fmat2 covar, float prefactor, float y = 1.0f / 255.0f) -> fvec2 {
    if (prefactor < y) {
        return fvec2(0.0f, 0.0f);
    }

    // Q = xᵀ * covar⁻¹ * x
    auto const Q = -2.0f * logf(y / prefactor);

    // When finding the AABB (axis-aligned bounding box), we only need to consider the
    // projections onto the coordinate axes, regardless of the correlation terms in
    // the covariance matrix.
    auto const radius = fvec2(sqrtf(Q * covar[0][0]), sqrtf(Q * covar[1][1]));
    return radius;
}

// Perspective projection of a 3D Gaussian to 2D.
// \param mean3d 3D mean of the Gaussian.
// \param covar3d 3D covariance of the Gaussian.
// \param resolution Image resolution (width, height)
// \param focal_length Focal length (fx, fy)
// \param principal_point Principal point (cx, cy)
// \return 2D mean and covariance of the Gaussian.
inline __device__ auto persp_proj(
    const fvec3 mean3d,
    const fmat3 covar3d,
    const ivec2 resolution,
    const fvec2 focal_length,
    const fvec2 principal_point
) -> std::pair<fmat2, fvec2> {
    auto const x = mean3d[0];
    auto const y = mean3d[1];
    auto const z = mean3d[2];

    auto const fx = focal_length[0];
    auto const fy = focal_length[1];
    auto const cx = principal_point[0];
    auto const cy = principal_point[1];

    auto const lim_x_pos = 1.15f * resolution[0] - cx;
    auto const lim_x_neg = 0.15f * resolution[0] + cx;
    auto const lim_y_pos = 1.15f * resolution[1] - cy;
    auto const lim_y_neg = 0.15f * resolution[1] + cy;

    auto const rz = 1.f / z;
    auto const fxtxrz = fx * x * rz;
    auto const fytyrz = fy * y * rz;

    // mat3x2 is 3 columns x 2 rows.
    auto const J = fmat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -min(lim_x_pos, max(-lim_x_neg, fxtxrz)) * rz,
        -min(lim_y_pos, max(-lim_y_neg, fytyrz)) * rz // 3rd column
    );
    auto const covar2d = J * covar3d * J.transpose();
    auto const mean2d = fvec2({fxtxrz + cx, fytyrz + cy});

    return {covar2d, mean2d};
}

inline __device__ auto persp_proj_vjp(
    // fwd inputs
    const fvec3 mean3d,
    const fmat3 covar3d,
    const ivec2 resolution,
    const fvec2 focal_length,
    const fvec2 principal_point,
    // grad outputs
    const fmat2 v_covar2d,
    const fvec2 v_mean2d
) -> std::pair<fmat3, fvec3> {
    auto const x = mean3d[0];
    auto const y = mean3d[1];
    auto const z = mean3d[2];

    auto const fx = focal_length[0];
    auto const fy = focal_length[1];
    auto const cx = principal_point[0];
    auto const cy = principal_point[1];

    auto const lim_x_pos = 1.15f * resolution[0] - cx;
    auto const lim_x_neg = 0.15f * resolution[0] + cx;
    auto const lim_y_pos = 1.15f * resolution[1] - cy;
    auto const lim_y_neg = 0.15f * resolution[1] + cy;

    auto const rz = 1.f / z;
    auto const rz2 = rz * rz;
    auto const fxtxrz = fx * x * rz;
    auto const fytyrz = fy * y * rz;

    // mat3x2 is 3 columns x 2 rows.
    auto const J = fmat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -min(lim_x_pos, max(-lim_x_neg, fxtxrz)) * rz,
        -min(lim_y_pos, max(-lim_y_neg, fytyrz)) * rz // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    auto const v_covar3d = J.transpose() * v_covar2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    auto v_mean3d = fvec3(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    fmat3x2 v_J =
        v_covar2d * J * covar3d.transpose() + v_covar2d.transpose() * J * covar3d;
    v_covar2d.transpose() * J *covar3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d[0] += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d[2] += -rz2 * v_J[2][0] * fxtxrz;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d[1] += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d[2] += -rz2 * v_J[2][1] * fytyrz;
    }
    v_mean3d[2] += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                   2.f * fxtxrz * rz2 * v_J[2][0] + 2.f * fytyrz * rz2 * v_J[2][1];

    return {v_covar3d, v_mean3d};
}

} // namespace cugsplat