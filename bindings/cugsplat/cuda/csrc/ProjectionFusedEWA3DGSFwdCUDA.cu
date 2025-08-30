#pragma once

#include <cooperative_groups.h>
#include <cstdint>

#include "tinyrend/camera/model.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"
#include "tinyrend/common/scalar_grad.h"
#include "tinyrend/common/vec.h"

namespace cugsplat {

using namespace tinyrend;

namespace cg = cooperative_groups;

// A helper struct to check if a type is a valid camera model.
template <typename T>
struct is_camera_model : std::is_base_of<BaseCameraModel<T>, T> {};

template <typename CameraModelType>
__global__ void projection_fused_ewa_3dgs_fwd_kernel(
    // Camera
    CameraModelType camera_model,            // All C cameras share the same intrinsic.
    const fmat3 *__restrict__ viewmat_R_ptr, // (C, 3, 3) world to camera rotation
    const fvec3 *__restrict__ viewmat_t_ptr, // (C, 3) world to camera translation
    const float near_plane,
    const float far_plane,
    const float eps2d,

    // Gaussians
    const float *__restrict__ opacity_ptr, // (N,)
    const fvec3 *__restrict__ mean_ptr,    // (N, 3)
    const fquat *__restrict__ quat_ptr,    // (N, 4)
    const fvec3 *__restrict__ scale_ptr,   // (N, 3)

    // Outputs
    ivec2 *__restrict__ radii_ptr,   // (C, N, 2)
    fvec2 *__restrict__ means2d_ptr, // (C, N, 2)
    float *__restrict__ depth_ptr,   // (C, N)
    fvec3 *__restrict__ conic_ptr    // (C, N, 3)
) {
    // parallelize over C * N.
    auto const idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }

    auto const cid = idx / N; // camera id
    auto const gid = idx % N; // gaussian id

    // load viewmat. be aware that input is row-major but fmat3 is column-major.
    auto const viewmat_R = fmat3::from_ptr_row_major(viewmat_R_ptr + cid);
    auto const viewmat_t = viewmat_t_ptr[cid];

    // transform Gaussian center to camera space
    auto const mean = mean_ptr[gid];
    auto const mean_c = tinyrend::se3::transform_point(viewmat_R, viewmat_t, mean);
    if (mean_c[2] < near_plane || mean_c[2] > far_plane) {
        // If the point is not in the frustum, skip
        radii_ptr[idx] = ivec2(0, 0);
        return;
    }

    // Compute Gaussian covariance
    auto const quat = quat_ptr[gid];
    auto const scale = scale_ptr[gid];
    auto const R = mat3_cast(quat);
    auto const RS = R * fmat3(R[0] * scale[0], R[1] * scale[1], R[2] * scale[2]);
    auto const covar = RS * RS.transpose();

    // transform Gaussian covariance to camera space
    auto const covar_c = tinyrend::se3::transform_covar(viewmat_R, covar);
}

} // namespace cugsplat