#pragma once

#include <cmath>
#include <cstdint>

#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"
#include "tinyrend/common/vec.h"

namespace cugsplat {

using namespace tinyrend;

void launch_projection_fused_ewa_3dgs_fwd_kernel(
    // Perfect Pinhole Camera: All cameras share the same intrinsic.
    const int32_t C,
    const int32_t N,
    const ivec2 resolution,
    const fvec2 principal_point,
    const fvec2 focal_length,

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
);

void launch_projection_fused_ewa_3dgs_bwd_kernel(
    // Perfect Pinhole Camera: All cameras share the same intrinsic.
    const int32_t C,
    const int32_t N,
    const ivec2 resolution,
    const fvec2 principal_point,
    const fvec2 focal_length,

    const fmat3 *__restrict__ viewmat_R_ptr, // (C, 3, 3) world to camera rotation
    const fvec3 *__restrict__ viewmat_t_ptr, // (C, 3) world to camera translation

    // Gaussians
    const fvec3 *__restrict__ mean_ptr,  // (N, 3)
    const fquat *__restrict__ quat_ptr,  // (N, 4)
    const fvec3 *__restrict__ scale_ptr, // (N, 3)

    // Forward Outputs
    const ivec2 *__restrict__ radii_ptr, // (C, N, 2)
    const fvec3 *__restrict__ conic_ptr, // (C, N, 3)

    // Gradient of Forward Outputs
    const fvec2 *__restrict__ v_means2d_ptr, // (C, N, 2)
    const float *__restrict__ v_depth_ptr,   // (C, N)
    const fvec3 *__restrict__ v_conic_ptr,   // (C, N, 3)

    // Gradient of Gaussians
    fvec3 *__restrict__ v_mean_ptr, // (N, 3)
    fquat *__restrict__ v_quat_ptr, // (N, 4)
    fvec3 *__restrict__ v_scale_ptr // (N, 3)
);

} // namespace cugsplat
