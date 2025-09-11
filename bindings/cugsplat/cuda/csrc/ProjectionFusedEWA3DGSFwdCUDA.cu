#include <cooperative_groups.h>
#include <cstdint>

#include "ProjectionFusedEWA3DGS.h"
#include "Utils.cuh"
#include "tinyrend/camera/model.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"
#include "tinyrend/common/scalar_grad.h"
#include "tinyrend/common/vec.h"

namespace cugsplat {

using namespace tinyrend;

namespace cg = cooperative_groups;

__global__ void projection_fused_ewa_3dgs_fwd_kernel(
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
) {
    // Parallelize over C * N.
    auto const idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }

    auto const cid = idx / N; // camera id
    auto const gid = idx % N; // gaussian id

    // Load viewmat. Be aware that input is row-major but fmat3 is column-major.
    auto const viewmat_R = viewmat_R_ptr[cid].transpose();
    auto const viewmat_t = viewmat_t_ptr[cid];

    // Transform Gaussian center to camera space.
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
    auto const RS =
        R * fmat3(scale[0], 0.0f, 0.0f, 0.0f, scale[1], 0.0f, 0.0f, 0.0f, scale[2]);
    auto const covar = RS * RS.transpose();

    // transform Gaussian covariance to camera space
    auto const covar_c = tinyrend::se3::transform_covar(viewmat_R, covar);

    // Project Camera-space 3D Gaussian to Image-space 2D Gaussian.
    auto const &[covar2d, mean2d] =
        persp_proj(mean_c, covar_c, resolution, focal_length, principal_point);

    // Add a small epsilon to the diagonal to avoid numerical instability.
    auto const covar2d_blurred = covar2d + fmat2(eps2d, 0.f, 0.f, eps2d);

    // Compute 2D Gaussian extend on the image plane.
    auto const opacity = opacity_ptr[gid];
    auto const radius = solve_tight_radius(covar2d_blurred, opacity, 1.0f / 255.0f);
    auto const radii = ivec2(ceilf(radius[0]), ceilf(radius[1]));

    // Mask out Gaussian outside the image plane.
    if (mean2d[0] + radii[0] <= 0.0f || mean2d[0] - radii[0] >= resolution[0] ||
        mean2d[1] + radii[1] <= 0.0f || mean2d[1] - radii[1] >= resolution[1]) {
        radii_ptr[idx] = ivec2(0, 0);
        return;
    }

    // Compute the inverse of the 2D covariance -- we get 2D precision matrix.
    // Note that we add a small epsilon to the diagonal to avoid numerical instability.
    auto const preci2d = inverse(covar2d_blurred);

    // Store results
    radii_ptr[idx] = radii;
    means2d_ptr[idx] = mean2d;
    depth_ptr[idx] = mean_c[2];
    conic_ptr[idx] = fvec3(preci2d[0][0], preci2d[0][1], preci2d[1][1]);
}

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
) {
    auto const n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);

    if (n_elements == 0) {
        // No work to do. Early return.
        return;
    }

    projection_fused_ewa_3dgs_fwd_kernel<<<grid, threads, 0>>>(
        C,
        N,
        resolution,
        principal_point,
        focal_length,
        viewmat_R_ptr,
        viewmat_t_ptr,
        near_plane,
        far_plane,
        eps2d,
        opacity_ptr,
        mean_ptr,
        quat_ptr,
        scale_ptr,
        radii_ptr,
        means2d_ptr,
        depth_ptr,
        conic_ptr
    );
}

} // namespace cugsplat