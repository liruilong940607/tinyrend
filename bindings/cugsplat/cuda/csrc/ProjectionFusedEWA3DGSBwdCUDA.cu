#include <cooperative_groups.h>
#include <cstdint>

#include "ProjectionFusedEWA3DGS.h"
#include "Utils.cuh"
#include "tinyrend/camera/model.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"
#include "tinyrend/common/scalar_grad.h"
#include "tinyrend/common/vec.h"
#include "tinyrend/util/warp.cuh"

namespace cugsplat {

using namespace tinyrend;

namespace cg = cooperative_groups;

__global__ void projection_fused_ewa_3dgs_bwd_kernel(
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
) {
    // Parallelize over C * N.
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * N || radii_ptr[idx] == ivec2(0, 0)) {
        return;
    }

    auto const cid = idx / N; // camera id
    auto const gid = idx % N; // gaussian id

    // --- Start: some forward computations ---

    // Load viewmat. Be aware that input is row-major but fmat3 is column-major.
    auto const viewmat_R = viewmat_R_ptr[cid].transpose();
    auto const viewmat_t = viewmat_t_ptr[cid];

    // Transform Gaussian center to camera space.
    auto const mean = mean_ptr[gid];
    auto const mean_c = tinyrend::se3::transform_point(viewmat_R, viewmat_t, mean);

    // Compute Gaussian covariance
    auto const quat_raw = quat_ptr[gid];
    auto const quat = normalize(quat_raw);
    auto const scale = scale_ptr[gid];
    auto const S =
        fmat3(scale[0], 0.0f, 0.0f, 0.0f, scale[1], 0.0f, 0.0f, 0.0f, scale[2]);
    auto const R = mat3_cast(quat);
    auto const RS = R * S;
    auto const covar = RS * RS.transpose();

    // Transform Gaussian covariance to camera space
    auto const covar_c = tinyrend::se3::transform_covar(viewmat_R, covar);

    // --- End: some forward computations ---

    // Given v_conic, compute v_covar2d
    auto const conic = conic_ptr[idx];
    auto const preci2d = fmat2(conic[0], conic[1], conic[1], conic[2]);
    auto const v_conic = v_conic_ptr[idx];
    auto const v_preci2d =
        fmat2(v_conic[0], v_conic[1] * 0.5f, v_conic[1] * 0.5f, v_conic[2]);
    auto const v_covar2d = -preci2d * v_preci2d * preci2d;

    // Given {v_covar2d, v_means2d}, compute {v_covar_c, v_mean_c}
    auto const v_means2d = v_means2d_ptr[idx];
    auto const &[v_covar_c, v_mean_c_from_proj] = persp_proj_vjp(
        mean_c, covar_c, resolution, focal_length, principal_point, v_covar2d, v_means2d
    );

    // Add contribution from depth
    auto const v_depth = v_depth_ptr[idx];
    auto const v_mean_c = v_mean_c_from_proj + fvec3(0.0f, 0.0f, v_depth);

    // Given v_mean_c, compute v_mean
    auto v_mean = viewmat_R.transpose() * v_mean_c;

    // Given v_covar_c, compute v_covar
    auto const v_covar = viewmat_R.transpose() * v_covar_c * viewmat_R;

    // Given v_covar, compute v_quat, v_scale
    auto const v_RS = (v_covar + v_covar.transpose()) * RS;
    auto const v_R = v_RS * S;
    auto v_quat = normalize_vjp(quat_raw, mat3_cast_vjp(quat, v_R));
    auto v_scale = fvec3(
        R[0][0] * v_RS[0][0] + R[0][1] * v_RS[0][1] + R[0][2] * v_RS[0][2],
        R[1][0] * v_RS[1][0] + R[1][1] * v_RS[1][1] + R[1][2] * v_RS[1][2],
        R[2][0] * v_RS[2][0] + R[2][1] * v_RS[2][1] + R[2][2] * v_RS[2][2]
    );

    // #if __CUDA_ARCH__ >= 700
    // Write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    tinyrend::warp::warpSum(v_mean, warp_group_g);
    tinyrend::warp::warpSum(v_quat, warp_group_g);
    tinyrend::warp::warpSum(v_scale, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        float *v_mean_ptr_float = (float *)(v_mean_ptr + gid);
        atomicAdd(v_mean_ptr_float, v_mean[0]);
        atomicAdd(v_mean_ptr_float + 1, v_mean[1]);
        atomicAdd(v_mean_ptr_float + 2, v_mean[2]);

        float *v_quat_ptr_float = (float *)(v_quat_ptr + gid);
        atomicAdd(v_quat_ptr_float, v_quat.w);
        atomicAdd(v_quat_ptr_float + 1, v_quat.x);
        atomicAdd(v_quat_ptr_float + 2, v_quat.y);
        atomicAdd(v_quat_ptr_float + 3, v_quat.z);

        float *v_scale_ptr_float = (float *)(v_scale_ptr + gid);
        atomicAdd(v_scale_ptr_float, v_scale[0]);
        atomicAdd(v_scale_ptr_float + 1, v_scale[1]);
        atomicAdd(v_scale_ptr_float + 2, v_scale[2]);
    }
}

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
) {
    auto const n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);

    if (n_elements == 0) {
        // No work to do. Early return.
        return;
    }

    projection_fused_ewa_3dgs_bwd_kernel<<<grid, threads, 0>>>(
        C,
        N,
        resolution,
        principal_point,
        focal_length,
        viewmat_R_ptr,
        viewmat_t_ptr,
        mean_ptr,
        quat_ptr,
        scale_ptr,
        radii_ptr,
        conic_ptr,
        v_means2d_ptr,
        v_depth_ptr,
        v_conic_ptr,
        v_mean_ptr,
        v_quat_ptr,
        v_scale_ptr
    );
}

} // namespace cugsplat