#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include "Common.h"
#include "Ops.h"
#include "ProjectionFusedEWA3DGS.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"
#include "tinyrend/common/vec.h"

namespace cugsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_fused_ewa_3dgs_fwd(
    // Perfect Pinhole Camera: All cameras share the same intrinsic.
    const uint32_t image_width,
    const uint32_t image_height,
    const float principal_point_x,
    const float principal_point_y,
    const float focal_length_x,
    const float focal_length_y,

    const at::Tensor viewmat_R, // [C, 3, 3]
    const at::Tensor viewmat_t, // [C, 3]

    const float near_plane,
    const float far_plane,
    const float eps2d,

    // Gaussians
    const at::Tensor opacities, // [N]
    const at::Tensor means,     // [N, 3]
    const at::Tensor quats,     // [N, 4]
    const at::Tensor scales     // [N, 3]
) {
    DEVICE_GUARD(means);
    auto options = means.options();

    // Check inputs
    CHECK_INPUT(viewmat_R);
    CHECK_INPUT(viewmat_t);
    CHECK_INPUT(opacities);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);

    const int32_t N = means.size(0);
    const int32_t C = viewmat_R.size(0);

    // Create outputs
    at::Tensor radii = at::empty({C, N, 2}, options.dtype(at::kInt));
    at::Tensor means2d = at::empty({C, N, 2}, options);
    at::Tensor depth = at::empty({C, N}, options);
    at::Tensor conic = at::empty({C, N, 3}, options);

    launch_projection_fused_ewa_3dgs_fwd_kernel(
        // Perfect Pinhole Camera: All cameras share the same intrinsic.
        C,
        N,
        ivec2(image_width, image_height),
        fvec2(principal_point_x, principal_point_y),
        fvec2(focal_length_x, focal_length_y),
        reinterpret_cast<fmat3 *>(viewmat_R.data_ptr<float>()),
        reinterpret_cast<fvec3 *>(viewmat_t.data_ptr<float>()),
        near_plane,
        far_plane,
        eps2d,
        // Gaussians
        opacities.data_ptr<float>(),
        reinterpret_cast<fvec3 *>(means.data_ptr<float>()),
        reinterpret_cast<fquat *>(quats.data_ptr<float>()),
        reinterpret_cast<fvec3 *>(scales.data_ptr<float>()),
        // Outputs
        reinterpret_cast<ivec2 *>(radii.data_ptr<int32_t>()),
        reinterpret_cast<fvec2 *>(means2d.data_ptr<float>()),
        depth.data_ptr<float>(),
        reinterpret_cast<fvec3 *>(conic.data_ptr<float>())
    );

    return {radii, means2d, depth, conic};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> projection_fused_ewa_3dgs_bwd(
    // Perfect Pinhole Camera: All cameras share the same intrinsic.
    const uint32_t image_width,
    const uint32_t image_height,
    const float principal_point_x,
    const float principal_point_y,
    const float focal_length_x,
    const float focal_length_y,

    const at::Tensor viewmat_R, // [C, 3, 3]
    const at::Tensor viewmat_t, // [C, 3]

    // Gaussians
    const at::Tensor means,  // [N, 3]
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]

    // Forward Outputs
    const at::Tensor radii, // [C, N, 2]
    const at::Tensor conic, // [C, N, 3]

    // Gradient of Forward Outputs
    const at::Tensor v_means2d, // [C, N, 2]
    const at::Tensor v_depth,   // [C, N]
    const at::Tensor v_conic    // [C, N, 3]
) {
    DEVICE_GUARD(means);
    auto options = means.options();

    // Check inputs
    CHECK_INPUT(viewmat_R);
    CHECK_INPUT(viewmat_t);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(radii);
    CHECK_INPUT(conic);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depth);
    CHECK_INPUT(v_conic);

    const int32_t N = means.size(0);
    const int32_t C = viewmat_R.size(0);

    // Create outputs: gradients of Gaussians
    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_quats = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);

    launch_projection_fused_ewa_3dgs_bwd_kernel(
        // Perfect Pinhole Camera: All cameras share the same intrinsic.
        C,
        N,
        ivec2(image_width, image_height),
        fvec2(principal_point_x, principal_point_y),
        fvec2(focal_length_x, focal_length_y),
        reinterpret_cast<fmat3 *>(viewmat_R.data_ptr<float>()),
        reinterpret_cast<fvec3 *>(viewmat_t.data_ptr<float>()),
        // Gaussians
        reinterpret_cast<fvec3 *>(means.data_ptr<float>()),
        reinterpret_cast<fquat *>(quats.data_ptr<float>()),
        reinterpret_cast<fvec3 *>(scales.data_ptr<float>()),
        // Forward Outputs
        reinterpret_cast<ivec2 *>(radii.data_ptr<int32_t>()),
        reinterpret_cast<fvec3 *>(conic.data_ptr<float>()),
        // Gradient of Forward Outputs
        reinterpret_cast<fvec2 *>(v_means2d.data_ptr<float>()),
        v_depth.data_ptr<float>(),
        reinterpret_cast<fvec3 *>(v_conic.data_ptr<float>()),
        // Gradient of Gaussians
        reinterpret_cast<fvec3 *>(v_means.data_ptr<float>()),
        reinterpret_cast<fquat *>(v_quats.data_ptr<float>()),
        reinterpret_cast<fvec3 *>(v_scales.data_ptr<float>())
    );

    return {v_means, v_quats, v_scales};
}

} // namespace cugsplat