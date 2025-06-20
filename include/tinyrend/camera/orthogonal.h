#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "tinyrend/core/macros.h" // for GSPLAT_HOST_DEVICE

namespace tinyrend::camera::orthogonal {

/// \brief Project a 3D point in camera space to 2D image space using
/// orthogonal projection.
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \return Projected 2D point in image space
GSPLAT_HOST_DEVICE inline auto project(
    glm::fvec3 const &camera_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point
) -> glm::fvec2 {
    auto const xy = glm::fvec2(camera_point);
    auto const image_point = focal_length * xy + principal_point;
    return image_point;
}

/// \brief Compute the Jacobian of the projection function.
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \return Jacobian of the projection function
GSPLAT_HOST_DEVICE inline auto project_jac(
    glm::fvec3 const &camera_point, glm::fvec2 const &focal_length
) -> glm::fmat3x2 {
    auto const J = glm::fmat3x2{focal_length[0], 0.f, 0.f, focal_length[1], 0.f, 0.f};
    return J;
}

/// \brief Compute the Hessian of the projection: H = d²(image_point) / d(camera_point)²
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \return Array of two 3x3 Hessian matrices (H1 = ∂²u/∂p², H2 = ∂²v/∂p²)
GSPLAT_HOST_DEVICE inline auto project_hess(
    glm::fvec3 const &camera_point, glm::fvec2 const &focal_length
) -> std::array<glm::fmat3x3, 2> {
    // Hessian is zero for orthogonal projection
    return {glm::fmat3x3{}, glm::fmat3x3{}};
}

/// \brief Unproject a 2D image point to a ray in camera space.
/// \param image_point 2D point in image space
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \return Ray in camera space (origin, direction)
GSPLAT_HOST_DEVICE inline auto unproject(
    glm::fvec2 const &image_point,
    glm::fvec2 const &focal_length,
    glm::fvec2 const &principal_point
) -> std::pair<glm::fvec3, glm::fvec3> {
    auto const xy = (image_point - principal_point) / focal_length;
    auto const origin = glm::fvec3{xy[0], xy[1], 0.f};
    auto const dir = glm::fvec3{0.f, 0.f, 1.f};
    return {origin, dir};
}

} // namespace tinyrend::camera::orthogonal
