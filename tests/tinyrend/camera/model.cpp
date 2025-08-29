#include <algorithm>
#include <cstdint>

#include "../helpers.h"
#include "tinyrend/camera/model.h"

using namespace tinyrend;
using namespace tinyrend::camera;

int test_perfect_pinhole() {
    int fails = 0;

    {
        ShutterPoses shutter_poses;
        shutter_poses.start = ShutterPoses::Pose{fvec3::zero(), fquat::identity()};
        shutter_poses.end = ShutterPoses::Pose{fvec3::ones(), fquat::identity()};

        auto parameters = PerfectPinholeCameraModel::Parameters{};
        parameters.resolution = {640, 480};
        parameters.shutter_type =
            tinyrend::camera::impl::shutter::Type::ROLLING_TOP_TO_BOTTOM;
        parameters.margin_factor = 0.0f;
        parameters.focal_length = fvec2(100.0f, 100.0f);
        parameters.principal_point = fvec2(320.0f, 240.0f);
        auto const model = PerfectPinholeCameraModel(parameters);

        auto const world_point = fvec3(1.0f, 1.0f, 1.0f);
        auto const image_point =
            model.world_point_to_image_point(world_point, shutter_poses);
        auto const world_ray =
            model.image_point_to_world_ray(image_point.p, shutter_poses);

        auto const t_dist = length(world_point - world_ray.o);
        auto const world_point_reproj = world_ray.o + world_ray.d * t_dist;

        fails += CHECK(image_point.valid_flag && world_ray.valid_flag, "");
        fails += CHECK(is_close(world_point_reproj, world_point), "");
    }

    return fails;
}

int test_opencv_pinhole() {
    int fails = 0;

    {
        ShutterPoses shutter_poses;
        shutter_poses.start = ShutterPoses::Pose{fvec3::zero(), fquat::identity()};
        shutter_poses.end = ShutterPoses::Pose{fvec3::ones(), fquat::identity()};

        auto parameters = OpenCVPinholeCameraModel::Parameters{};
        parameters.resolution = {640, 480};
        parameters.shutter_type =
            tinyrend::camera::impl::shutter::Type::ROLLING_TOP_TO_BOTTOM;
        parameters.margin_factor = 0.0f;
        parameters.focal_length = fvec2(100.0f, 100.0f);
        parameters.principal_point = fvec2(320.0f, 240.0f);
        parameters.radial_coeffs = {0.8f, -0.6f, 0.2f, 0.1f, -0.1f, 0.1f};
        parameters.tangential_coeffs = {0.1f, -0.1f};
        parameters.thin_prism_coeffs = {0.01f, 0.01f, -0.01f, -0.01f};

        auto const model = OpenCVPinholeCameraModel(parameters);

        auto const world_point = fvec3(1.0f, 1.0f, 1.0f);
        auto const image_point =
            model.world_point_to_image_point(world_point, shutter_poses);
        auto const world_ray =
            model.image_point_to_world_ray(image_point.p, shutter_poses);

        auto const t_dist = length(world_point - world_ray.o);
        auto const world_point_reproj = world_ray.o + world_ray.d * t_dist;

        fails += CHECK(image_point.valid_flag && world_ray.valid_flag, "");
        fails += CHECK(is_close(world_point_reproj, world_point), "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_perfect_pinhole();
    fails += test_opencv_pinhole();

    if (fails > 0) {
        printf("[camera/model.cpp] %d tests failed!\n", fails);
    } else {
        printf("[camera/model.cpp] All tests passed!\n");
    }

    return fails;
}