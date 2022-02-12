#include "types/Camera.h"

namespace SFM
{

    Camera::Camera()
    {
    }

    Vec3 Camera::world2camera(const Vec3 &p_w, const SE3 &T_c_w)
    {
        return pose_ * T_c_w * p_w;
    }

    Vec3 Camera::camera2world(const Vec3 &p_c, const SE3 &T_c_w)
    {
        return T_c_w.inverse() * pose_inv_ * p_c;
    }

    Vec2 Camera::camera2pixel(const Vec3 &p_c)
    {
        return Vec2(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_);
    }

    Vec3 Camera::pixel2camera(const Vec2 &p_p, double depth)
    {
        // from image plane pixel to camera frame coordinates
        return Vec3(
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth);
    }

    Vec2 Camera::world2pixel(const Vec3 &p_w, const SE3 &T_c_w)
    {
        return camera2pixel(world2camera(p_w, T_c_w));
    }

    Vec3 Camera::pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth)
    {
        return camera2world(pixel2camera(p_p, depth), T_c_w);
    }

    // /* Pinhole Camera */

    // SimplePinholeCamera::SimplePinholeCamera() {}

    // template <typename T>
    // void SimplePinholeCamera::WorldToImage(const T *params, const T u, const T v, T *x, T *y)
    // {
    //     const T f = params[0];
    //     const T c1 = params[1];
    //     const T c2 = params[2];

    //     // No Distortion

    //     // Transform to image coordinates
    //     *x = f * u + c1;
    //     *y = f * v + c2;
    // }

    // template <typename T>
    // void SimplePinholeCamera::ImageToWorld(const T *params, const T x, const T y, T *u, T *v)
    // {
    //     const T f = params[0];
    //     const T c1 = params[1];
    //     const T c2 = params[2];

    //     *u = (x - c1) / f;
    //     *v = (y - c2) / f;
    // }

} // namespace RGBDSLAM
