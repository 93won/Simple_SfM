#ifndef SFM_FACTOR_H
#define SFM_FACTOR_H

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace SFM
{

    // Standard bundle adjustment cost function for variable
    // camera pose and calibration and point parameters.

    class ProjectionFactorSimplePinholeConstantIntrinsic
    {
    public:
        explicit ProjectionFactorSimplePinholeConstantIntrinsic(const Vec2 &point2D, const Mat33 &intrinsic) : observed_x_(point2D(0)),
                                                                                                               observed_y_(point2D(1)),
                                                                                                               intrinsic_(intrinsic) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D, const Mat33 &intrinsic)
        {
            return (new ceres::AutoDiffCostFunction<ProjectionFactorSimplePinholeConstantIntrinsic, 2, 4, 3, 3>(
                new ProjectionFactorSimplePinholeConstantIntrinsic(point2D, intrinsic)));
        }

        template <typename T>
        bool operator()(const T *const qvec, const T *const tvec, const T *const point3D, T *residuals) const
        {
            // Rotate and translate.
            T projection[3];
            // R * P + t / w x y z
            ceres::QuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            // World To Image

            T fx = T(intrinsic_(0, 0));
            T fy = T(intrinsic_(1, 1));
            T cx = T(intrinsic_(0, 2));
            T cy = T(intrinsic_(1, 2));

            // No distortion
            residuals[0] = (fx * projection[0] + cx) - T(observed_x_);
            residuals[1] = (fy * projection[1] + cy) - T(observed_y_);

            // std::cerr<<"Check Residual: "<<residuals[0]<<" / "<<residuals[1]<<std::endl;

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
        const Mat33 intrinsic_;
    };

    class ProjectionFactorSimplePinhole
    {
    public:
        explicit ProjectionFactorSimplePinhole(const Vec2 &point2D) : observed_x_(point2D(0)),
                                                                      observed_y_(point2D(1)) {}

        static ceres::CostFunction *Create(const Eigen::Vector2d &point2D)
        {
            return (new ceres::AutoDiffCostFunction<ProjectionFactorSimplePinhole, 2, 4, 3, 3, 3>(
                new ProjectionFactorSimplePinhole(point2D)));
        }

        template <typename T>
        bool operator()(const T *const qvec, const T *const tvec, const T *const point3D, const T *const intrinsic, T *residuals) const
        {
            // Rotate and translate.
            T projection[3];
            // R * P + t / w x y z
            ceres::QuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // No distortion
            residuals[0] = (intrinsic[0] * projection[0] + intrinsic[1]) - T(observed_x_);
            residuals[1] = (intrinsic[0] * projection[1] + intrinsic[2]) - T(observed_y_);

            // std::cerr<<"Check Residual: "<<residuals[0]<<" / "<<residuals[1]<<std::endl;

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
    };

}
#endif