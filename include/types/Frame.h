#pragma once

#ifndef SFM_FRAME_H
#define SFM_FRAME_H

#include "types/Common.h"
#include "types/Camera.h"
#include "types/Feature.h"
#include "types/MapPoint.h"
#include <opencv2/features2d.hpp>

namespace SFM
{
    // forward declare
    // struct MapPoint;
    // struct Feature;

    struct Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned long id_ = 0;          // id of this frame
        unsigned long keyframe_id_ = 0; // id of key frame
        bool is_keyframe_ = false;      // is this frame keyframe?
        double time_stamp_;             // time stamp
        SE3 pose_;                      // Tcw Pose
        std::mutex pose_mutex_;         // Pose lock
        cv::Mat rgb_;                   // RGB image
        cv::Mat gray_;                  // Gray image
        Mat33 K_;                       // Intrinsic

        
        std::vector<std::shared_ptr<Feature>> features_;

        // triangle pathces

    public: // data members
        Frame() {}
        Frame(long id, double time_stamp, const SE3 &pose, const cv::Mat &rgb, const cv::Mat& gray, const Mat33 &K);

        // set and get pose, thread safe
        SE3 Pose()
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(const SE3 &pose)
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        // Set up keyframes, allocate and keyframe id
        void SetKeyFrame();

        // factory function
        static std::shared_ptr<Frame> CreateFrame();

        // coordinate transform: world, camera, pixel
        Vec3 world2camera(const Vec3 &p_w);
        Vec3 camera2world(const Vec3 &p_c);
        Vec2 camera2pixel(const Vec3 &p_c);
        Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);
        Vec3 pixel2world(const Vec2 &p_p, double depth = 1);
        Vec2 world2pixel(const Vec3 &p_w);
    };

}

#endif