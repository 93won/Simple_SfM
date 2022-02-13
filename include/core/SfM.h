#pragma once
#ifndef SFM_SFM_H
#define SFM_SFM_H


#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <sophus/local_parameterization_se3.hpp>
#include <ctime>

#include "Config.h"
#include "types/Feature.h"
#include "factor/Factor.h"
#include "types/Common.h"
#include "types/Frame.h"
#include "types/Map.h"
#include "utils/Viewer.h"

namespace SFM
{

    class SfM
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<SfM> Ptr;

        SfM();

        bool AddFrame(Frame::Ptr frame);

        void SetMap(Map::Ptr map)
        {
            map_ = map;
        }

        void SetViewer(std::shared_ptr<Viewer> viewer)
        {
            viewer_ = viewer;
            viewer_->Initialize();
        }

        void SetCamera(Camera::Ptr camera)
        {
            camera_ = camera;
        }

        bool SfMInit();
        bool BuildMap();
        bool InsertKeyFrame();
        void SetObservationsForKeyFrame();
        bool Optimize();

        int DetectFeatures();
        int MatchTwoFrames(Frame::Ptr frame_dst,
                           std::vector<cv::DMatch> &matches,
                           std::vector<cv::KeyPoint> &kps1,
                           std::vector<cv::KeyPoint> &kps2,
                           bool ransac,
                           bool showMatch);


        Frame::Ptr current_frame_ = nullptr;
        Camera::Ptr camera_ = nullptr;

        Map::Ptr map_ = nullptr;
        std::shared_ptr<Viewer> viewer_ = nullptr;

        // params
        int num_features_ = 200;
        int match_threshold_ = 50;

        // utilities
        cv::Ptr<cv::FeatureDetector> detector_;       // feature detector in opencv
        cv::Ptr<cv::DescriptorExtractor> descriptor_; // feature descriptor extractor in opencv
        cv::Ptr<cv::DescriptorMatcher> matcher_SIFT; 
        cv::FlannBasedMatcher matcher_ORB;
        double knn_ratio_ = 0.7;

        std::vector<Frame::Ptr> frames_;

        std::string match_save_dir_;
    };

}

#endif