//
// Created by gaoxiang on 19-5-4.
//

#ifndef SFM_VIEWER_H
#define SFM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include "types/Common.h"
#include "types/Frame.h"
#include "types/Map.h"

namespace SFM
{

    class Viewer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Viewer> Ptr;

        Viewer();

        void SetMap(Map::Ptr map) { map_ = map; }

        void AddCurrentFrame(Frame::Ptr current_frame);

        // void UpdateMap();

        void SpinOnce();

        void ShowResult();

        void Initialize();

        void DrawFrame(Frame::Ptr frame, const float *color);

        void DrawMapPoints();

        void FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera);

        /// plot the features in current frame into an image
        cv::Mat PlotFrameImage();

        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        std::mutex viewer_data_mutex_;
        pangolin::View vis_display;
        pangolin::OpenGlRenderState vis_camera;

        double fx, fy, cx, cy;
        int width, height;
    };
}

#endif
