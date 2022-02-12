#pragma once
#ifndef SFM_MAP_H
#define SFM_MAP_H

#include "types/Frame.h"
#include "types/MapPoint.h"
#include "Config.h"

namespace SFM
{

    class Map
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType; // id and class (hash)
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;    // id and class (hash)

        Map(){}

        void InsertKeyFrame(Frame::Ptr frame);
        void InsertMapPoint(MapPoint::Ptr map_point);

        LandmarksType GetAllMapPoints()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return landmarks_;
        }
        KeyframesType GetAllKeyFrames()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return keyframes_;
        }

   

        std::mutex data_mutex_;
        LandmarksType landmarks_;        // all landmarks
        KeyframesType keyframes_;        // all keyframes
        Frame::Ptr current_keyframe_;    // current keyframe
        Frame::Ptr current_frame_ = nullptr;

    };
}

#endif
