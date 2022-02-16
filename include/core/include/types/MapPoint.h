#pragma once
#ifndef SFM_MAPPOINT_H
#define SFM_MAPPOINT_H

#include "types/Common.h"

namespace SFM
{

    struct Frame;
    struct Feature;

    struct MapPoint
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_ = 0; // ID
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero(); // Position in world
        std::vector<double> rgb_;
        std::mutex data_mutex_;
        int observed_times_ = 0; // being observed by feature matching algo.
        std::list<std::weak_ptr<Feature>> observations_;

        unsigned long id_frame_ = 0; // first observation frame id

        MapPoint() {}

        MapPoint(long id, Vec3 position);

        Vec3 Pos()
        {
            // share same mutex for protecting data
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos(const Vec3 &pos)
        {
            // share same mutex for protecting data
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        };

        void AddObservation(std::shared_ptr<Feature> feature)
        {
            // share same mutex for protecting data
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);
            observed_times_++;
        }

        std::list<std::weak_ptr<Feature>> GetObs()
        {
            // share same mutex for protecting data
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };
}
#endif
