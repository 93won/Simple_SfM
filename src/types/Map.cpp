#include "types/Map.h"
#include "types/Feature.h"

namespace SFM
{

    void Map::InsertKeyFrame(Frame::Ptr frame)
    {
        current_frame_ = frame;
        current_keyframe_ = frame;

        keyframes_.insert(make_pair(frame->id_, frame));
    }

    void Map::InsertMapPoint(MapPoint::Ptr map_point)
    {
        if (landmarks_.find(map_point->id_) == landmarks_.end())
        {
            // New map point
            landmarks_.insert(make_pair(map_point->id_, map_point));
        }
        else
        {
            // This map point already exist in the map
            landmarks_[map_point->id_] = map_point;
        }
    }
}