#include "core/SfM.h"

namespace SFM
{

    SfM::SfM()
    {
        std::string config_file_path_ = "../config/3dii.yaml";
        if (!Config::SetParameterFile(config_file_path_))
            LOG(INFO) << "No configuration file loaded.";

        if (Config::Get<std::string>("feature_type") == "ORB")
        {
            detector_ = cv::ORB::create(Config::Get<int>("num_features"));
            descriptor_ = cv::ORB::create(Config::Get<int>("num_features"));
        }
        // else if (Config::Get<std::string>("feature_type") == "SIFT")
        // {
        //     detector_ = cv::SIFT::create(Config::Get<int>("num_features"));
        //     descriptor_ = cv::SIFT::create(Config::Get<int>("num_features"));
        // }

        LOG(INFO) <<"OpenCV Version: "<<CV_VERSION;

        matcher_ = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        knn_ratio_ = Config::Get<double>("knn_ratio");
        num_features_ = Config::Get<int>("num_features");
        match_save_dir = Config::Get<std::string>("match_save_dir");
    }

    bool SfM::AddFrame(Frame::Ptr frame)
    {
        current_frame_ = frame;

        if (map_->keyframes_.size() == 0)
        {
            SfMInit();
        }
        else
        {
            BuildMap();
        }

        return true;
    }

    bool SfM::SfMInit()
    {
        int num_features = DetectFeatures();
        InsertKeyFrame();
        return true;
    }

    int SfM::DetectFeatures()
    {
        // detect features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector_->detect(current_frame_->gray_, keypoints);
        int cnt_detected = 0;
        for (auto &kp : keypoints)
        {
            int u = (int)kp.pt.y;
            int v = (int)kp.pt.x;

            std::vector<double> rgb_{current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[2] / 1.0,
                                     current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[1] / 1.0,
                                     current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[0] / 1.0};

            current_frame_->features_.push_back(Feature::Ptr(new Feature(current_frame_, kp, 1.0, rgb_)));

            cnt_detected++;
        }

        return cnt_detected;
    }

    int SfM::MatchTwoFrames(Frame::Ptr frame_dst,
                            std::vector<cv::DMatch> &matches,
                            std::vector<cv::KeyPoint> &kps1,
                            std::vector<cv::KeyPoint> &kps2,
                            bool ransac,
                            bool showMatch)
    {
        // calculate descriptor
        cv::Mat des_1, des_2;
        for (auto &feature : frame_dst->features_)
            kps1.emplace_back(feature->position_);

        for (auto &feature : current_frame_->features_)
            kps2.emplace_back(feature->position_);

        descriptor_->compute(frame_dst->gray_, kps1, des_1);
        descriptor_->compute(current_frame_->gray_, kps2, des_2);

        std::vector<std::vector<cv::DMatch>> matches_temp;
        matcher_.knnMatch(des_1, des_2, matches_temp, 2);

        for (auto &match : matches_temp)
        {
            if (match.size() == 2)
            {
                if (match[0].distance < match[1].distance * knn_ratio_)
                {
                    matches.emplace_back(match[0]);
                }
            }
        }

        if (ransac)
        {
            // filtering using ransac

            std::vector<cv::DMatch> matches_good;

            std::vector<cv::Point2f> kps_1_pt;
            std::vector<cv::Point2f> kps_2_pt;
            for (size_t i = 0; i < matches.size(); i++)
            {
                //-- Get the keypoints from the good matches
                kps_1_pt.emplace_back(kps1[matches[i].queryIdx].pt);
                kps_2_pt.emplace_back(kps2[matches[i].trainIdx].pt);
            }

            std::vector<int> mask;
            cv::Mat H = cv::findHomography(kps_1_pt, kps_2_pt, cv::RANSAC, 50, mask, 10000, 0.9995);

            for (size_t i = 0; i < mask.size(); i++)
            {
                if (mask[i] == 1)
                {
                    matches_good.emplace_back(matches[i]);
                }
            }
            matches = matches_good;
        }

        // Ransac filter

        if (showMatch)
        {
            cv::Mat img_match;

            cv::drawMatches(frame_dst->rgb_, kps1, current_frame_->rgb_, kps2, matches, img_match);

            std::string title = "Match_" + std::to_string(frame_dst->id_) + "_" + std::to_string(current_frame_->id_);
            std::string save_dir = match_save_dir + "/" + title + ".jpg";
            cv::imwrite(save_dir, img_match);
            // cv::imshow(title, img_match);
            // cv::waitKey(0);
        }

        return (int)matches.size();
    }

    bool SfM::BuildMap()
    {
        int num_features = DetectFeatures(); // current frame feature detection

        for (auto &kf_map : map_->keyframes_)
        {
            Frame::Ptr kf = kf_map.second;

            std::vector<cv::DMatch> matches;
            std::vector<cv::KeyPoint> kps1, kps2;

            int num_matches = MatchTwoFrames(kf, matches, kps1, kps2, true, true);

            std::vector<Eigen::Vector3d> pts1, pts2;
            Eigen::Matrix3d K = camera_->K();

            for (size_t i = 0; i < matches.size(); i++)
            {
                auto m = matches[i];

                // link map point
                auto mp = kf->features_[m.queryIdx]->map_point_.lock();

                if (mp)
                {
                    // existing map point
                    current_frame_->features_[m.trainIdx]->map_point_ = kf->features_[m.queryIdx]->map_point_;
                    mp->AddObservation(current_frame_->features_[m.trainIdx]);
                }
                else
                {
                    // new map point
                    Vec2 p_c_last(kf->features_[m.queryIdx]->position_.pt.x, kf->features_[m.queryIdx]->position_.pt.y);
                    Vec3 p_w_last = kf->pixel2world(p_c_last, kf->features_[m.queryIdx]->depth_);
                    auto new_map_point = MapPoint::CreateNewMappoint();
                    new_map_point->SetPos(p_w_last);
                    new_map_point->rgb_ = kf->features_[m.queryIdx]->rgb_;
                    new_map_point->AddObservation(kf->features_[m.queryIdx]);
                    new_map_point->AddObservation(current_frame_->features_[m.trainIdx]);
                    new_map_point->id_frame_ = kf->id_;

                    kf->features_[m.queryIdx]->map_point_ = new_map_point;
                    current_frame_->features_[m.trainIdx]->map_point_ = new_map_point;

                    map_->InsertMapPoint(new_map_point);
                }
            }
        }

        InsertKeyFrame();

        LOG(INFO) << "The number of keyframes: " << map_->keyframes_.size();
        LOG(INFO) << "The number of landmarks: " << map_->landmarks_.size();

        return true;
    }

    bool SfM::InsertKeyFrame()
    {

        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        SetObservationsForKeyFrame();

        return true;
    }

    void SfM::SetObservationsForKeyFrame()
    {
        for (auto &feat : current_frame_->features_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
                mp->AddObservation(feat);
        }
    }
}
