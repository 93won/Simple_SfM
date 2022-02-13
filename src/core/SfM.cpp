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
            matcher_ORB = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        }
        else if (Config::Get<std::string>("feature_type") == "SIFT")
        {
            detector_ = cv::SIFT::create(Config::Get<int>("num_features"));
            descriptor_ = cv::SIFT::create(Config::Get<int>("num_features"));
            matcher_SIFT = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }

        LOG(INFO) << "OpenCV Version: " << CV_VERSION;

        knn_ratio_ = Config::Get<double>("knn_ratio");
        num_features_ = Config::Get<int>("num_features");
        match_save_dir_ = Config::Get<std::string>("match_save_dir");
        match_threshold_ = Config::Get<int>("match_threshold");
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

        // Need to improve
        if (Config::Get<std::string>("feature_type") == "ORB")
            matcher_ORB.knnMatch(des_1, des_2, matches_temp, 2);
        else if (Config::Get<std::string>("feature_type") == "SIFT")
            matcher_SIFT->knnMatch(des_1, des_2, matches_temp, 2);

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

            if (matches.size() > match_threshold_)
            {
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
            }
            matches = matches_good;
        }

        // Ransac filter

        if (showMatch)
        {
            cv::Mat img_match;

            cv::drawMatches(frame_dst->rgb_, kps1, current_frame_->rgb_, kps2, matches, img_match);

            std::string title = "Match_" + std::to_string(frame_dst->id_) + "_" + std::to_string(current_frame_->id_);
            std::string save_dir = match_save_dir_ + "/" + title + ".jpg";
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

        // Optimize();

        InsertKeyFrame();

        if (viewer_)
        {
            viewer_->AddCurrentFrame(current_frame_);
            //viewer_->SpinOnce();
        }

        LOG(INFO) << "The number of keyframes: " << map_->keyframes_.size();
        LOG(INFO) << "The number of landmarks: " << map_->landmarks_.size();

        return true;
    }

    bool SfM::Optimize()
    {
        Eigen::Matrix3d K = camera_->K();
        int window_size = (int)(map_->keyframes_.size());

        Vec3 intrinsic_param[1];
        Vec4 qvec_param[window_size + 1];
        Vec3 tvec_param[window_size + 1];
        Vec3 lm_param[map_->landmarks_.size()];

        intrinsic_param[0] = Vec3(K(0, 0), K(0, 2), K(1, 2));

        std::unordered_map<int, int> idx2idx;
        ceres::Problem problem;
        ceres::LocalParameterization *se3_parameterization = new Sophus::test::LocalParameterizationSE3;

        // Pose parameters
        int ii = 0;
        for (auto &kfs : map_->keyframes_)
        {
            idx2idx.insert(std::make_pair(kfs.second->id_, ii));
            SE3 pose_ = kfs.second->Pose();
            Eigen::Quaterniond q_eigen(pose_.rotationMatrix());
            q_eigen.normalize();
            qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[ii] = pose_.translation();
            problem.AddParameterBlock(qvec_param[ii].data(), 4);
            problem.AddParameterBlock(tvec_param[ii].data(), 3);
            ii += 1;
        }

        idx2idx.insert(std::make_pair(current_frame_->id_, ii));
        SE3 pose_ = current_frame_->Pose();
        Eigen::Quaterniond q_eigen(pose_.rotationMatrix());
        qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
        tvec_param[ii] = pose_.translation();
        problem.AddParameterBlock(qvec_param[ii].data(), 4);
        problem.AddParameterBlock(tvec_param[ii].data(), 3);

        // Landmark parameters

        std::unordered_map<int, int> idx2idx_lm;
        int jj = 0;
        for (auto &lm : map_->landmarks_)
        {
            idx2idx_lm.insert(std::make_pair(lm.second->id_, jj));
            lm_param[jj] = lm.second->Pos();
            problem.AddParameterBlock(lm_param[jj].data(), 3);
            jj += 1;
        }

        // Add Reprojection factors
        for (auto &mp : map_->landmarks_)
        {
            int lm_id = idx2idx_lm.find(mp.second->id_)->second;
            for (auto &ob : mp.second->observations_)
            {
                auto feature = ob.lock();
                auto item = idx2idx.find(feature->frame_.lock()->id_);
                if (item != idx2idx.end())
                {
                    int frame_id = item->second;
                    Eigen::Vector2d obs_src(ob.lock()->position_.pt.x, ob.lock()->position_.pt.y);
                    // ceres::CostFunction *cost_function = ProjectionFactorSimplePinholeConstantIntrinsic::Create(obs_src, K);
                    // problem.AddResidualBlock(cost_function,
                    //                          new ceres::CauchyLoss(0.5),
                    //                          qvec_param[frame_id].data(),
                    //                          tvec_param[frame_id].data(),
                    //                          lm_param[lm_id].data());
                    ceres::CostFunction *cost_function = ProjectionFactorSimplePinhole::Create(obs_src);
                    problem.AddResidualBlock(cost_function,
                                             new ceres::CauchyLoss(0.5),
                                             qvec_param[frame_id].data(),
                                             tvec_param[frame_id].data(),
                                             lm_param[lm_id].data(),
                                             intrinsic_param[0].data());
                }
            }
        }

        // for (int i = 0; i < window_size_temp; i++)
        // {
        //     problem.SetParameterBlockConstant(qvec_param[i].data());
        //     problem.SetParameterBlockConstant(tvec_param[i].data());
        // }

        // for (int i = 0; i < (int)map_->active_landmarks_.size(); i++)
        // {
        //     problem.SetParameterBlockConstant(lm_param[i].data());
        // }

        // Solve optimization problem

        LOG(INFO) <<"Intrinsic before update(f, cx, cy): "<<intrinsic_param[0][0]<<", "<<intrinsic_param[0][1]<<", "<<intrinsic_param[0][2];
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.num_threads = 8;
        options.max_num_iterations = 10000;
        // options.max_solver_time_in_seconds = 0.04;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        LOG(INFO) <<"Intrinsic after update(f, cx, cy): "<<intrinsic_param[0][0]<<", "<<intrinsic_param[0][1]<<", "<<intrinsic_param[0][2];
        // Update params
        for (auto &kf_map : map_->keyframes_)
        {
            auto kf = kf_map.second;
            auto item = idx2idx.find(kf->id_);
            if (item != idx2idx.end())
            {
                int frame_id = item->second;
                Eigen::Quaterniond Q;
                Vec3 Trans = tvec_param[frame_id];

                Q.w() = qvec_param[frame_id][0];
                Q.x() = qvec_param[frame_id][1];
                Q.y() = qvec_param[frame_id][2];
                Q.z() = qvec_param[frame_id][3];
                kf->SetPose(SE3(Q, Trans));
            }
        }

        for (auto &mp : map_->landmarks_)
        {
            int lm_id = idx2idx_lm.find(mp.second->id_)->second;
            mp.second->pos_ = lm_param[lm_id];
        }
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
