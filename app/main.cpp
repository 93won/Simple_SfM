
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <pangolin/pangolin.h>
#include <pcl/io/pcd_io.h>
#include "utils/Viewer.h"
#include "core/SfM.h"
#include "Config.h"

#include <pangolin/pangolin.h>

DEFINE_string(config_file, "./config/default.yaml", "config file path");

using namespace SFM;

int main(int argc, char **argv)
{

    // Initialize detector, descriptor extractor, the number of features to extract

    SfM::Ptr sfm = SfM::Ptr(new SfM);
    Map::Ptr map = Map::Ptr(new Map);
    

    double fx = Config::Get<double>("camera.fx");
    double fy = Config::Get<double>("camera.fy");
    double cx = Config::Get<double>("camera.cx");
    double cy = Config::Get<double>("camera.cy");
    int max = Config::Get<int>("max");

    Mat33 K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    LOG(INFO) << "Intrinsic (fx, fy, cx, cy) : " << fx << ", " << fy << ", " << cx << ", " << cy;

    Vec3 t_1(0, 0, 0);
    Camera::Ptr camera = Camera::Ptr(new Camera(fx, fy, cx, cy, SE3(SO3(), t_1)));
    Viewer::Ptr viewer = Viewer::Ptr(new Viewer);

    sfm->SetCamera(camera);
    sfm->SetMap(map);
    viewer->SetMap(map);
    sfm->SetViewer(viewer);

    std::vector<SE3> poses;


    for (int i = 0; i < max + 1; i++)
    {
        LOG(INFO) << i << "-th frame";

        int n_zero = 0;
        std::string original_string = std::to_string(i);
        std::string dest = std::string(n_zero, '0').append(original_string);
        cv::Mat img = cv::imread(Config::Get<std::string>("rgb_dir") + std::to_string(i) + ".jpg");
        cv::Mat gray;
        cv::cvtColor(img, gray, CV_RGB2GRAY);

        Frame::Ptr frame(new Frame(i, 0, SE3(SO3(), t_1), img, gray, K));
        sfm->AddFrame(frame);

    }

    return 0;
}
