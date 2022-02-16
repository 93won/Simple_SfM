#pragma once
#ifndef SFM_CONFIG_H
#define SFM_CONFIG_H

#include "types/Common.h"

namespace SFM
{

    class Config
    {
    private:
        static std::shared_ptr<Config> config_;
        cv::FileStorage file_;

        Config() {} // private constructor makes a singleton
    public:
        ~Config(); // close the file when deconstructing

        // set a new config file
        static bool SetParameterFile(const std::string &filename);

        // access the parameter values
        template <typename T>
        static T Get(const std::string &key)
        {
            return T(Config::config_->file_[key]);
        }
    };
}

#endif