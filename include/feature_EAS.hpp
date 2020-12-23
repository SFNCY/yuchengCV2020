#ifndef FEATURE_EAS_HPP
#define FEATURE_EAS_HPP

#include "opencv2/core.hpp"
#include "opencv2/xfeatures2d.hpp"


void calKeyPointbyEAS(cv::InputArray _image,
                       std::vector<cv::Point2f> corners,
                       std::vector<cv::KeyPoint> &keypoints);


#endif // !FEATURE_EAS_HPP