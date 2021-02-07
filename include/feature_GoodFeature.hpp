#ifndef FEATURE_GOOD_FEATURE_HPP
#define FEATURE_GOOD_FEATURE_HPP

#include "opencv2/core.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

CV_EXPORTS_W void yuchengcornerMinEigenVal(InputArray src, OutputArray dst,
                                           int blockSize, int ksize = 3,
                                           int borderType = BORDER_DEFAULT);

CV_EXPORTS_W void getGoodFeatures(InputArray image, OutputArray corners,
                                  int maxCorners, double qualityLevel, double minDistance,
                                  InputArray mask = noArray(), int blockSize = 3,
                                  bool useHarrisDetector = false, double k = 0.04);

CV_EXPORTS_W void getGoodFeatures(InputArray image, OutputArray corners,
                                  int maxCorners, double qualityLevel, double minDistance,
                                  InputArray mask, int blockSize,
                                  int gradientSize, bool useHarrisDetector = false,
                                  double k = 0.04);

void calKeyPointbyGoodFeatures(cv::InputArray _image,
                               std::vector<cv::Point2f> corners,
                               std::vector<cv::KeyPoint> &keypoints);

#endif // ! FEATURE_GOOD_FEATURE_HPP
