#ifndef  FEATURE_GOOD_FEATURE_HPP
#define  FEATURE_GOOD_FEATURE_HPP

#include "opencv2/core.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

CV_EXPORTS_W void getGoodFeatures(InputArray image, OutputArray corners,
                                  int maxCorners, double qualityLevel, double minDistance,
                                  InputArray mask = noArray(), int blockSize = 3,
                                  bool useHarrisDetector = false, double k = 0.04);

CV_EXPORTS_W void getGoodFeatures(InputArray image, OutputArray corners,
                                  int maxCorners, double qualityLevel, double minDistance,
                                  InputArray mask, int blockSize,
                                  int gradientSize, bool useHarrisDetector = false,
                                  double k = 0.04);

#endif // ! FEATURE_GOOD_FEATURE_HPP

