#ifndef FEATURE_SIFT_HPP
#define FEATURE_SIFT_HPP

#include "opencv2/core.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;


// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#define DoG_TYPE_SHORT 0
#if DoG_TYPE_SHORT
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

void calKeyPointbySIFT(InputArray _image,
                       std::vector<cv::Point2f> corners,
                       std::vector<KeyPoint> &keypoints);

void calDescriptorsbySIFT(InputArray _image,
                          std::vector<KeyPoint> &keypoints,
                          OutputArray _descriptors);

void calKeypointswithDescriptorsbySIFT(InputArray _image,
                                       std::vector<KeyPoint> &keypoints,
                                       OutputArray _descriptors);

bool removeNoiseByStd(std::vector<cv::Point2f> &temp_point_vec, std::vector<cv::Point2f> &current_point_vec);

#endif // !FEATURE_SIFT_HPP
