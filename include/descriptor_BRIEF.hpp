#ifndef DESCRIPTOR_BRIEF_HPP
#define DESCRIPTOR_BRIEF_HPP
#include "opencv2/core.hpp"

enum { PATCH_SIZE = 48, KERNEL_SIZE = 9 };

void calDescriptorsbyBRIEF(cv::InputArray image,
                           std::vector<cv::KeyPoint> &keypoints,
                           cv::OutputArray descriptors);
#endif // !DESCRIPTOR_BRIEF_HPP
