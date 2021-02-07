#include "descriptor_BRIEF.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>

inline int smoothedSum(const cv::Mat &sum, const cv::KeyPoint &pt, int y, int x, bool use_orientation, cv::Matx21f R)
{
    // static const int HALF_KERNEL = BriefDescriptorExtractorImpl::KERNEL_SIZE / 2;
    static const int HALF_KERNEL = 9 / 2;

    if (use_orientation)
    {
        int rx = (int)(((float)x) * R(1, 0) - ((float)y) * R(0, 0));
        int ry = (int)(((float)x) * R(0, 0) + ((float)y) * R(1, 0));
        if (rx > 24)
            rx = 24;
        if (rx < -24)
            rx = -24;
        if (ry > 24)
            ry = 24;
        if (ry < -24)
            ry = -24;
        x = rx;
        y = ry;
    }
    const int img_y = (int)(pt.pt.y + 0.5) + y;
    const int img_x = (int)(pt.pt.x + 0.5) + x;
    return sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1) - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL) - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1) + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}

static void pixelTests16(cv::InputArray _sum, const std::vector<cv::KeyPoint> &keypoints, cv::OutputArray _descriptors, bool use_orientation)
{
    cv::Matx21f R;
    cv::Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar *desc = descriptors.ptr(static_cast<int>(i));
        const cv::KeyPoint &pt = keypoints[i];
        if (use_orientation)
        {
            float angle = pt.angle;
            angle *= (float)(CV_PI / 180.f);
            R(0, 0) = sin(angle);
            R(1, 0) = cos(angle);
        }

#include "xgenerated_16.i"
    }
}

static void pixelTests32(cv::InputArray _sum, const std::vector<cv::KeyPoint> &keypoints, cv::OutputArray _descriptors, bool use_orientation)
{
    cv::Matx21f R;
    cv::Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar *desc = descriptors.ptr(static_cast<int>(i));
        const cv::KeyPoint &pt = keypoints[i];
        if (use_orientation)
        {
            float angle = pt.angle;
            angle *= (float)(CV_PI / 180.f);
            R(0, 0) = sin(angle);
            R(1, 0) = cos(angle);
        }

#include "xgenerated_32.i"
    }
}

static void pixelTests64(cv::InputArray _sum, const std::vector<cv::KeyPoint> &keypoints, cv::OutputArray _descriptors, bool use_orientation)
{
    cv::Matx21f R;
    cv::Mat sum = _sum.getMat(), descriptors = _descriptors.getMat();
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        uchar *desc = descriptors.ptr(static_cast<int>(i));
        const cv::KeyPoint &pt = keypoints[i];
        if (use_orientation)
        {
            float angle = pt.angle;
            angle *= (float)(CV_PI / 180.f);
            R(0, 0) = sin(angle);
            R(1, 0) = cos(angle);
        }

#include "xgenerated_64.i"
    }
}

void calDescriptorsbyBRIEF(cv::InputArray image,
                           std::vector<cv::KeyPoint> &keypoints,
                           cv::OutputArray descriptors)
{
    typedef void (*PixelTestFn)(cv::InputArray, const std::vector<cv::KeyPoint> &, cv::OutputArray, bool use_orientation);
    PixelTestFn test_fn_;

    int bytes = 64;
    bool use_orientation = false;

    switch (bytes)
    {
    case 16:
        test_fn_ = pixelTests16;
        break;
    case 32:
        test_fn_ = pixelTests32;
        break;
    case 64:
        test_fn_ = pixelTests64;
        break;
    default:
        CV_Error(cv::Error::StsBadArg, "bytes must be 16, 32, or 64");
    }

    // Construct integral image for fast smoothing (box filter)
    cv::Mat sum;

    cv::Mat grayImage = image.getMat();
    if (image.type() != CV_8U)
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    ///TODO allow the user to pass in a precomputed integral image
    //if(image.type() == CV_32S)
    //  sum = image;
    //else

    cv::integral(grayImage, sum, CV_32S);

    //Remove keypoints very close to the border
    cv::KeyPointsFilter::runByImageBorder(keypoints, image.size(), PATCH_SIZE / 2 + KERNEL_SIZE / 2);

    descriptors.create((int)keypoints.size(), bytes, CV_8U);
    descriptors.setTo(cv::Scalar::all(0));
    test_fn_(sum, keypoints, descriptors, use_orientation);
}