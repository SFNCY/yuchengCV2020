#include "feature_EAS.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <iostream>

void calKeyPointbyEAS(cv::InputArray _image,
                      std::vector<cv::Point2f> corners,
                      std::vector<cv::KeyPoint> &keypoints)
{
    if (_image.empty())
    {
        keypoints.clear();
        return;
    }

    cv::Mat image = _image.getMat();
    cv::Mat dx, dy;

    cv::Sobel(image, dx, CV_8UC1, 1, 0, 3);
    cv::Sobel(image, dy, CV_8UC1, 0, 1, 3);

    cv::namedWindow("dx", cv::WINDOW_NORMAL);
    cv::namedWindow("dy", cv::WINDOW_NORMAL);
    cv::namedWindow("B", cv::WINDOW_NORMAL);
    int r = 2;
    std::vector<std::vector<double>> d;
    std::cout << "image.cols" << image.cols << std::endl;
    std::cout << "image.rows" << image.rows << std::endl;

    for (int xCol = r; xCol < image.cols - r; xCol++)
    {
        for (int yRow = r; yRow < image.rows - r; yRow++)
        {
            cv::Rect area = cv::Rect(xCol - r, yRow - r, r, r);
            cv::Mat LT = image(area);
            area = cv::Rect(xCol + 1, yRow + 1, r, r);
            cv::Mat RB = image(area);

            area = cv::Rect(xCol + 1, yRow - r, r, r);
            cv::Mat RT = image(area);
            area = cv::Rect(xCol - r, yRow + 1, r, r);
            cv::Mat LB = image(area);

            area = cv::Rect(xCol - r, yRow - round((r - 1)) / 2, r, r);
            cv::Mat L = image(area);
            area = cv::Rect(xCol + 1, yRow - round((r - 1)) / 2, r, r);
            cv::Mat R = image(area);

            area = cv::Rect(xCol - round((r - 1)) / 2, yRow - r, r, r);
            cv::Mat T = image(area);
            area = cv::Rect(xCol - round((r - 1)) / 2, yRow + 1, r, r);
            cv::Mat B = image(area);

            // std::cout << "det_B:" << B.rows << " " << B.cols << std::endl;

            B.convertTo(B, CV_64F);
            cv::Scalar mean;
            cv::Scalar dev;

            cv::meanStdDev(B, mean, dev);
            float m = mean.val[0];
            float s = dev.val[0];

            std::cout << "mean:" << m << std::endl;
            std::cout << "std:" << s << std::endl;
            cv::imshow("B", B);
            cv::waitKey(0);
        }
    }

    cv::imshow("dx", dx);
    cv::imshow("dy", dy);
    cv::waitKey(0);
    cv::destroyAllWindows();
}