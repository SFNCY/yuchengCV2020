#include "feature_EAS.hpp"
#include "feature_SIFT.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <numeric>
#include <map>

#include <time.h>
#include <sys/time.h>

using namespace cv;

static inline void
EASunpackOctave(const KeyPoint &kpt, int &octave, int &layer, float &scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}

static Mat EAScreateInitialImage(const Mat &img, bool doubleImageSize, float sigma)
{
    Mat gray, gray_fpt;
    if (img.channels() == 3 || img.channels() == 4)
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
    }
    else
        img.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;

    if (doubleImageSize)
    {
        sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
        Mat dbl;
#if DoG_TYPE_SHORT
        resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR_EXACT);
#else
        resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR);
#endif
        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        return dbl;
    }
    else
    {
        sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
        GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        return gray_fpt;
    }
}

void EASbuildGaussianPyramid(const Mat &base, std::vector<Mat> &pyr, int nOctaves)
{
    int nOctaveLayers = 3;
    double sigma = 1.6;

    std::vector<double>
        sig(nOctaveLayers + 3);
    pyr.resize(nOctaves * (nOctaveLayers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow(2., 1. / nOctaveLayers);
    for (int i = 1; i < nOctaveLayers + 3; i++)
    {
        double sig_prev = std::pow(k, (double)(i - 1)) * sigma;
        double sig_total = sig_prev * k;
        sig[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
    }

    for (int o = 0; o < nOctaves; o++)
    {
        for (int i = 0; i < nOctaveLayers + 3; i++)
        {
            Mat &dst = pyr[o * (nOctaveLayers + 3) + i];
            if (o == 0 && i == 0)
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if (i == 0)
            {
                const Mat &src = pyr[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols / 2, src.rows / 2),
                       0, 0, INTER_NEAREST);
            }
            else
            {
                const Mat &src = pyr[o * (nOctaveLayers + 3) + i - 1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]);
            }
        }
    }
}

class EASbuildDoGPyramidComputer : public ParallelLoopBody
{
public:
    EASbuildDoGPyramidComputer(
        int _nOctaveLayers,
        const std::vector<Mat> &_gpyr,
        std::vector<Mat> &_dogpyr)
        : nOctaveLayers(_nOctaveLayers),
          gpyr(_gpyr),
          dogpyr(_dogpyr) {}

    void operator()(const cv::Range &range) const CV_OVERRIDE
    {
        const int begin = range.start;
        const int end = range.end;

        for (int a = begin; a < end; a++)
        {
            const int o = a / (nOctaveLayers + 2);
            const int i = a % (nOctaveLayers + 2);

            const Mat &src1 = gpyr[o * (nOctaveLayers + 3) + i];
            const Mat &src2 = gpyr[o * (nOctaveLayers + 3) + i + 1];
            Mat &dst = dogpyr[o * (nOctaveLayers + 2) + i];
            subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
        }
    }

private:
    int nOctaveLayers;
    const std::vector<Mat> &gpyr;
    std::vector<Mat> &dogpyr;
};

void EASbuildDoGPyramid(const std::vector<Mat> &gpyr, std::vector<Mat> &dogpyr)
{
    int nOctaveLayers = 3;
    int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
    dogpyr.resize(nOctaves * (nOctaveLayers + 2));

    parallel_for_(Range(0, nOctaves * (nOctaveLayers + 2)), EASbuildDoGPyramidComputer(nOctaveLayers, gpyr, dogpyr));
}

//根据second的值升序排序
bool cmp_y(std::pair<int, int> a, std::pair<int, int> b)
{
    return a.second < b.second;
}

inline double calCovarTrace(std::vector<char> dx, std::vector<char> dy)
{
    double sum_dx = std::accumulate(std::begin(dx), std::end(dx), 0.0);
    double mean_dx = sum_dx / dx.size(); //均值

    double accum_dx = 0.0;
    std::for_each(std::begin(dx), std::end(dx), [&](const double d_dx) {
        accum_dx += (d_dx - mean_dx) * (d_dx - mean_dx);
    });

    double var_dx = accum_dx / (dx.size() - 1);

    double sum_dy = std::accumulate(std::begin(dy), std::end(dy), 0.0);
    double mean_dy = sum_dy / dy.size(); //均值

    double accum_dy = 0.0;
    std::for_each(std::begin(dy), std::end(dy), [&](const double d_dy) {
        accum_dy += (d_dy - mean_dy) * (d_dy - mean_dy);
    });

    double var_dy = accum_dy / (dy.size() - 1);

    return var_dx + var_dy;
}

void findScalSapceEAS(const std::vector<Mat> &gauss_pyr, std::vector<KeyPoint> &keypoints)
{
    int nOctaveLayers = 3;
    const int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);

    keypoints.clear();

    for (int o = 0; o < nOctaves - 3; o++)
        for (int i = 0; i < nOctaveLayers - 1; i++)
        {
    // for (int o = 0; o < 1; o++)
    //     for (int i = 0; i < 1; i++)
    //     {
            const int idx = o * (nOctaveLayers + 3) + i;
            const Mat &img = gauss_pyr[idx];

            int r = 2 ^ (nOctaves - o);
            double EAS_thred = 1000.0;
            // if (img.empty())
            // {
            //     keypoints.clear();
            //     return;
            // }

            cv::Mat image = img.clone();
            cv::Mat dx, dy;
            cv::Mat src = cv::Mat::zeros(image.rows, image.cols, CV_64F);
            cv::Mat dst = cv::Mat::zeros(image.rows, image.cols, CV_8U);

            cv::Sobel(image, dx, CV_8UC1, 1, 0, 3);
            cv::Sobel(image, dy, CV_8UC1, 0, 1, 3);

            // cv::namedWindow("dx", cv::WINDOW_NORMAL);
            // cv::namedWindow("dy", cv::WINDOW_NORMAL);
            // cv::namedWindow("src", cv::WINDOW_NORMAL);

            std::vector<std::vector<double>> patch;

            int rows = image.rows;
            int cols = image.cols;

            std::map<double, cv::Point> EAS_map;

            //calculate EAS by matrix
            for (int yRow = r; yRow < image.rows - r; yRow++)
            {
                for (int xCol = r; xCol < image.cols - r; xCol++)
                {
                    cv::Rect area = cv::Rect(xCol - r, yRow - r, r, r);
                    cv::Mat LT_dx = dx(area);
                    cv::Mat LT_dy = dy(area);
                    area = cv::Rect(xCol + 1, yRow + 1, r, r);
                    cv::Mat RB_dx = dx(area);
                    cv::Mat RB_dy = dy(area);

                    area = cv::Rect(xCol + 1, yRow - r, r, r);
                    cv::Mat RT_dx = dx(area);
                    cv::Mat RT_dy = dy(area);
                    area = cv::Rect(xCol - r, yRow + 1, r, r);
                    cv::Mat LB_dx = dx(area);
                    cv::Mat LB_dy = dy(area);

                    area = cv::Rect(xCol - r, yRow - round((r - 1)) / 2, r, r);
                    cv::Mat L_dx = dx(area);
                    cv::Mat L_dy = dy(area);
                    area = cv::Rect(xCol + 1, yRow - round((r - 1)) / 2, r, r);
                    cv::Mat R_dx = dx(area);
                    cv::Mat R_dy = dy(area);

                    area = cv::Rect(xCol - round((r - 1)) / 2, yRow - r, r, r);
                    cv::Mat T_dx = dx(area);
                    cv::Mat T_dy = dy(area);
                    area = cv::Rect(xCol - round((r - 1)) / 2, yRow + 1, r, r);
                    cv::Mat B_dx = dx(area);
                    cv::Mat B_dy = dy(area);

                    // B.convertTo(B, CV_64F);
                    cv::Scalar LT_mean_dx, LT_mean_dy;
                    cv::Scalar RB_mean_dx, RB_mean_dy;
                    cv::Scalar RT_mean_dx, RT_mean_dy;
                    cv::Scalar LB_mean_dx, LB_mean_dy;
                    cv::Scalar L_mean_dx, L_mean_dy;
                    cv::Scalar R_mean_dx, R_mean_dy;
                    cv::Scalar T_mean_dx, T_mean_dy;
                    cv::Scalar B_mean_dx, B_mean_dy;

                    cv::Scalar LT_dev_dx, LT_dev_dy;
                    cv::Scalar RB_dev_dx, RB_dev_dy;
                    cv::Scalar RT_dev_dx, RT_dev_dy;
                    cv::Scalar LB_dev_dx, LB_dev_dy;
                    cv::Scalar L_dev_dx, L_dev_dy;
                    cv::Scalar R_dev_dx, R_dev_dy;
                    cv::Scalar T_dev_dx, T_dev_dy;
                    cv::Scalar B_dev_dx, B_dev_dy;

                    cv::meanStdDev(LT_dx, LT_mean_dx, LT_dev_dx);
                    cv::meanStdDev(LT_dy, LT_mean_dy, LT_dev_dy);
                    cv::meanStdDev(RB_dx, RB_mean_dx, RB_dev_dx);
                    cv::meanStdDev(RB_dy, RB_mean_dy, RB_dev_dy);
                    cv::meanStdDev(RT_dx, RT_mean_dx, RT_dev_dx);
                    cv::meanStdDev(RT_dy, RT_mean_dy, RT_dev_dy);
                    cv::meanStdDev(LB_dx, LB_mean_dx, LB_dev_dx);
                    cv::meanStdDev(LB_dy, LB_mean_dy, LB_dev_dy);

                    cv::meanStdDev(L_dx, L_mean_dx, L_dev_dx);
                    cv::meanStdDev(L_dy, L_mean_dy, L_dev_dy);
                    cv::meanStdDev(R_dx, R_mean_dx, R_dev_dx);
                    cv::meanStdDev(R_dy, R_mean_dy, R_dev_dy);
                    cv::meanStdDev(T_dx, T_mean_dx, T_dev_dx);
                    cv::meanStdDev(T_dy, T_mean_dy, T_dev_dy);
                    cv::meanStdDev(B_dx, B_mean_dx, B_dev_dx);
                    cv::meanStdDev(B_dy, B_mean_dy, B_dev_dy);
                    //边缘抑制
                    // if (((LT_dev_dx.val[0] * LT_dev_dx.val[0]) / (LT_dev_dy.val[0] * LT_dev_dy.val[0])) > 5.0 ||
                    //     ((LT_dev_dx.val[0] * LT_dev_dx.val[0]) / (LT_dev_dy.val[0] * LT_dev_dy.val[0])) < 1 / 5.0||
                    //     ((RB_dev_dx.val[0] * RB_dev_dx.val[0]) / (RB_dev_dy.val[0] * RB_dev_dy.val[0])) > 5.0 ||
                    //     ((RB_dev_dx.val[0] * RB_dev_dx.val[0]) / (RB_dev_dy.val[0] * RB_dev_dy.val[0])) < 1 / 5.0||
                    //     ((LB_dev_dx.val[0] * LB_dev_dx.val[0]) / (LB_dev_dy.val[0] * LB_dev_dy.val[0])) > 5.0 ||
                    //     ((LB_dev_dx.val[0] * LB_dev_dx.val[0]) / (LB_dev_dy.val[0] * LB_dev_dy.val[0])) < 1 / 5.0||
                    //     ((RT_dev_dx.val[0] * RT_dev_dx.val[0]) / (RT_dev_dy.val[0] * RT_dev_dy.val[0])) > 5.0 ||
                    //     ((RT_dev_dx.val[0] * RT_dev_dx.val[0]) / (RT_dev_dy.val[0] * RT_dev_dy.val[0])) < 1 / 5.0||
                    //     ((L_dev_dx.val[0] * L_dev_dx.val[0]) / (L_dev_dy.val[0] * L_dev_dy.val[0])) > 5.0 ||
                    //     ((L_dev_dx.val[0] * L_dev_dx.val[0]) / (L_dev_dy.val[0] * L_dev_dy.val[0])) < 1 / 5.0||
                    //     ((R_dev_dx.val[0] * R_dev_dx.val[0]) / (R_dev_dy.val[0] * R_dev_dy.val[0])) > 5.0 ||
                    //     ((R_dev_dx.val[0] * R_dev_dx.val[0]) / (R_dev_dy.val[0] * R_dev_dy.val[0])) < 1 / 5.0||
                    //     ((T_dev_dx.val[0] * T_dev_dx.val[0]) / (T_dev_dy.val[0] * T_dev_dy.val[0])) > 5.0 ||
                    //     ((T_dev_dx.val[0] * T_dev_dx.val[0]) / (T_dev_dy.val[0] * T_dev_dy.val[0])) < 1 / 5.0||
                    //     ((B_dev_dx.val[0] * B_dev_dx.val[0]) / (B_dev_dy.val[0] * B_dev_dy.val[0])) > 5.0 ||
                    //     ((B_dev_dx.val[0] * B_dev_dx.val[0]) / (B_dev_dy.val[0] * B_dev_dy.val[0])) < 1 / 5.0)
                    //     {
                    //         continue;
                    //     }
                    double EAS = (fabs((LT_dev_dx.val[0] * LT_dev_dx.val[0] + LT_dev_dy.val[0] * LT_dev_dy.val[0]) -
                                       (RB_dev_dx.val[0] * RB_dev_dx.val[0] + RB_dev_dy.val[0] * RB_dev_dy.val[0])) +
                                  fabs((LB_dev_dx.val[0] * LB_dev_dx.val[0] + LB_dev_dy.val[0] * LB_dev_dy.val[0]) -
                                       (RT_dev_dx.val[0] * RT_dev_dx.val[0] + RT_dev_dy.val[0] * RT_dev_dy.val[0])) +
                                  fabs((L_dev_dx.val[0] * L_dev_dx.val[0] + L_dev_dy.val[0] * L_dev_dy.val[0]) -
                                       (R_dev_dx.val[0] * R_dev_dx.val[0] + R_dev_dy.val[0] * R_dev_dy.val[0])) +
                                  fabs((T_dev_dx.val[0] * T_dev_dx.val[0] + T_dev_dy.val[0] * T_dev_dy.val[0]) -
                                       (B_dev_dx.val[0] * B_dev_dx.val[0] + B_dev_dy.val[0] * B_dev_dy.val[0]))) /
                                 4;

                    // src.at<double>(yRow, xCol) = EAS;
                    if (EAS > EAS_thred)
                    {
                        src.at<double>(yRow, xCol) = EAS;
                        // cv::Point temp_circle(xCol * 1 / 2 * (1 << o), yRow * 1 / 2 * (1 << o));
                        // cv::KeyPoint temp_keypoint(temp_circle, 1);
                        // EAS_map.insert(std::make_pair(EAS, temp_circle));
                    }
                }
            }

            //非极大值抑制
            double max_local_EAS = 0.0;
            int max_local_EAS_x_col, max_local_EAS_y_row;
            int iCol, iRow;
            r = 2;
            for (int xCol = r; xCol < image.cols - r; xCol = xCol + 2 * r + 1)
            {
                for (int yRow = r; yRow < image.rows - r; yRow = yRow + 2 * r + 1)
                {
                    max_local_EAS = EAS_thred;
                    for (iCol = xCol - r; iCol < xCol + r + 1; iCol++)
                    {
                        for (iRow = yRow - r; iRow < yRow + r + 1; iRow++)
                        {
                            if (src.at<double>(iRow, iCol) > max_local_EAS)
                            {
                                max_local_EAS = src.at<double>(iRow, iCol);
                                max_local_EAS_x_col = iCol;
                                max_local_EAS_y_row = iRow;
                            }
                        }
                    }
                    if (max_local_EAS > EAS_thred)
                    {
                        dst.at<char>(max_local_EAS_y_row, max_local_EAS_x_col) = 255;
                        cv::Point temp_circle(max_local_EAS_x_col * 1 / 2 * (1 << o), max_local_EAS_y_row * 1 / 2 * (1 << o));
                        EAS_map.insert(std::make_pair(max_local_EAS, temp_circle));
                    }
                }
            }

            int count = 0;
            std::map<double, cv::Point>::iterator iter_last = EAS_map.end();
            --iter_last;
            // std::cout << "EAS_map.begin()->first:" << iter_last->first << std::endl;
            for (std::map<double, cv::Point>::iterator iter = iter_last;
                 iter != EAS_map.begin();
                 --iter)
            {
                if (count < 500)
                {
                    // std::cout << "EAS_map.begin()->first:" << iter_last->first << std::endl;
                    // std::cout << "iter->first:" << iter->first << std::endl;
                    cv::KeyPoint temp_keypoint(iter->second, 50 * iter->first / iter_last->first);
                    keypoints.push_back(temp_keypoint);
                }
                else
                {
                    break;
                }
                count++;
            }
            std::cout << "keypoints.size()" << keypoints.size() << std::endl;
            int a = 2;

            /*//calculate EAS by manual
    for (int yRow = r; yRow < image.rows - r; yRow++)
    {
        for (int xCol = r; xCol < image.cols - r; xCol++)
        {
            //LT:left top
            std::vector<char> patch_LT_dx, patch_LT_dy;
            for (int jRow = yRow - r; jRow < yRow; jRow++)
            {
                for (int iCol = xCol - r; iCol < xCol; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_LT_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_LT_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_LT=calCovarTrace(patch_LT_dx, patch_LT_dy);

            //RB:right bottom
            std::vector<char> patch_RB_dx, patch_RB_dy;
            for (int jRow = yRow + 1; jRow < yRow + r + 1; jRow++)
            {
                for (int iCol = xCol + 1; iCol < xCol + r + 1; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_RB_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_RB_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_RB = calCovarTrace(patch_RB_dx, patch_RB_dy);

            //RT:right top
            std::vector<char> patch_RT_dx, patch_RT_dy;
            for (int jRow = yRow - r; jRow < yRow; jRow++)
            {
                for (int iCol = xCol + 1; iCol < xCol + r + 1; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_RT_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_RT_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_RT = calCovarTrace(patch_RT_dx, patch_RT_dy);

            //LB:left bottom
            std::vector<char> patch_LB_dx, patch_LB_dy;
            for (int jRow = yRow + 1; jRow < yRow + r + 1; jRow++)
            {
                for (int iCol = xCol - r; iCol < xCol; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_LB_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_LB_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_LB = calCovarTrace(patch_LB_dx, patch_LB_dy);

            //T:top
            std::vector<char> patch_T_dx, patch_T_dy;
            for (int jRow = yRow - r; jRow < yRow; jRow++)
            {
                for (int iCol = xCol - round((r - 1)) / 2; iCol < xCol + round((r - 1)) / 2; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_T_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_T_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_T = calCovarTrace(patch_T_dx, patch_T_dy);

            //B:bottom
            std::vector<char> patch_B_dx, patch_B_dy;
            for (int jRow = yRow + 1; jRow < yRow + r + 1; jRow++)
            {
                for (int iCol = xCol - round((r - 1)) / 2; iCol < xCol + round((r - 1)) / 2; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_B_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_B_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_B = calCovarTrace(patch_B_dx, patch_B_dy);

            //L:left
            std::vector<char> patch_L_dx, patch_L_dy;
            for (int jRow = yRow - round((r - 1)) / 2; jRow < yRow + round((r - 1)) / 2; jRow++)
            {
                for (int iCol = xCol - r; iCol < xCol; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_L_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_L_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_L = calCovarTrace(patch_L_dx, patch_L_dy);

            //R:right
            std::vector<char> patch_R_dx, patch_R_dy;
            for (int jRow = yRow - round((r - 1)) / 2; jRow < yRow + round((r - 1)) / 2; jRow++)
            {
                for (int iCol = xCol + 1; iCol < xCol + r + 1; iCol++)
                {
                    // double val_x = dx.at<char>(jRow, iCol);
                    // double val_y = dy.at<char>(jRow, iCol);
                    // std::cout << "val_x:" << val_x << std::endl;
                    // std::cout << "val_y:" << val_y << std::endl;
                    patch_R_dx.push_back(dx.at<char>(jRow, iCol));
                    patch_R_dy.push_back(dy.at<char>(jRow, iCol));
                }
            }
            double CovarTrace_R = calCovarTrace(patch_R_dx, patch_R_dy);

            double EAS = (fabs(CovarTrace_LT - CovarTrace_RB) + fabs(CovarTrace_RT - CovarTrace_LB) + fabs(CovarTrace_L - CovarTrace_R) + fabs(CovarTrace_T - CovarTrace_B)) / 4;

            if (EAS > EAS_thred)
            {
                src.at<double>(yRow, xCol) = EAS;
                // cv::Point temp_circle(xCol, yRow);
                // cv::circle(image, temp_circle, 1, cv::Scalar(255, 255, 255));
                // dst.at<char>(yRow, xCol) = 255;
            }
        }
    }*/

            // //非极大值抑制
            // double max_local_EAS = 0.0;
            // int max_local_EAS_x_col, max_local_EAS_y_row;
            // int iCol, iRow;
            // r = 10;
            // for (int xCol = r; xCol < image.cols - r; xCol = xCol + 2 * r + 1)
            // {
            //     for (int yRow = r; yRow < image.rows - r; yRow = yRow + 2 * r + 1)
            //     {
            //         max_local_EAS = EAS_thred;
            //         for (iCol = xCol - r; iCol < xCol + r + 1; iCol++)
            //         {
            //             for (iRow = yRow - r; iRow < yRow + r + 1; iRow++)
            //             {
            //                 if (src.at<double>(iRow, iCol) > max_local_EAS)
            //                 {
            //                     max_local_EAS = src.at<double>(iRow, iCol);
            //                     max_local_EAS_x_col = iCol;
            //                     max_local_EAS_y_row = iRow;
            //                 }
            //             }
            //         }
            //         if (max_local_EAS > EAS_thred)
            //         {
            //             dst.at<char>(max_local_EAS_y_row, max_local_EAS_x_col) = 255;
            //             cv::Point temp_circle(xCol, yRow);
            //             cv::KeyPoint temp_keypoint(temp_circle, max_local_EAS / 1000);
            //             keypoints.push_back(temp_keypoint);
            //         }
            //     }
            // }

            // cv::imshow("dst", dst);
            // cv::waitKey(0);

            // cv::imshow("dx", dx);
            // cv::imshow("dy", dy);
            // cv::waitKey(0);
            // cv::destroyAllWindows();
        }
}

void calKeyPointbyEAS(cv::InputArray _image,
                      std::vector<cv::Point2f> corners,
                      std::vector<cv::KeyPoint> &keypoints)
{
    double sigma = 1.6;
    int nfeatures = 0;

    if (_image.empty())
    {
        keypoints.clear();
        return;
    }

    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    cv::Mat image = _image.getMat();

    if (image.empty() || image.depth() != CV_8U)
        CV_Error(Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

    cv::Mat base = EAScreateInitialImage(image, firstOctave < 0, (float)sigma);

    std::vector<Mat> gpyr, dogpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log((double)std::min(base.cols, base.rows)) / std::log(2.) - 2) - firstOctave;

    EASbuildGaussianPyramid(base, gpyr, nOctaves);

    // gpyr.push_back(_image.getMat());

    findScalSapceEAS(gpyr, keypoints);
}