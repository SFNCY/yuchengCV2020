#include "feature_EAS.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <numeric>
#include <map>

#include <time.h>
#include <sys/time.h>

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

void calKeyPointbyEAS(cv::InputArray _image,
                      std::vector<cv::Point2f> corners,
                      std::vector<cv::KeyPoint> &keypoints)
{
    int r = 5;
    double EAS_thred = 5000.0;
    if (_image.empty())
    {
        keypoints.clear();
        return;
    }

    cv::Mat image = _image.getMat();
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
                cv::Point temp_circle(xCol, yRow);
                cv::KeyPoint temp_keypoint(temp_circle, 1);
                // EAS_map.insert(std::make_pair(EAS, temp_circle));
            }
        }
    }
    // int count = 0;
    // std::map<double, cv::Point>::iterator iter_last = EAS_map.end();
    // --iter_last;
    // // std::cout << "EAS_map.begin()->first:" << iter_last->first << std::endl;
    // for (std::map<double, cv::Point>::iterator iter = iter_last;
    //      iter != EAS_map.begin();
    //      --iter)
    // {
    //     if (count < 500)
    //     {
    //         // std::cout << "EAS_map.begin()->first:" << iter_last->first << std::endl;
    //         // std::cout << "iter->first:" << iter->first << std::endl;
    //         cv::KeyPoint temp_keypoint(iter->second, 50 * iter->first / iter_last->first);
    //         keypoints.push_back(temp_keypoint);
    //     }
    //     else
    //     {
    //         break;
    //     }
    //     count++;
    // }

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

    //非极大值抑制
    double max_local_EAS = 0.0;
    int max_local_EAS_x_col, max_local_EAS_y_row;
    int iCol, iRow;
    r = 10;
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
                cv::Point temp_circle(xCol, yRow);
                cv::KeyPoint temp_keypoint(temp_circle, max_local_EAS / 1000);
                keypoints.push_back(temp_keypoint);
            }
        }
    }

    // cv::imshow("dst", dst);
    // cv::waitKey(0);

    // cv::imshow("dx", dx);
    // cv::imshow("dy", dy);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
}