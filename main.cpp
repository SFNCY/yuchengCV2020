#include <iostream>
#include <sstream>
#include <numeric>

#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/flann/flann.hpp>

#include "feature_SIFT.hpp"
#include "track_KLT.hpp"

#include <time.h>
#include <sys/time.h>

template <class Type>
std::string num2str(Type num)
{
    std::stringstream ss;
    ss << num;
    return ss.str();
}

class xFeature
{
public:
    std::vector<Point2f> temp_track_points, current_track_points;
    std::vector<cv::KeyPoint> keyPoint1, keyPoint2;
    std::vector<cv::Point2f> corners1, corners2;
    // bool track_failed = true;
    // cv::Mat prevGray;

    bool updateSIFTFeature(cv::Mat temp, cv::Mat current)
    {
        temp_track_points.clear();
        current_track_points.clear();
        // if (prevGray.empty())
        //     current.copyTo(prevGray);
        keyPoint1.clear();
        keyPoint2.clear();

        // timeval t_start, t_end;
        // clock_t startTime, endTime;
        // gettimeofday(&t_start, NULL);
        // startTime = clock();
        calKeyPointbySIFT(temp, corners1, keyPoint1);
        calKeyPointbySIFT(current, corners2, keyPoint2);
        // endTime = clock();
        // gettimeofday(&t_end, NULL);
        // std::cout << "clock_t Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        // double delta_t = (t_end.tv_sec - t_start.tv_sec) +
        //                  (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
        // std::cout << "multi-thread time : " << delta_t << "s" << std::endl;

        if (keyPoint1.size() < 6 || keyPoint2.size() < 6)
        {
            return false;
        }

        // cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        Mat des1, des2;

        // descriptor->compute(temp, keyPoint1, des1);
        // descriptor->compute(current, keyPoint2, des2);

        calDescriptorsbySIFT(temp, keyPoint1, des1);
        calDescriptorsbySIFT(current, keyPoint2, des2);

        cv::Ptr<cv::DescriptorMatcher>
            matcher = cv::DescriptorMatcher::create("FlannBased");
        std::vector<std::vector<cv::DMatch>> matches;
        matcher->knnMatch(des1, des2, matches, 2);

        //筛选出较好的匹配点
        std::vector<cv::DMatch> goodMatches;
        for (int m = 0; m < matches.size(); m++)
        {
            const float minRatio = 0.8f;
            const cv::DMatch &bestMatch = matches[m][0];
            const cv::DMatch &betterMatch = matches[m][1];
            float distanceRatio = bestMatch.distance /
                                  betterMatch.distance;
            if (distanceRatio < minRatio)
            {
                goodMatches.push_back(bestMatch);
            }
        }
        std::cout << "The number of good matches:" << goodMatches.size() << std::endl;
        //画出匹配结果
        // Mat img_out;
        //红色连接的是匹配的特征点数，绿色连接的是未匹配的特征点数
        //matchColor – Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1) , the color is generated randomly.
        //singlePointColor – Color of single keypoints(circles), which means that keypoints do not have the matches.If singlePointColor == Scalar::all(-1), the color is generated randomly.
        //CV_RGB(0, 255, 0)存储顺序为R-G-B,表示绿色
        // drawMatches(OriginalGrayImage, keyPoint1, targetGrayImage, keyPoint2, goodMatches, img_out, Scalar::all(-1), CV_RGB(0, 0, 255), Mat(), 2);
        // imshow("Match image", img_out);
        // cv::waitKey(1);

        //RANSAC匹配过程
        std::vector<DMatch> m_Matches;
        m_Matches = goodMatches;
        int ptCount = goodMatches.size();
        // if (ptCount < 20)
        // {
        //     cout << "Don't find enough match points" << endl;
        //     return;
        // }

        //坐标转换为float类型
        std::vector<KeyPoint> RAN_KP1, RAN_KP2;
        //size_t是标准C库中定义的，应为unsigned int，在64位系统中为long unsigned int,在C++中为了适应不同的平台，增加可移植性。
        for (size_t i = 0; i < m_Matches.size(); i++)
        {
            RAN_KP1.push_back(keyPoint1[goodMatches[i].queryIdx]);
            RAN_KP2.push_back(keyPoint2[goodMatches[i].trainIdx]);
            //RAN_KP1是要存储img01中能与img02匹配的点
            //goodMatches存储了这些匹配点对的img01和img02的索引值
        }
        //坐标变换
        std::vector<Point2f> p01, p02;
        for (size_t i = 0; i < m_Matches.size(); i++)
        {
            p01.push_back(RAN_KP1[i].pt);
            p02.push_back(RAN_KP2[i].pt);
        }
        // std::vector<Point2f> img1_corners(4);
        // img1_corners[0] = Point(0, 0);
        // img1_corners[1] = Point(temp.cols, 0);
        // img1_corners[2] = Point(temp.cols, temp.rows);
        // img1_corners[3] = Point(0, temp.rows);
        // std::vector<Point2f> img2_corners(4);
        // 求转换矩阵
        //Mat m_homography;
        //vector<uchar> m;
        //m_homography = findHomography(p01, p02, RANSAC);//寻找匹配图像
        //求基础矩阵 Fundamental,3*3的基础矩阵
        std::vector<uchar> RansacStatus;
        // Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
        if (p01.size() < 6 || p02.size() < 6)
        {
            return false;
        }

        Mat Fundamental = cv::findHomography(p01, p02, RANSAC, 0.5, RansacStatus);

        // perspectiveTransform(img1_corners, img2_corners, Fundamental);
        // //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        // line(current, img2_corners[0],
        //      img2_corners[1], Scalar(0, 255, 0), 4);
        // line(current, img2_corners[1],
        //      img2_corners[2], Scalar(0, 255, 0), 4);
        // line(current, img2_corners[2],
        //      img2_corners[3], Scalar(0, 255, 0), 4);
        // line(current, img2_corners[3],
        //      img2_corners[0], Scalar(0, 255, 0), 4);

        // cv::imshow("track", current);
        // cv::waitKey(1);

        //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
        std::vector<KeyPoint> RR_KP1, RR_KP2;
        std::vector<DMatch> RR_matches;
        int index = 0;
        current_track_points.clear();
        cv::Mat image = current.clone();
        for (size_t i = 0; i < m_Matches.size(); i++)
        {
            if (RansacStatus[i] != 0)
            {
                temp_track_points.push_back(RAN_KP1[i].pt);
                current_track_points.push_back(RAN_KP2[i].pt);
            }
        }

        if (current_track_points.size() < 6)
        {
            return false;
        }

        // if (!removeNoiseByStd(temp_track_points, current_track_points))
        // {
        //     return false;
        // }

        return true;
    }
};

int main()
{
    cv::Mat current_frame;
    cv::Mat prev_frame_gray;
    cv::Mat current_frame_gray;
    std::string base_path = "../data/12_11/60Hz/";
    std::string current_file_path = base_path + "1.bmp";
    current_frame = cv::imread(current_file_path);
    int frame_id = 2;
    bool track_failed = true;

    std::vector<cv::Point2f> temp_track_points, pre_track_points, current_track_points;

    xFeature feature_extractor;

    while (!current_frame.empty())
    {
        //使用灰度图像进行角点检测
        cv::cvtColor(current_frame, current_frame_gray, cv::COLOR_BGR2GRAY);

        cv::Mat temp_frame = cv::imread("../data/temp_color.bmp", -1);
        cv::Mat temp_frame_gray;
        cv::cvtColor(temp_frame, temp_frame_gray, cv::COLOR_BGR2GRAY);

        if (track_failed)
        {
            bool update_secussed = false;
            update_secussed = feature_extractor.updateSIFTFeature(temp_frame_gray, current_frame_gray);

            if (!update_secussed)
            {
                current_file_path = base_path + num2str<int>(frame_id) + ".bmp";
                current_frame = cv::imread(current_file_path);
                frame_id++;
                continue;
            }
            current_frame_gray.copyTo(prev_frame_gray);
            temp_track_points.clear();
            temp_track_points = feature_extractor.temp_track_points;
            pre_track_points = feature_extractor.current_track_points;

            // cornerSubPix(temp_frame_gray, temp_track_points, Size(10,10), Size(-1,-1), TermCriteria(1 | 2, 20, 0.03));
            // cornerSubPix(current_frame_gray, pre_track_points, Size(10,10), Size(-1,-1), TermCriteria(1 | 2, 20, 0.03));
            // current_track_points = feature_extractor.current_track_points;
            track_failed = false;
        }

        // bool extract_successed = feature_extractor.updateSIFTFeature(temp_frame_gray, current_frame_gray);

        // if (extract_successed)
        // {
        // for (int i = 0; i < pre_track_points.size(); i++)
        // {
        //     circle(current_frame, pre_track_points[i], 3, Scalar(0, 255, 0), -1, 8);
        // }
        // cv::imshow("track", current_frame);
        // cv::waitKey(1);
        // }

        std::vector<uchar> status;
        std::vector<float> err;
        TermCriteria termcrit(1 | 2, 20, 0.03);
        Size subPixWinSize(10, 10), winSize(31, 31);

        // timeval t_start, t_end;
        // clock_t startTime, endTime;
        // gettimeofday(&t_start, NULL);
        // startTime = clock();
        yuchengcalcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, pre_track_points, current_track_points, status, err, winSize,
                                    3, termcrit, 0, 0.001);
        // endTime = clock();
        // gettimeofday(&t_end, NULL);
        // std::cout << "clock_t Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        // double delta_t = (t_end.tv_sec - t_start.tv_sec) +
        //                  (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
        // std::cout << "multi-thread time : " << delta_t << "s" << std::endl;

        for (int i = err.size() - 1; i > 0; i--)
        {
            // std::cout << "err" << i << ":" << err[i] << std::endl;
            if (err[i] != 0 & (err[i] > 8 | err[i] < 1))
            {
                current_track_points.erase(current_track_points.begin() + i);
                status.erase(status.begin() + i);
                temp_track_points.erase(temp_track_points.begin() + i);
                err.erase(err.begin() + i);
            }
        }

        // std::cout << "current_track_points.size:" << current_track_points.size() << std::endl;
        // std::cout << "status.size:" << status.size() << std::endl;
        // std::cout << "err.size:" << err.size() << std::endl;

        std::swap(pre_track_points, current_track_points);
        cv::swap(prev_frame_gray, current_frame_gray);

        // if (!removeNoiseByStd(temp_track_points, current_track_points))
        // {
        //     return false;
        // }

        // removeNoiseByStd(temp_track_points, pre_track_points);

        if (pre_track_points.size() < 5)
        {
            track_failed = true;
            current_file_path = base_path + num2str<int>(frame_id) + ".bmp";
            current_frame = cv::imread(current_file_path);
            frame_id++;
            continue;
        }

        for (int i = 0; i < pre_track_points.size(); i++)
        {
            circle(current_frame, pre_track_points[i], 3, Scalar(0, 255, 0), -1, 8);
            circle(temp_frame, temp_track_points[i], 3, Scalar(0, 255, 0), -1, 8);
        }
        cv::imshow("temp", temp_frame);
        cv::imshow("track", current_frame);
        cv::waitKey(10);

        current_file_path = base_path + num2str<int>(frame_id) + ".bmp";
        current_frame = cv::imread(current_file_path);
        frame_id++;
    }

    return 0;
}
