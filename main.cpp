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
#include "feature_SURF.hpp"
#include "feature_EAS.hpp"
#include "feature_GoodFeature.hpp"
#include "descriptor_BRIEF.hpp"
#include "denoise_RANSAC.hpp"
#include "track_KLT.hpp"

#include <time.h>
#include <sys/time.h>

#define USE_OPTICAL_FLOW false

template <class Type>
std::string num2str(Type num)
{
    std::stringstream ss;
    ss << num;
    return ss.str();
}

inline bool findKeyPointsHomography(std::vector<KeyPoint> &kpts1, std::vector<KeyPoint> &kpts2,
                                    std::vector<DMatch> &matches, std::vector<uchar> &match_mask, cv::Mat &homography_matrix)
{
    if (static_cast<int>(match_mask.size()) < 3)
    {
        return false;
    }
    std::vector<Point2f> pts1;
    std::vector<Point2f> pts2;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        pts1.push_back(kpts1[matches[i].queryIdx].pt);
        pts2.push_back(kpts2[matches[i].trainIdx].pt);
    }
    // homography_matrix = findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);
    homography_matrix = yuchengfindHomography(pts1, pts2, yuchengRANSAC, 0.5, match_mask);

    // homography_matrix = findFundamentalMat(pts1, pts2, match_mask, FM_RANSAC, 0.5);
    // homography_matrix = yuchengfindFundamentalMat(pts1, pts2, match_mask, yuchengFM_RANSAC, 0.05);

    // intrinsic[0] = 606.5764;
    // intrinsic[1] = 330.197;
    // intrinsic[2] = 607.7239;
    // intrinsic[3] = 232.6243;

    // cv::Mat intrinsic = cv::Mat(3, 3, CV_32F);
    // intrinsic = (cv::Mat_<int>(3, 3) << 606.5764, 0, 330.197, 0, 607.7239, 232.6243, 0, 0, 1);
    // homography_matrix = findEssentialMat(pts1, pts2, intrinsic, RANSAC, 0.998, 0.005, match_mask);

    std::cout
        << "homography_matrix:" << std::endl;
    std::cout << homography_matrix << std::endl;

    //单应矩阵条件数计算
    // cv::Mat eigenvalue, eigenvector;
    // // cv::eigen(Fundamental, eigenvalue, eigenvector);
    // // double maxEigenValue, minEigenValue;
    // // cv::minMaxLoc(eigenvalue, &minEigenValue, &maxEigenValue);

    // // std::cout << "maxEigenValue:" << maxEigenValue << std::endl;
    // // std::cout << "minEigenValue:" << minEigenValue << std::endl;
    // // std::cout << "cond:" << maxEigenValue / minEigenValue << std::endl;

    // if (homography_matrix.at<double>(0, 0) < -1e-5 || homography_matrix.at<double>(1, 1) < -1e-5)
    // {
    //     return false;
    // }
    return true;
}

inline void findGoodMatches(const cv::Mat &des1, const cv::Mat &des2, std::vector<DMatch> &goodMatches)
{
    //筛选条件1：所有KeyPoints中 当前KeyPoint距离/最小KeyPoint距离>4则抛弃
    BFMatcher desc_matcher(cv::NORM_L1, true);
    std::vector<std::vector<DMatch>> matches;

    desc_matcher.knnMatch(des1, des2, matches, 1);

    for (int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        if (!matches[i].size())
        {
            continue;
        }
        goodMatches.push_back(matches[i][0]);
    }
    std::sort(goodMatches.begin(), goodMatches.end());

    const double kDistanceCoef = 4.0;
    const int kMaxMatchingSize = 50;

    while (goodMatches.front().distance * kDistanceCoef < goodMatches.back().distance)
    {
        goodMatches.pop_back();
    }
    while (goodMatches.size() > kMaxMatchingSize)
    {
        goodMatches.pop_back();
    }

    //筛选条件2（距离比率测试）：当前KeyPoint次小距离/最小距离>0.8 则抛弃
    // cv::Ptr<cv::DescriptorMatcher>
    //     matcher = cv::DescriptorMatcher::create("FlannBased");
    // std::vector<std::vector<cv::DMatch>> matches;
    // std::vector<DMatch> goodMatches;
    // matcher->knnMatch(des1, des2, matches, 2);
    //筛选出较好的匹配点
    // for (int m = 0; m < matches.size(); m++)
    // {
    //     const float minRatio = 0.8f;
    //     const cv::DMatch &bestMatch = matches[m][0];
    //     const cv::DMatch &betterMatch = matches[m][1];
    //     float distanceRatio = bestMatch.distance /
    //                           betterMatch.distance;
    //     if (distanceRatio < minRatio)
    //     {
    //         goodMatches.push_back(bestMatch);
    //     }
    // }
}

class xFeature
{
public:
    std::vector<Point2f> temp_track_points, current_track_points;
    std::vector<cv::KeyPoint> keyPoint1, keyPoint2;
    std::vector<cv::Point2f> corners1, corners2;
    // bool track_failed = true;
    // cv::Mat prevGray;

    bool updateHarrisFeature(cv::Mat temp, cv::Mat current)
    {
        temp_track_points.clear();
        current_track_points.clear();
        keyPoint1.clear();
        keyPoint2.clear();

        calKeyPointbyGoodFeatures(temp, corners1, keyPoint1);
        calKeyPointbyGoodFeatures(current, corners2, keyPoint2);

        Mat des1, des2;
        Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(64);
        brief->compute(temp, keyPoint1, des1);
        brief->compute(current, keyPoint2, des2);

        // cv::Mat out_temp;
        // cv::drawKeypoints(temp, keyPoint1, out_temp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // cv::Mat out_current;
        // cv::drawKeypoints(current, keyPoint2, out_current, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // cv::namedWindow("out_temp", cv::WINDOW_NORMAL);
        // cv::namedWindow("out_current", cv::WINDOW_NORMAL);
        // cv::imshow("out_temp", out_temp);
        // cv::imshow("out_current", out_current);
        // cv::waitKey(0);
        // cv::destroyAllWindows();

        if (keyPoint1.size() < 6 || keyPoint2.size() < 6)
        {
            return false;
        }

        std::vector<DMatch> goodMatches;
        findGoodMatches(des1, des2, goodMatches);

        if (goodMatches.size() < 6)
        {
            return false;
        }

        //RANSAC去噪过程
        std::vector<uchar> match_mask(goodMatches.size(), 1);
        cv::Mat homography_matrix;
        bool find_successed = findKeyPointsHomography(keyPoint1, keyPoint2, goodMatches, match_mask, homography_matrix);

        if (find_successed == false)
        {
            return false;
        }

        //绘制投影框
        {
            std::vector<Point2f> img1_corners(4);
            img1_corners[0] = Point(0, 0);
            img1_corners[1] = Point(temp.cols, 0);
            img1_corners[2] = Point(temp.cols, temp.rows);
            img1_corners[3] = Point(0, temp.rows);
            std::vector<Point2f> img2_corners(4);

            perspectiveTransform(img1_corners, img2_corners, homography_matrix);
            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line(current, img2_corners[0],
                 img2_corners[1], Scalar(0, 255, 0), 4);
            line(current, img2_corners[1],
                 img2_corners[2], Scalar(0, 255, 0), 4);
            line(current, img2_corners[2],
                 img2_corners[3], Scalar(0, 255, 0), 4);
            line(current, img2_corners[3],
                 img2_corners[0], Scalar(0, 255, 0), 4);
        }

        //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
        std::vector<KeyPoint> RR_KP1, RR_KP2;
        std::vector<DMatch> RR_matches;
        int index = 0;
        // current_track_points.clear();
        // cv::Mat image = current.clone();
        for (size_t i = 0; i < goodMatches.size(); i++)
        {
            if (match_mask[i] != 0)
            {
                temp_track_points.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
                current_track_points.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
                RR_KP1.push_back(keyPoint1[goodMatches[i].queryIdx]);
                RR_KP2.push_back(keyPoint2[goodMatches[i].trainIdx]);
                goodMatches[i].queryIdx = index;
                goodMatches[i].trainIdx = index;
                RR_matches.push_back(goodMatches[i]);
                index++;
            }
        }

        std::cout << "inliers / matches:" << index << "/" << goodMatches.size() << std::endl;

        cv::Mat img_RR_matches;
        cv::drawMatches(temp, RR_KP1, current, RR_KP2, RR_matches, img_RR_matches);
        imshow("After RANSAC", img_RR_matches);
        //等待任意按键按下
        cv::waitKey(0);

        if (current_track_points.size() < 6)
        {
            return false;
        }

        return true;
    }

    bool updateEASFeature(cv::Mat temp, cv::Mat current)
    {
        temp_track_points.clear();
        current_track_points.clear();
        // if (prevGray.empty())
        //     current.copyTo(prevGray);
        keyPoint1.clear();
        keyPoint2.clear();

        calKeyPointbyEAS(temp, corners1, keyPoint1);
        cv::Mat out_temp;
        std::cout << "keyPoint1.size()" << keyPoint1.size() << std::endl;
        cv::drawKeypoints(temp, keyPoint1, out_temp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        calKeyPointbyEAS(current, corners2, keyPoint2);
        cv::Mat out_current;
        cv::drawKeypoints(current, keyPoint2, out_current, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // cv::imwrite("/home/yucheng/Code/1.Visual_Servoing/1.offline/1.track/yuchengCV2020/data/Lena_out.bmp", out_temp);

        cv::namedWindow("out_temp", cv::WINDOW_NORMAL);
        cv::namedWindow("out_current", cv::WINDOW_NORMAL);
        cv::imshow("out_temp", out_temp);
        cv::imshow("out_current", out_current);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool updateFastBriefFeature(cv::Mat temp, cv::Mat current)
    {
        temp_track_points.clear();
        current_track_points.clear();

        keyPoint1.clear();
        keyPoint2.clear();

        Mat des1, des2;

        Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
        detector->detect(temp, keyPoint1);
        detector->detect(current, keyPoint2);

        // Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(64);
        // brief->compute(temp, keyPoint1, des1);
        // brief->compute(current, keyPoint2, des2);

        calDescriptorsbyBRIEF(temp, keyPoint1, des1);
        calDescriptorsbyBRIEF(current, keyPoint2, des2);

        // Ptr<ORB> orb = ORB::create();
        // orb->detectAndCompute(temp, Mat(), keyPoint1, des1);
        // orb->detectAndCompute(current, Mat(), keyPoint2, des2);

        // cv::Mat out_temp;
        // cv::drawKeypoints(temp, keyPoint1, out_temp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // cv::Mat out_current;
        // cv::drawKeypoints(current, keyPoint2, out_current, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // cv::namedWindow("out_temp", cv::WINDOW_NORMAL);
        // cv::namedWindow("out_current", cv::WINDOW_NORMAL);
        // cv::imshow("out_temp", out_temp);
        // cv::imshow("out_current", out_current);
        // cv::waitKey(0);
        // cv::destroyAllWindows();

        if (keyPoint1.size() < 6 || keyPoint2.size() < 6)
        {
            return false;
        }

        std::vector<DMatch> goodMatches;
        findGoodMatches(des1, des2, goodMatches);

        if (goodMatches.size() < 6)
        {
            return false;
        }

        //RANSAC去噪过程
        std::vector<uchar> match_mask(goodMatches.size(), 1);
        cv::Mat homography_matrix;
        bool find_successed = findKeyPointsHomography(keyPoint1, keyPoint2, goodMatches, match_mask, homography_matrix);

        if (find_successed == false)
        {
            return false;
        }

        //绘制投影框
        {
            std::vector<Point2f> img1_corners(4);
            img1_corners[0] = Point(0, 0);
            img1_corners[1] = Point(temp.cols, 0);
            img1_corners[2] = Point(temp.cols, temp.rows);
            img1_corners[3] = Point(0, temp.rows);
            std::vector<Point2f> img2_corners(4);

            perspectiveTransform(img1_corners, img2_corners, homography_matrix);
            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line(current, img2_corners[0],
                 img2_corners[1], Scalar(0, 255, 0), 4);
            line(current, img2_corners[1],
                 img2_corners[2], Scalar(0, 255, 0), 4);
            line(current, img2_corners[2],
                 img2_corners[3], Scalar(0, 255, 0), 4);
            line(current, img2_corners[3],
                 img2_corners[0], Scalar(0, 255, 0), 4);
        }

        //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
        std::vector<KeyPoint> RR_KP1, RR_KP2;
        std::vector<DMatch> RR_matches;
        int index = 0;
        // current_track_points.clear();
        // cv::Mat image = current.clone();
        for (size_t i = 0; i < goodMatches.size(); i++)
        {
            if (match_mask[i] != 0)
            {
                temp_track_points.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
                current_track_points.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
                RR_KP1.push_back(keyPoint1[goodMatches[i].queryIdx]);
                RR_KP2.push_back(keyPoint2[goodMatches[i].trainIdx]);
                goodMatches[i].queryIdx = index;
                goodMatches[i].trainIdx = index;
                RR_matches.push_back(goodMatches[i]);
                index++;
            }
        }

        std::cout << "inliers / matches:" << index << "/" << goodMatches.size() << std::endl;

        cv::Mat img_RR_matches;
        cv::drawMatches(temp, RR_KP1, current, RR_KP2, RR_matches, img_RR_matches);
        imshow("After RANSAC", img_RR_matches);
        //等待任意按键按下
        cv::waitKey(0);

        if (current_track_points.size() < 6)
        {
            return false;
        }

        return true;
    }

    bool updateSURFFeature(cv::Mat temp, cv::Mat current)
    {
        temp_track_points.clear();
        current_track_points.clear();
        keyPoint1.clear();
        keyPoint2.clear();

        Mat des1, des2;

        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();

        // //提取特征点、计算描述子
        // detector->detectAndCompute(temp, Mat(), keyPoint1, des1);
        // detector->detectAndCompute(current, Mat(), keyPoint2, des2);
        calKeypointswithDescriptorsbySURF(temp, Mat(), keyPoint1, des1);
        calKeypointswithDescriptorsbySURF(current, Mat(), keyPoint2, des2);

        if (keyPoint1.size() < 6 || keyPoint2.size() < 6)
        {
            return false;
        }

        std::vector<DMatch> goodMatches;
        findGoodMatches(des1, des2, goodMatches);

        if (goodMatches.size() < 6)
        {
            return false;
        }

        //RANSAC去噪过程
        std::vector<uchar> match_mask(goodMatches.size(), 1);
        cv::Mat homography_matrix;
        bool find_successed = findKeyPointsHomography(keyPoint1, keyPoint2, goodMatches, match_mask, homography_matrix);

        if (find_successed == false)
        {
            return false;
        }

        //绘制投影框
        {
            std::vector<Point2f> img1_corners(4);
            img1_corners[0] = Point(0, 0);
            img1_corners[1] = Point(temp.cols, 0);
            img1_corners[2] = Point(temp.cols, temp.rows);
            img1_corners[3] = Point(0, temp.rows);
            std::vector<Point2f> img2_corners(4);

            perspectiveTransform(img1_corners, img2_corners, homography_matrix);
            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line(current, img2_corners[0],
                 img2_corners[1], Scalar(0, 255, 0), 4);
            line(current, img2_corners[1],
                 img2_corners[2], Scalar(0, 255, 0), 4);
            line(current, img2_corners[2],
                 img2_corners[3], Scalar(0, 255, 0), 4);
            line(current, img2_corners[3],
                 img2_corners[0], Scalar(0, 255, 0), 4);
        }

        //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
        std::vector<KeyPoint> RR_KP1, RR_KP2;
        std::vector<DMatch> RR_matches;
        int index = 0;
        // current_track_points.clear();
        // cv::Mat image = current.clone();
        for (size_t i = 0; i < goodMatches.size(); i++)
        {
            if (match_mask[i] != 0)
            {
                temp_track_points.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
                current_track_points.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
                RR_KP1.push_back(keyPoint1[goodMatches[i].queryIdx]);
                RR_KP2.push_back(keyPoint2[goodMatches[i].trainIdx]);
                goodMatches[i].queryIdx = index;
                goodMatches[i].trainIdx = index;
                RR_matches.push_back(goodMatches[i]);
                index++;
            }
        }

        std::cout << "inliers / matches:" << index << "/" << goodMatches.size() << std::endl;

        cv::Mat img_RR_matches;
        cv::drawMatches(temp, RR_KP1, current, RR_KP2, RR_matches, img_RR_matches);
        imshow("After RANSAC", img_RR_matches);
        //等待任意按键按下
        cv::waitKey(0);

        if (current_track_points.size() < 6)
        {
            return false;
        }

        return true;
    }

    bool updateSIFTFeature(cv::Mat temp, cv::Mat current)
    {
        temp_track_points.clear();
        current_track_points.clear();
        // if (prevGray.empty())
        //     current.copyTo(prevGray);
        keyPoint1.clear();
        keyPoint2.clear();

        // calKeyPointbySIFT(temp, corners1, keyPoint1);
        // calKeyPointbySIFT(current, corners2, keyPoint2);

        // if (keyPoint1.size() < 6 || keyPoint2.size() < 6)
        // {
        //     return false;
        // }

        Mat des1, des2;

        calKeypointswithDescriptorsbySIFT(temp, keyPoint1, des1);
        calKeypointswithDescriptorsbySIFT(current, keyPoint2, des2);

        // cv::Mat out_temp;
        // cv::drawKeypoints(temp, keyPoint1, out_temp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // cv::Mat out_current;
        // cv::drawKeypoints(current, keyPoint2, out_current, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // cv::namedWindow("out_temp", cv::WINDOW_NORMAL);
        // cv::namedWindow("out_current", cv::WINDOW_NORMAL);
        // cv::imshow("out_temp", out_temp);
        // cv::imshow("out_current", out_current);
        // cv::waitKey(0);
        // cv::destroyAllWindows();

        if (keyPoint1.size() < 6 || keyPoint2.size() < 6)
        {
            return false;
        }

        std::vector<DMatch> goodMatches;
        findGoodMatches(des1, des2, goodMatches);

        if (goodMatches.size() < 6)
        {
            return false;
        }

        //RANSAC去噪过程
        std::vector<uchar> match_mask(goodMatches.size(), 1);
        cv::Mat homography_matrix;
        bool find_successed = findKeyPointsHomography(keyPoint1, keyPoint2, goodMatches, match_mask, homography_matrix);

        if (find_successed == false)
        {
            return false;
        }

        //绘制投影框
        {
            std::vector<Point2f> img1_corners(4);
            img1_corners[0] = Point(0, 0);
            img1_corners[1] = Point(temp.cols, 0);
            img1_corners[2] = Point(temp.cols, temp.rows);
            img1_corners[3] = Point(0, temp.rows);
            std::vector<Point2f> img2_corners(4);

            perspectiveTransform(img1_corners, img2_corners, homography_matrix);
            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line(current, img2_corners[0],
                 img2_corners[1], Scalar(0, 255, 0), 4);
            line(current, img2_corners[1],
                 img2_corners[2], Scalar(0, 255, 0), 4);
            line(current, img2_corners[2],
                 img2_corners[3], Scalar(0, 255, 0), 4);
            line(current, img2_corners[3],
                 img2_corners[0], Scalar(0, 255, 0), 4);
        }

        //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
        std::vector<KeyPoint> RR_KP1, RR_KP2;
        std::vector<DMatch> RR_matches;
        int index = 0;
        // current_track_points.clear();
        // cv::Mat image = current.clone();
        for (size_t i = 0; i < goodMatches.size(); i++)
        {
            // if (match_mask[i] != '0')
            if (match_mask[i] == 1)
            {
                temp_track_points.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
                current_track_points.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
                RR_KP1.push_back(keyPoint1[goodMatches[i].queryIdx]);
                RR_KP2.push_back(keyPoint2[goodMatches[i].trainIdx]);
                goodMatches[i].queryIdx = index;
                goodMatches[i].trainIdx = index;
                RR_matches.push_back(goodMatches[i]);
                index++;
            }
        }

        std::cout << "inliers / matches:" << index << "/" << goodMatches.size() << std::endl;

        cv::Mat img_RR_matches;
        cv::drawMatches(temp, RR_KP1, current, RR_KP2, RR_matches, img_RR_matches);
        imshow("After RANSAC", img_RR_matches);
        //等待任意按键按下
        cv::waitKey(0);

        if (current_track_points.size() < 6)
        {
            return false;
        }

        return true;
    }
};

int main()
{
    cv::Mat current_frame;
    cv::Mat prev_frame_gray;
    cv::Mat current_frame_gray;
    cv::Mat temp_frame_gray;

    // cv::Mat temp_frame = cv::imread("../data/temp_color.bmp", -1);
    cv::Mat temp_frame_src = cv::imread("/home/yucheng/Code/1.Visual_Servoing/1.offline/1.track/yuchengCV2020/data/temp_color.bmp", -1);
    // cv::Mat temp_frame = cv::imread("/home/yucheng/Code/1.Visual_Servoing/1.offline/1.track/yuchengCV2020/data/lena.bmp", -1);
    // cv::Mat temp_frame = cv::imread("/home/yucheng/Code/1.Visual_Servoing/1.offline/1.track/yuchengCV2020/data/57.bmp", -1);
    cv::cvtColor(temp_frame_src, temp_frame_gray, cv::COLOR_BGR2GRAY);

    // std::string base_path = "../data/12_11/60Hz/";
    // std::string base_path = "/home/yucheng/Code/1.Visual_Servoing/1.offline/1.track/yuchengCV2020/data/12_11/60Hz/";
    std::string base_path = "/home/yucheng/Code/1.Visual_Servoing/2.half_online/visual_servoing/data/image_seq/";

    std::string current_file_path = base_path + "10.bmp";
    current_frame = cv::imread(current_file_path);
    int frame_id = 11;
    bool track_failed = true;

    std::vector<cv::Point2f> temp_track_points, pre_track_points, current_track_points;

    xFeature feature_extractor;
    int loss_count = 0;

    while (!current_frame.empty())
    {
        //使用灰度图像进行角点检测
        cv::cvtColor(current_frame, current_frame_gray, cv::COLOR_BGR2GRAY);

        if (track_failed)
        {
            bool update_secussed = false;

            timeval t_start, t_end;
            clock_t startTime, endTime;
            gettimeofday(&t_start, NULL);
            startTime = clock();
            // update_secussed = feature_extractor.updateFastBriefFeature(temp_frame_gray, current_frame_gray);
            // update_secussed = feature_extractor.updateHarrisFeature(temp_frame_gray, current_frame_gray);
            update_secussed = feature_extractor.updateSIFTFeature(temp_frame_gray, current_frame_gray);
            // update_secussed = feature_extractor.updateSURFFeature(temp_frame_gray, current_frame_gray);
            std::cout << "frame_id:" << frame_id << std::endl;
            // update_secussed = feature_extractor.updateEASFeature(temp_frame_gray, current_frame_gray);
            endTime = clock();
            gettimeofday(&t_end, NULL);
            std::cout << "clock_t Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
            double delta_t = (t_end.tv_sec - t_start.tv_sec) +
                             (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
            std::cout << "multi-thread time : " << delta_t << "s" << std::endl;

            if (!update_secussed)
            {
                loss_count++;
                // std::string failed_path = "/home/yucheng/Code/1.Visual_Servoing/2.half_online/visual_servoing/data/failed_image/" + num2str<int>(frame_id - 1) + ".bmp";
                // cv::imwrite(failed_path, current_frame);
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
            // track_failed = false;
        }

#if USE_OPTICAL_FLOW
        track_failed = false;
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
#endif

        // cv::Mat temp_frame = temp_frame_src.clone();
        // for (int i = 0; i < pre_track_points.size(); i++)
        // {
        //     circle(current_frame, pre_track_points[i], 3, Scalar(0, 255, 0), -1, 8);
        //     circle(temp_frame, temp_track_points[i], 3, Scalar(0, 255, 0), -1, 8);
        // }
        // cv::imshow("temp", temp_frame);
        // cv::imshow("track", current_frame);
        // cv::waitKey(30);

        current_file_path = base_path + num2str<int>(frame_id) + ".bmp";
        current_frame = cv::imread(current_file_path);
        frame_id++;
    }
    std::cout << "loss images: = " << loss_count << std::endl;

    return 0;
}
