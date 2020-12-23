#ifndef TRACK_KLT_HPP
#define TRACK_KLT_HPP

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

using namespace cv;

namespace cv
{
namespace detail
{

    typedef short deriv_type;

    struct LKTrackerInvoker : ParallelLoopBody
    {
        LKTrackerInvoker( const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                          const Point2f* _prevPts, Point2f* _nextPts,
                          uchar* _status, float* _err,
                          Size _winSize, TermCriteria _criteria,
                          int _level, int _maxLevel, int _flags, float _minEigThreshold );

        void operator()(const Range& range) const CV_OVERRIDE;

        const Mat* prevImg;
        const Mat* nextImg;
        const Mat* prevDeriv;
        const Point2f* prevPts;
        Point2f* nextPts;
        uchar* status;
        float* err;
        Size winSize;
        TermCriteria criteria;
        int level;
        int maxLevel;
        int flags;
        float minEigThreshold;
    };

}// namespace detail
}// namespace cv

void yuchengcalcOpticalFlowPyrLK(InputArray _prevImg, InputArray _nextImg,
                                 InputArray _prevPts, InputOutputArray _nextPts,
                                 OutputArray _status, OutputArray _err,
                                 Size winSize, int maxLevel,
                                 TermCriteria criteria,
                                 int flags, double minEigThreshold);

#endif // !TRACK_KLT_HPP