#include "denoise_RANSAC.hpp"

#include "xcv/xprivate.hpp"
#include "xcv/calib3d_precomp.hpp"
#include "xcv/xrho.h"

class HomographyRefineCallback CV_FINAL : public cv::LMSolver::Callback
{
public:
    HomographyRefineCallback(cv::InputArray _src, cv::InputArray _dst)
    {
        src = _src.getMat();
        dst = _dst.getMat();
    }

    bool compute(cv::InputArray _param, cv::OutputArray _err, cv::OutputArray _Jac) const CV_OVERRIDE
    {
        int i, count = src.checkVector(2);
        cv::Mat param = _param.getMat();
        _err.create(count * 2, 1, CV_64F);
        cv::Mat err = _err.getMat(), J;
        if (_Jac.needed())
        {
            _Jac.create(count * 2, param.rows, CV_64F);
            J = _Jac.getMat();
            CV_Assert(J.isContinuous() && J.cols == 8);
        }

        const cv::Point2f *M = src.ptr<cv::Point2f>();
        const cv::Point2f *m = dst.ptr<cv::Point2f>();
        const double *h = param.ptr<double>();
        double *errptr = err.ptr<double>();
        double *Jptr = J.data ? J.ptr<double>() : 0;

        for (i = 0; i < count; i++)
        {
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6] * Mx + h[7] * My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1. / ww : 0;
            double xi = (h[0] * Mx + h[1] * My + h[2]) * ww;
            double yi = (h[3] * Mx + h[4] * My + h[5]) * ww;
            errptr[i * 2] = xi - m[i].x;
            errptr[i * 2 + 1] = yi - m[i].y;

            if (Jptr)
            {
                Jptr[0] = Mx * ww;
                Jptr[1] = My * ww;
                Jptr[2] = ww;
                Jptr[3] = Jptr[4] = Jptr[5] = 0.;
                Jptr[6] = -Mx * ww * xi;
                Jptr[7] = -My * ww * xi;
                Jptr[8] = Jptr[9] = Jptr[10] = 0.;
                Jptr[11] = Mx * ww;
                Jptr[12] = My * ww;
                Jptr[13] = ww;
                Jptr[14] = -Mx * ww * yi;
                Jptr[15] = -My * ww * yi;

                Jptr += 16;
            }
        }

        return true;
    }

    cv::Mat src, dst;
};

static bool createAndRunRHORegistrator(double confidence,
                                       int maxIters,
                                       double ransacReprojThreshold,
                                       int npoints,
                                       cv::InputArray _src,
                                       cv::InputArray _dst,
                                       cv::OutputArray _H,
                                       cv::OutputArray _tempMask)
{
    cv::Mat src = _src.getMat();
    cv::Mat dst = _dst.getMat();
    cv::Mat tempMask;
    bool result;
    double beta = 0.35; /* 0.35 is a value that often works. */

    /* Create temporary output matrix (RHO outputs a single-precision H only). */
    cv::Mat tmpH = cv::Mat(3, 3, CV_32FC1);

    /* Create output mask. */
    tempMask = cv::Mat(npoints, 1, CV_8U);

    /**
     * Make use of the RHO estimator API.
     *
     * This is where the math happens. A homography estimation context is
     * initialized, used, then finalized.
     */

    cv::Ptr<cv::RHO_HEST> p = cv::rhoInit();

    /**
     * Optional. Ideally, the context would survive across calls to
     * findHomography(), but no clean way appears to exit to do so. The price
     * to pay is marginally more computational work than strictly needed.
     */

    rhoEnsureCapacity(p, npoints, beta);

    /**
     * The critical call. All parameters are heavily documented in rho.h.
     *
     * Currently, NR (Non-Randomness criterion) and Final Refinement (with
     * internal, optimized Levenberg-Marquardt method) are enabled. However,
     * while refinement seems to correctly smooth jitter most of the time, when
     * refinement fails it tends to make the estimate visually very much worse.
     * It may be necessary to remove the refinement flags in a future commit if
     * this behaviour is too problematic.
     */

    result = !!cv::rhoHest(p,
                           (const float *)src.data,
                           (const float *)dst.data,
                           (char *)tempMask.data,
                           (unsigned)npoints,
                           (float)ransacReprojThreshold,
                           (unsigned)maxIters,
                           (unsigned)maxIters,
                           confidence,
                           4U,
                           beta,
                           RHO_FLAG_ENABLE_NR | RHO_FLAG_ENABLE_FINAL_REFINEMENT,
                           NULL,
                           (float *)tmpH.data);

    /* Convert float homography to double precision. */
    tmpH.convertTo(_H, CV_64FC1);

    /* Maps non-zero mask elements to 1, for the sake of the test case. */
    for (int k = 0; k < npoints; k++)
    {
        tempMask.data[k] = !!tempMask.data[k];
    }
    tempMask.copyTo(_tempMask);

    return result;
}

class HomographyEstimatorCallback CV_FINAL : public cv::PointSetRegistrator::Callback
{
public:
    bool checkSubset(cv::InputArray _ms1, cv::InputArray _ms2, int count) const CV_OVERRIDE
    {
        cv::Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        if (haveCollinearPoints(ms1, count) || haveCollinearPoints(ms2, count))
            return false;

        // We check whether the minimal set of points for the homography estimation
        // are geometrically consistent. We check if every 3 correspondences sets
        // fulfills the constraint.
        //
        // The usefullness of this constraint is explained in the paper:
        //
        // "Speeding-up homography estimation in mobile devices"
        // Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
        // Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
        if (count == 4)
        {
            static const int tt[][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
            const cv::Point2f *src = ms1.ptr<cv::Point2f>();
            const cv::Point2f *dst = ms2.ptr<cv::Point2f>();
            int negative = 0;

            for (int i = 0; i < 4; i++)
            {
                const int *t = tt[i];
                cv::Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
                cv::Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);

                negative += determinant(A) * determinant(B) < 0;
            }
            if (negative != 0 && negative != 4)
                return false;
        }

        return true;
    }

    /**
     * Normalization method:
     *  - $x$ and $y$ coordinates are normalized independently
     *  - first the coordinates are shifted so that the average coordinate is \f$(0,0)\f$
     *  - then the coordinates are scaled so that the average L1 norm is 1, i.e,
     *  the average L1 norm of the \f$x\f$ coordinates is 1 and the average
     *  L1 norm of the \f$y\f$ coordinates is also 1.
     *
     * @param _m1 source points containing (X,Y), depth is CV_32F with 1 column 2 channels or
     *            2 columns 1 channel
     * @param _m2 destination points containing (x,y), depth is CV_32F with 1 column 2 channels or
     *            2 columns 1 channel
     * @param _model, CV_64FC1, 3x3, normalized, i.e., the last element is 1
     */
    int runKernel(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model) const CV_OVERRIDE
    {
        cv::Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int i, count = m1.checkVector(2);
        const cv::Point2f *M = m1.ptr<cv::Point2f>();
        const cv::Point2f *m = m2.ptr<cv::Point2f>();

        double LtL[9][9], W[9][1], V[9][9];
        cv::Mat _LtL(9, 9, CV_64F, &LtL[0][0]);
        cv::Mat matW(9, 1, CV_64F, W);
        cv::Mat matV(9, 9, CV_64F, V);
        cv::Mat _H0(3, 3, CV_64F, V[8]);
        cv::Mat _Htemp(3, 3, CV_64F, V[7]);
        cv::Point2d cM(0, 0), cm(0, 0), sM(0, 0), sm(0, 0);

        for (i = 0; i < count; i++)
        {
            cm.x += m[i].x;
            cm.y += m[i].y;
            cM.x += M[i].x;
            cM.y += M[i].y;
        }

        cm.x /= count;
        cm.y /= count;
        cM.x /= count;
        cM.y /= count;

        for (i = 0; i < count; i++)
        {
            sm.x += fabs(m[i].x - cm.x);
            sm.y += fabs(m[i].y - cm.y);
            sM.x += fabs(M[i].x - cM.x);
            sM.y += fabs(M[i].y - cM.y);
        }

        if (fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
            fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON)
            return 0;
        sm.x = count / sm.x;
        sm.y = count / sm.y;
        sM.x = count / sM.x;
        sM.y = count / sM.y;

        double invHnorm[9] = {1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1};
        double Hnorm2[9] = {sM.x, 0, -cM.x * sM.x, 0, sM.y, -cM.y * sM.y, 0, 0, 1};
        cv::Mat _invHnorm(3, 3, CV_64FC1, invHnorm);
        cv::Mat _Hnorm2(3, 3, CV_64FC1, Hnorm2);

        _LtL.setTo(cv::Scalar::all(0));
        for (i = 0; i < count; i++)
        {
            double x = (m[i].x - cm.x) * sm.x, y = (m[i].y - cm.y) * sm.y;
            double X = (M[i].x - cM.x) * sM.x, Y = (M[i].y - cM.y) * sM.y;
            double Lx[] = {X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x};
            double Ly[] = {0, 0, 0, X, Y, 1, -y * X, -y * Y, -y};
            int j, k;
            for (j = 0; j < 9; j++)
                for (k = j; k < 9; k++)
                    LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
        }
        completeSymm(_LtL);

        eigen(_LtL, matW, matV);
        _Htemp = _invHnorm * _H0;
        _H0 = _Htemp * _Hnorm2;
        _H0.convertTo(_model, _H0.type(), 1. / _H0.at<double>(2, 2));

        return 1;
    }

    /**
     * Compute the reprojection error.
     * m2 = H*m1
     * @param _m1 depth CV_32F, 1-channel with 2 columns or 2-channel with 1 column
     * @param _m2 depth CV_32F, 1-channel with 2 columns or 2-channel with 1 column
     * @param _model CV_64FC1, 3x3
     * @param _err, output, CV_32FC1, square of the L2 norm
     */
    void computeError(cv::InputArray _m1, cv::InputArray _m2, cv::InputArray _model, cv::OutputArray _err) const CV_OVERRIDE
    {
        cv::Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        int i, count = m1.checkVector(2);
        const cv::Point2f *M = m1.ptr<cv::Point2f>();
        const cv::Point2f *m = m2.ptr<cv::Point2f>();
        const double *H = model.ptr<double>();
        float Hf[] = {(float)H[0], (float)H[1], (float)H[2], (float)H[3], (float)H[4], (float)H[5], (float)H[6], (float)H[7]};

        _err.create(count, 1, CV_32F);
        float *err = _err.getMat().ptr<float>();

        for (i = 0; i < count; i++)
        {
            float ww = 1.f / (Hf[6] * M[i].x + Hf[7] * M[i].y + 1.f);
            float dx = (Hf[0] * M[i].x + Hf[1] * M[i].y + Hf[2]) * ww - m[i].x;
            float dy = (Hf[3] * M[i].x + Hf[4] * M[i].y + Hf[5]) * ww - m[i].y;
            err[i] = dx * dx + dy * dy;
        }
    }
};

void yuchengconvertPointsFromHomogeneous(cv::InputArray _src, cv::OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    cv::Mat src = _src.getMat();
    if (!src.isContinuous())
        src = src.clone();
    int i, npoints = src.checkVector(3), depth = src.depth(), cn = 3;
    if (npoints < 0)
    {
        npoints = src.checkVector(4);
        CV_Assert(npoints >= 0);
        cn = 4;
    }
    CV_Assert(npoints >= 0 && (depth == CV_32S || depth == CV_32F || depth == CV_64F));

    int dtype = CV_MAKETYPE(depth <= CV_32F ? CV_32F : CV_64F, cn - 1);
    _dst.create(npoints, 1, dtype);
    cv::Mat dst = _dst.getMat();
    if (!dst.isContinuous())
    {
        _dst.release();
        _dst.create(npoints, 1, dtype);
        dst = _dst.getMat();
    }
    CV_Assert(dst.isContinuous());

    if (depth == CV_32S)
    {
        if (cn == 3)
        {
            const cv::Point3i *sptr = src.ptr<cv::Point3i>();
            cv::Point2f *dptr = dst.ptr<cv::Point2f>();
            for (i = 0; i < npoints; i++)
            {
                float scale = sptr[i].z != 0 ? 1.f / sptr[i].z : 1.f;
                dptr[i] = cv::Point2f(sptr[i].x * scale, sptr[i].y * scale);
            }
        }
        else
        {
            const cv::Vec4i *sptr = src.ptr<cv::Vec4i>();
            cv::Point3f *dptr = dst.ptr<cv::Point3f>();
            for (i = 0; i < npoints; i++)
            {
                float scale = sptr[i][3] != 0 ? 1.f / sptr[i][3] : 1.f;
                dptr[i] = cv::Point3f(sptr[i][0] * scale, sptr[i][1] * scale, sptr[i][2] * scale);
            }
        }
    }
    else if (depth == CV_32F)
    {
        if (cn == 3)
        {
            const cv::Point3f *sptr = src.ptr<cv::Point3f>();
            cv::Point2f *dptr = dst.ptr<cv::Point2f>();
            for (i = 0; i < npoints; i++)
            {
                float scale = sptr[i].z != 0.f ? 1.f / sptr[i].z : 1.f;
                dptr[i] = cv::Point2f(sptr[i].x * scale, sptr[i].y * scale);
            }
        }
        else
        {
            const cv::Vec4f *sptr = src.ptr<cv::Vec4f>();
            cv::Point3f *dptr = dst.ptr<cv::Point3f>();
            for (i = 0; i < npoints; i++)
            {
                float scale = sptr[i][3] != 0.f ? 1.f / sptr[i][3] : 1.f;
                dptr[i] = cv::Point3f(sptr[i][0] * scale, sptr[i][1] * scale, sptr[i][2] * scale);
            }
        }
    }
    else if (depth == CV_64F)
    {
        if (cn == 3)
        {
            const cv::Point3d *sptr = src.ptr<cv::Point3d>();
            cv::Point2d *dptr = dst.ptr<cv::Point2d>();
            for (i = 0; i < npoints; i++)
            {
                double scale = sptr[i].z != 0. ? 1. / sptr[i].z : 1.;
                dptr[i] = cv::Point2d(sptr[i].x * scale, sptr[i].y * scale);
            }
        }
        else
        {
            const cv::Vec4d *sptr = src.ptr<cv::Vec4d>();
            cv::Point3d *dptr = dst.ptr<cv::Point3d>();
            for (i = 0; i < npoints; i++)
            {
                double scale = sptr[i][3] != 0.f ? 1. / sptr[i][3] : 1.;
                dptr[i] = cv::Point3d(sptr[i][0] * scale, sptr[i][1] * scale, sptr[i][2] * scale);
            }
        }
    }
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");
}

cv::Mat yuchengfindHomography(cv::InputArray _points1, cv::InputArray _points2,
                       int method, double ransacReprojThreshold, cv::OutputArray _mask,
                       const int maxIters, const double confidence)
{
    CV_INSTRUMENT_REGION()

    const double defaultRANSACReprojThreshold = 3;
    bool result = false;

    cv::Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    cv::Mat src, dst, H, tempMask;
    int npoints = -1;

    for (int i = 1; i <= 2; i++)
    {
        cv::Mat &p = i == 1 ? points1 : points2;
        cv::Mat &m = i == 1 ? src : dst;
        npoints = p.checkVector(2, -1, false);
        if (npoints < 0)
        {
            npoints = p.checkVector(3, -1, false);
            if (npoints < 0)
                CV_Error(cv::Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
            if (npoints == 0)
                return cv::Mat();
            yuchengconvertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
    }

    CV_Assert(src.checkVector(2) == dst.checkVector(2));

    if (ransacReprojThreshold <= 0)
        ransacReprojThreshold = defaultRANSACReprojThreshold;

    cv::Ptr<cv::PointSetRegistrator::Callback> cb = cv::makePtr<HomographyEstimatorCallback>();

    if (method == 0 || npoints == 4)
    {
        tempMask = cv::Mat::ones(npoints, 1, CV_8U);
        result = cb->runKernel(src, dst, H) > 0;
    }
    else if (method == yuchengRANSAC)
        result = createRANSACPointSetRegistrator(cb, 4, ransacReprojThreshold, confidence, maxIters)->run(src, dst, H, tempMask);
    else if (method == yuchengLMEDS)
        result = createLMeDSPointSetRegistrator(cb, 4, confidence, maxIters)->run(src, dst, H, tempMask);
    else if (method == yuchengRHO)
        result = createAndRunRHORegistrator(confidence, maxIters, ransacReprojThreshold, npoints, src, dst, H, tempMask);
    else
        CV_Error(cv::Error::StsBadArg, "Unknown estimation method");

    if (result && npoints > 4 && method != yuchengRHO)
    {
        compressElems(src.ptr<cv::Point2f>(), tempMask.ptr<uchar>(), 1, npoints);
        npoints = compressElems(dst.ptr<cv::Point2f>(), tempMask.ptr<uchar>(), 1, npoints);
        if (npoints > 0)
        {
            cv::Mat src1 = src.rowRange(0, npoints);
            cv::Mat dst1 = dst.rowRange(0, npoints);
            src = src1;
            dst = dst1;
            if (method == yuchengRANSAC || method == yuchengLMEDS)
                cb->runKernel(src, dst, H);
            cv::Mat H8(8, 1, CV_64F, H.ptr<double>());
            cv::createLMSolver(cv::makePtr<HomographyRefineCallback>(src, dst), 10)->run(H8);
        }
    }

    if (result)
    {
        if (_mask.needed())
            tempMask.copyTo(_mask);
    }
    else
    {
        H.release();
        if (_mask.needed())
        {
            tempMask = cv::Mat::zeros(npoints >= 0 ? npoints : 0, 1, CV_8U);
            tempMask.copyTo(_mask);
        }
    }

    return H;
}

cv::Mat yuchengfindHomography(cv::InputArray _points1, cv::InputArray _points2,
                       cv::OutputArray _mask, int method, double ransacReprojThreshold)
{
    return yuchengfindHomography(_points1, _points2, method, ransacReprojThreshold, _mask);
}

/**
 * Compute the fundamental matrix using the 7-point algorithm.
 *
 * \f[
 *  (\mathrm{m2}_i,1)^T \mathrm{fmatrix} (\mathrm{m1}_i,1) = 0
 * \f]
 *
 * @param _m1 Contain points in the reference view. Depth CV_32F with 2-channel
 *            1 column or 1-channel 2 columns. It has 7 rows.
 * @param _m2 Contain points in the other view. Depth CV_32F with 2-channel
 *            1 column or 1-channel 2 columns. It has 7 rows.
 * @param _fmatrix Output fundamental matrix (or matrices) of type CV_64FC1.
 *                 The user is responsible for allocating the memory before calling
 *                 this function.
 * @return Number of fundamental matrices. Valid values are 1, 2 or 3.
 *  - 1, row 0 to row 2 in _fmatrix is a valid fundamental matrix
 *  - 2, row 3 to row 5 in _fmatrix is a valid fundamental matrix
 *  - 3, row 6 to row 8 in _fmatrix is a valid fundamental matrix
 *
 * Note that the computed fundamental matrix is normalized, i.e.,
 * the last element \f$F_{33}\f$ is 1.
 */
static int run7Point( const cv::Mat& _m1, const cv::Mat& _m2, cv::Mat& _fmatrix )
{
    double a[7*9], w[7], u[9*9], v[9*9], c[4], r[3] = {0};
    double* f1, *f2;
    double t0, t1, t2;
    cv::Mat A( 7, 9, CV_64F, a );
    cv::Mat U( 7, 9, CV_64F, u );
    cv::Mat Vt( 9, 9, CV_64F, v );
    cv::Mat W( 7, 1, CV_64F, w );
    cv::Mat coeffs( 1, 4, CV_64F, c );
    cv::Mat roots( 1, 3, CV_64F, r );
    const cv::Point2f* m1 = _m1.ptr<cv::Point2f>();
    const cv::Point2f* m2 = _m2.ptr<cv::Point2f>();
    double* fmatrix = _fmatrix.ptr<double>();
    int i, k, n;

    // form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
    for( i = 0; i < 7; i++ )
    {
        double x0 = m1[i].x, y0 = m1[i].y;
        double x1 = m2[i].x, y1 = m2[i].y;

        a[i*9+0] = x1*x0;
        a[i*9+1] = x1*y0;
        a[i*9+2] = x1;
        a[i*9+3] = y1*x0;
        a[i*9+4] = y1*y0;
        a[i*9+5] = y1;
        a[i*9+6] = x0;
        a[i*9+7] = y0;
        a[i*9+8] = 1;
    }

    // A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
    // the solution is linear subspace of dimensionality 2.
    // => use the last two singular vectors as a basis of the space
    // (according to SVD properties)
    SVDecomp( A, W, U, Vt, cv::SVD::MODIFY_A + cv::SVD::FULL_UV );
    f1 = v + 7*9;
    f2 = v + 8*9;

    // f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary fundamental matrix,
    // as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
    // so f ~ lambda*f1 + (1 - lambda)*f2.
    // use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
    // it will be a cubic equation.
    // find c - polynomial coefficients.
    for( i = 0; i < 9; i++ )
        f1[i] -= f2[i];

    t0 = f2[4]*f2[8] - f2[5]*f2[7];
    t1 = f2[3]*f2[8] - f2[5]*f2[6];
    t2 = f2[3]*f2[7] - f2[4]*f2[6];

    c[3] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2;

    c[2] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2 -
    f1[3]*(f2[1]*f2[8] - f2[2]*f2[7]) +
    f1[4]*(f2[0]*f2[8] - f2[2]*f2[6]) -
    f1[5]*(f2[0]*f2[7] - f2[1]*f2[6]) +
    f1[6]*(f2[1]*f2[5] - f2[2]*f2[4]) -
    f1[7]*(f2[0]*f2[5] - f2[2]*f2[3]) +
    f1[8]*(f2[0]*f2[4] - f2[1]*f2[3]);

    t0 = f1[4]*f1[8] - f1[5]*f1[7];
    t1 = f1[3]*f1[8] - f1[5]*f1[6];
    t2 = f1[3]*f1[7] - f1[4]*f1[6];

    c[1] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2 -
    f2[3]*(f1[1]*f1[8] - f1[2]*f1[7]) +
    f2[4]*(f1[0]*f1[8] - f1[2]*f1[6]) -
    f2[5]*(f1[0]*f1[7] - f1[1]*f1[6]) +
    f2[6]*(f1[1]*f1[5] - f1[2]*f1[4]) -
    f2[7]*(f1[0]*f1[5] - f1[2]*f1[3]) +
    f2[8]*(f1[0]*f1[4] - f1[1]*f1[3]);

    c[0] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2;

    // solve the cubic equation; there can be 1 to 3 roots ...
    n = solveCubic( coeffs, roots );

    if( n < 1 || n > 3 )
        return n;

    for( k = 0; k < n; k++, fmatrix += 9 )
    {
        // for each root form the fundamental matrix
        double lambda = r[k], mu = 1.;
        double s = f1[8]*r[k] + f2[8];

        // normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
        if( fabs(s) > DBL_EPSILON )
        {
            mu = 1./s;
            lambda *= mu;
            fmatrix[8] = 1.;
        }
        else
            fmatrix[8] = 0.;

        for( i = 0; i < 8; i++ )
            fmatrix[i] = f1[i]*lambda + f2[i]*mu;
    }

    return n;
}

/**
 * Compute the fundamental matrix using the 8-point algorithm.
 *
 * \f[
 *  (\mathrm{m2}_i,1)^T \mathrm{fmatrix} (\mathrm{m1}_i,1) = 0
 * \f]
 *
 * @param _m1 Contain points in the reference view. Depth CV_32F with 2-channel
 *            1 column or 1-channel 2 columns. It has 8 rows.
 * @param _m2 Contain points in the other view. Depth CV_32F with 2-channel
 *            1 column or 1-channel 2 columns. It has 8 rows.
 * @param _fmatrix Output fundamental matrix (or matrices) of type CV_64FC1.
 *                 The user is responsible for allocating the memory before calling
 *                 this function.
 * @return 1 on success, 0 on failure.
 *
 * Note that the computed fundamental matrix is normalized, i.e.,
 * the last element \f$F_{33}\f$ is 1.
 */
static int run8Point( const cv::Mat& _m1, const cv::Mat& _m2, cv::Mat& _fmatrix )
{
    cv::Point2d m1c(0,0), m2c(0,0);
    double t, scale1 = 0, scale2 = 0;

    const cv::Point2f* m1 = _m1.ptr<cv::Point2f>();
    const cv::Point2f* m2 = _m2.ptr<cv::Point2f>();
    CV_Assert( (_m1.cols == 1 || _m1.rows == 1) && _m1.size() == _m2.size());
    int i, count = _m1.checkVector(2);

    // compute centers and average distances for each of the two point sets
    for( i = 0; i < count; i++ )
    {
        m1c += cv::Point2d(m1[i]);
        m2c += cv::Point2d(m2[i]);
    }

    // calculate the normalizing transformations for each of the point sets:
    // after the transformation each set will have the mass center at the coordinate origin
    // and the average distance from the origin will be ~sqrt(2).
    t = 1./count;
    m1c *= t;
    m2c *= t;

    for( i = 0; i < count; i++ )
    {
        scale1 += norm(cv::Point2d(m1[i].x - m1c.x, m1[i].y - m1c.y));
        scale2 += norm(cv::Point2d(m2[i].x - m2c.x, m2[i].y - m2c.y));
    }

    scale1 *= t;
    scale2 *= t;

    if( scale1 < FLT_EPSILON || scale2 < FLT_EPSILON )
        return 0;

    scale1 = std::sqrt(2.)/scale1;
    scale2 = std::sqrt(2.)/scale2;

    cv::Matx<double, 9, 9> A;

    // form a linear system Ax=0: for each selected pair of points m1 & m2,
    // the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
    // to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0.
    for( i = 0; i < count; i++ )
    {
        double x1 = (m1[i].x - m1c.x)*scale1;
        double y1 = (m1[i].y - m1c.y)*scale1;
        double x2 = (m2[i].x - m2c.x)*scale2;
        double y2 = (m2[i].y - m2c.y)*scale2;
        cv::Vec<double, 9> r( x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1 );
        A += r*r.t();
    }

    cv::Vec<double, 9> W;
    cv::Matx<double, 9, 9> V;

    eigen(A, W, V);

    for( i = 0; i < 9; i++ )
    {
        if( fabs(W[i]) < DBL_EPSILON )
            break;
    }

    if( i < 8 )
        return 0;

    cv::Matx33d F0( V.val + 9*8 ); // take the last column of v as a solution of Af = 0

    // make F0 singular (of rank 2) by decomposing it with SVD,
    // zeroing the last diagonal element of W and then composing the matrices back.

    cv::Vec3d w;
    cv::Matx33d U;
    cv::Matx33d Vt;

    cv::SVD::compute( F0, w, U, Vt);
    w[2] = 0.;

    F0 = U * cv::Matx33d::diag(w) * Vt;

    // apply the transformation that is inverse
    // to what we used to normalize the point coordinates
    cv::Matx33d T1( scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 );
    cv::Matx33d T2( scale2, 0, -scale2*m2c.x, 0, scale2, -scale2*m2c.y, 0, 0, 1 );

    F0 = T2.t()*F0*T1;

    // make F(3,3) = 1
    if( fabs(F0(2,2)) > FLT_EPSILON )
        F0 *= 1./F0(2,2);

    cv::Mat(F0).copyTo(_fmatrix);

    return 1;
}

class FMEstimatorCallback CV_FINAL : public cv::PointSetRegistrator::Callback
{
public:
    bool checkSubset( cv::InputArray _ms1, cv::InputArray _ms2, int count ) const CV_OVERRIDE
    {
        cv::Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        return !haveCollinearPoints(ms1, count) && !haveCollinearPoints(ms2, count);
    }

    int runKernel( cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model ) const CV_OVERRIDE
    {
        double f[9*3];
        cv::Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int count = m1.checkVector(2);
        cv::Mat F(count == 7 ? 9 : 3, 3, CV_64F, f);
        int n = count == 7 ? run7Point(m1, m2, F) : run8Point(m1, m2, F);

        if( n == 0 )
            _model.release();
        else
            F.rowRange(0, n*3).copyTo(_model);

        return n;
    }

    void computeError( cv::InputArray _m1, cv::InputArray _m2, cv::InputArray _model, cv::OutputArray _err ) const CV_OVERRIDE
    {
        cv::Mat __m1 = _m1.getMat(), __m2 = _m2.getMat(), __model = _model.getMat();
        int i, count = __m1.checkVector(2);
        const cv::Point2f* m1 = __m1.ptr<cv::Point2f>();
        const cv::Point2f* m2 = __m2.ptr<cv::Point2f>();
        const double* F = __model.ptr<double>();
        _err.create(count, 1, CV_32F);
        float* err = _err.getMat().ptr<float>();

        for( i = 0; i < count; i++ )
        {
            double a, b, c, d1, d2, s1, s2;

            a = F[0]*m1[i].x + F[1]*m1[i].y + F[2];
            b = F[3]*m1[i].x + F[4]*m1[i].y + F[5];
            c = F[6]*m1[i].x + F[7]*m1[i].y + F[8];

            s2 = 1./(a*a + b*b);
            d2 = m2[i].x*a + m2[i].y*b + c;

            a = F[0]*m2[i].x + F[3]*m2[i].y + F[6];
            b = F[1]*m2[i].x + F[4]*m2[i].y + F[7];
            c = F[2]*m2[i].x + F[5]*m2[i].y + F[8];

            s1 = 1./(a*a + b*b);
            d1 = m1[i].x*a + m1[i].y*b + c;

            err[i] = (float)std::max(d1*d1*s1, d2*d2*s2);
        }
    }
};

cv::Mat yuchengfindFundamentalMat(cv::InputArray _points1, cv::InputArray _points2,
                                  int method, double ransacReprojThreshold, double confidence,
                                  cv::OutputArray _mask)
{
    CV_INSTRUMENT_REGION()

    cv::Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    cv::Mat m1, m2, F;
    int npoints = -1;

    for( int i = 1; i <= 2; i++ )
    {
        cv::Mat& p = i == 1 ? points1 : points2;
        cv::Mat& m = i == 1 ? m1 : m2;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            if( npoints < 0 )
                CV_Error(cv::Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
            if( npoints == 0 )
                return cv::Mat();
            convertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
    }

    CV_Assert( m1.checkVector(2) == m2.checkVector(2) );

    if( npoints < 7 )
        return cv::Mat();

    cv::Ptr<cv::PointSetRegistrator::Callback> cb = cv::makePtr<FMEstimatorCallback>();
    int result;

    if( npoints == 7 || method == yuchengFM_8POINT )
    {
        result = cb->runKernel(m1, m2, F);
        if( _mask.needed() )
        {
            _mask.create(npoints, 1, CV_8U, -1, true);
            cv::Mat mask = _mask.getMat();
            CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == npoints );
            mask.setTo(cv::Scalar::all(1));
        }
    }
    else
    {
        if( ransacReprojThreshold <= 0 )
            ransacReprojThreshold = 3;
        if( confidence < DBL_EPSILON || confidence > 1 - DBL_EPSILON )
            confidence = 0.99;

        if( (method & ~3) == yuchengFM_RANSAC && npoints >= 15 )
            result = createRANSACPointSetRegistrator(cb, 7, ransacReprojThreshold, confidence)->run(m1, m2, F, _mask);
        else
            result = createLMeDSPointSetRegistrator(cb, 7, confidence)->run(m1, m2, F, _mask);
    }

    if( result <= 0 )
        return cv::Mat();

    return F;
}

cv::Mat yuchengfindFundamentalMat(cv::InputArray _points1, cv::InputArray _points2,
                                  cv::OutputArray _mask, int method,
                                  double ransacReprojThreshold, double confidence)
{
    return yuchengfindFundamentalMat(_points1, _points2, method, ransacReprojThreshold, confidence, _mask);
}