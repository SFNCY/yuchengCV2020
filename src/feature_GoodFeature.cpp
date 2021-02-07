#include "feature_GoodFeature.hpp"

#include "xcv/xprecomp.hpp"
#include "xcv/xopencl_kernels_imgproc.hpp"
#include "xcv/xprivate.hpp"

#include "opencv2/core/base.hpp"

#include <vector>
#include <iostream>

enum
{
    MINEIGENVAL = 0,
    HARRIS = 1,
    EIGENVALSVECS = 2
};

using namespace cv;

#ifdef CV_CXX11
struct greaterThanPtr
#else
struct greaterThanPtr : public std::binary_function<const float *, const float *, bool>
#endif
{
    bool operator()(const float *a, const float *b) const
    // Ensure a fully deterministic result of the sort
    {
        return (*a > *b) ? true : (*a < *b) ? false : (a > b);
    }
};

#if defined(HAVE_IPP)
namespace cv
{
    static bool ipp_cornerMinEigenVal(InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType)
    {
#if IPP_VERSION_X100 >= 800
        CV_INSTRUMENT_REGION_IPP()

        Mat src = _src.getMat();
        _dst.create(src.size(), CV_32FC1);
        Mat dst = _dst.getMat();

        {
            typedef IppStatus(CV_STDCALL * ippiMinEigenValGetBufferSize)(IppiSize, int, int, int *);
            typedef IppStatus(CV_STDCALL * ippiMinEigenVal)(const void *, int, Ipp32f *, int, IppiSize, IppiKernelType, int, int, Ipp8u *);
            IppiKernelType kerType;
            int kerSize = ksize;
            if (ksize < 0)
            {
                kerType = ippKernelScharr;
                kerSize = 3;
            }
            else
            {
                kerType = ippKernelSobel;
            }
            bool isolated = (borderType & BORDER_ISOLATED) != 0;
            int borderTypeNI = borderType & ~BORDER_ISOLATED;
            if ((borderTypeNI == BORDER_REPLICATE && (!src.isSubmatrix() || isolated)) &&
                (kerSize == 3 || kerSize == 5) && (blockSize == 3 || blockSize == 5))
            {
                ippiMinEigenValGetBufferSize getBufferSizeFunc = 0;
                ippiMinEigenVal ippiMinEigenVal_C1R = 0;
                float norm_coef = 0.f;

                if (src.type() == CV_8UC1)
                {
                    getBufferSizeFunc = (ippiMinEigenValGetBufferSize)ippiMinEigenValGetBufferSize_8u32f_C1R;
                    ippiMinEigenVal_C1R = (ippiMinEigenVal)ippiMinEigenVal_8u32f_C1R;
                    norm_coef = 1.f / 255.f;
                }
                else if (src.type() == CV_32FC1)
                {
                    getBufferSizeFunc = (ippiMinEigenValGetBufferSize)ippiMinEigenValGetBufferSize_32f_C1R;
                    ippiMinEigenVal_C1R = (ippiMinEigenVal)ippiMinEigenVal_32f_C1R;
                    norm_coef = 255.f;
                }
                norm_coef = kerType == ippKernelSobel ? norm_coef : norm_coef / 2.45f;

                if (getBufferSizeFunc && ippiMinEigenVal_C1R)
                {
                    int bufferSize;
                    IppiSize srcRoi = {src.cols, src.rows};
                    IppStatus ok = getBufferSizeFunc(srcRoi, kerSize, blockSize, &bufferSize);
                    if (ok >= 0)
                    {
                        AutoBuffer<uchar> buffer(bufferSize);
                        ok = CV_INSTRUMENT_FUN_IPP(ippiMinEigenVal_C1R, src.ptr(), (int)src.step, dst.ptr<Ipp32f>(), (int)dst.step, srcRoi, kerType, kerSize, blockSize, buffer);
                        CV_SUPPRESS_DEPRECATED_START
                        if (ok >= 0)
                            ok = CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C1IR, norm_coef, dst.ptr<Ipp32f>(), (int)dst.step, srcRoi);
                        CV_SUPPRESS_DEPRECATED_END
                        if (ok >= 0)
                        {
                            CV_IMPL_ADD(CV_IMPL_IPP);
                            return true;
                        }
                    }
                }
            }
        }
#else
        CV_UNUSED(_src);
        CV_UNUSED(_dst);
        CV_UNUSED(blockSize);
        CV_UNUSED(borderType);
#endif
        return false;
    }
} // namespace cv
#endif

static void calcMinEigenVal(const Mat &_cov, Mat &_dst)
{
    int i, j;
    Size size = _cov.size();
#if CV_TRY_AVX
    bool haveAvx = CV_CPU_HAS_SUPPORT_AVX;
#endif
#if CV_SIMD128
    bool haveSimd = hasSIMD128();
#endif

    if (_cov.isContinuous() && _dst.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for (i = 0; i < size.height; i++)
    {
        const float *cov = _cov.ptr<float>(i);
        float *dst = _dst.ptr<float>(i);
#if CV_TRY_AVX
        if (haveAvx)
            j = calcMinEigenValLine_AVX(cov, dst, size.width);
        else
#endif // CV_TRY_AVX
            j = 0;

#if CV_SIMD128
        if (haveSimd)
        {
            v_float32x4 half = v_setall_f32(0.5f);
            for (; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes)
            {
                v_float32x4 v_a, v_b, v_c, v_t;
                v_load_deinterleave(cov + j * 3, v_a, v_b, v_c);
                v_a *= half;
                v_c *= half;
                v_t = v_a - v_c;
                v_t = v_muladd(v_b, v_b, (v_t * v_t));
                v_store(dst + j, (v_a + v_c) - v_sqrt(v_t));
            }
        }
#endif // CV_SIMD128

        for (; j < size.width; j++)
        {
            float a = cov[j * 3] * 0.5f;
            float b = cov[j * 3 + 1];
            float c = cov[j * 3 + 2] * 0.5f;
            dst[j] = (float)((a + c) - std::sqrt((a - c) * (a - c) + b * b));
        }
    }
}

static void calcHarris(const Mat &_cov, Mat &_dst, double k)
{
    int i, j;
    Size size = _cov.size();
#if CV_TRY_AVX
    bool haveAvx = CV_CPU_HAS_SUPPORT_AVX;
#endif
#if CV_SIMD128
    bool haveSimd = hasSIMD128();
#endif

    if (_cov.isContinuous() && _dst.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for (i = 0; i < size.height; i++)
    {
        const float *cov = _cov.ptr<float>(i);
        float *dst = _dst.ptr<float>(i);

#if CV_TRY_AVX
        if (haveAvx)
            j = calcHarrisLine_AVX(cov, dst, k, size.width);
        else
#endif // CV_TRY_AVX
            j = 0;

#if CV_SIMD128
        if (haveSimd)
        {
            v_float32x4 v_k = v_setall_f32((float)k);

            for (; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes)
            {
                v_float32x4 v_a, v_b, v_c;
                v_load_deinterleave(cov + j * 3, v_a, v_b, v_c);

                v_float32x4 v_ac_bb = v_a * v_c - v_b * v_b;
                v_float32x4 v_ac = v_a + v_c;
                v_float32x4 v_dst = v_ac_bb - v_k * v_ac * v_ac;
                v_store(dst + j, v_dst);
            }
        }
#endif // CV_SIMD128

        for (; j < size.width; j++)
        {
            float a = cov[j * 3];
            float b = cov[j * 3 + 1];
            float c = cov[j * 3 + 2];
            dst[j] = (float)(a * c - b * b - k * (a + c) * (a + c));
        }
    }
}

static void eigen2x2(const float *cov, float *dst, int n)
{
    for (int j = 0; j < n; j++)
    {
        double a = cov[j * 3];
        double b = cov[j * 3 + 1];
        double c = cov[j * 3 + 2];

        double u = (a + c) * 0.5;
        double v = std::sqrt((a - c) * (a - c) * 0.25 + b * b);
        double l1 = u + v;
        double l2 = u - v;

        double x = b;
        double y = l1 - a;
        double e = fabs(x);

        if (e + fabs(y) < 1e-4)
        {
            y = b;
            x = l1 - c;
            e = fabs(x);
            if (e + fabs(y) < 1e-4)
            {
                e = 1. / (e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }

        double d = 1. / std::sqrt(x * x + y * y + DBL_EPSILON);
        dst[6 * j] = (float)l1;
        dst[6 * j + 2] = (float)(x * d);
        dst[6 * j + 3] = (float)(y * d);

        x = b;
        y = l2 - a;
        e = fabs(x);

        if (e + fabs(y) < 1e-4)
        {
            y = b;
            x = l2 - c;
            e = fabs(x);
            if (e + fabs(y) < 1e-4)
            {
                e = 1. / (e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }

        d = 1. / std::sqrt(x * x + y * y + DBL_EPSILON);
        dst[6 * j + 1] = (float)l2;
        dst[6 * j + 4] = (float)(x * d);
        dst[6 * j + 5] = (float)(y * d);
    }
}

static void calcEigenValsVecs(const Mat &_cov, Mat &_dst)
{
    Size size = _cov.size();
    if (_cov.isContinuous() && _dst.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for (int i = 0; i < size.height; i++)
    {
        const float *cov = _cov.ptr<float>(i);
        float *dst = _dst.ptr<float>(i);

        eigen2x2(cov, dst, size.width);
    }
}

static void
cornerEigenValsVecs(const Mat &src, Mat &eigenv, int block_size,
                    int aperture_size, int op_type, double k = 0.,
                    int borderType = BORDER_DEFAULT)
{

    std::cout << "cornerEigenValsVecs" << std::endl;
#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra() && tegra::cornerEigenValsVecs(src, eigenv, block_size, aperture_size, op_type, k, borderType))
        return;
#endif
#if CV_TRY_AVX
    bool haveAvx = CV_CPU_HAS_SUPPORT_AVX;
#endif
#if CV_SIMD128
    bool haveSimd = hasSIMD128();
#endif

    int depth = src.depth();
    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if (aperture_size < 0)
        scale *= 2.0;
    if (depth == CV_8U)
        scale *= 255.0;
    scale = 1.0 / scale;

    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);

    Mat Dx, Dy;
    if (aperture_size > 0)
    {
        Sobel(src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType);
        Sobel(src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType);
    }
    else
    {
        Scharr(src, Dx, CV_32F, 1, 0, scale, 0, borderType);
        Scharr(src, Dy, CV_32F, 0, 1, scale, 0, borderType);
    }

    Size size = src.size();
    Mat cov(size, CV_32FC3);
    int i, j;

    for (i = 0; i < size.height; i++)
    {
        float *cov_data = cov.ptr<float>(i);
        const float *dxdata = Dx.ptr<float>(i);
        const float *dydata = Dy.ptr<float>(i);

#if CV_TRY_AVX
        if (haveAvx)
            j = cornerEigenValsVecsLine_AVX(dxdata, dydata, cov_data, size.width);
        else
#endif // CV_TRY_AVX
            j = 0;

#if CV_SIMD128
        if (haveSimd)
        {
            for (; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes)
            {
                v_float32x4 v_dx = v_load(dxdata + j);
                v_float32x4 v_dy = v_load(dydata + j);

                v_float32x4 v_dst0, v_dst1, v_dst2;
                v_dst0 = v_dx * v_dx;
                v_dst1 = v_dx * v_dy;
                v_dst2 = v_dy * v_dy;

                v_store_interleave(cov_data + j * 3, v_dst0, v_dst1, v_dst2);
            }
        }
#endif // CV_SIMD128

        for (; j < size.width; j++)
        {
            float dx = dxdata[j];
            float dy = dydata[j];

            cov_data[j * 3] = dx * dx;
            cov_data[j * 3 + 1] = dx * dy;
            cov_data[j * 3 + 2] = dy * dy;
        }
    }

    boxFilter(cov, cov, cov.depth(), Size(block_size, block_size),
              Point(-1, -1), false, borderType);

    if (op_type == MINEIGENVAL)
        calcMinEigenVal(cov, eigenv);
    else if (op_type == HARRIS)
        calcHarris(cov, eigenv, k);
    else if (op_type == EIGENVALSVECS)
        calcEigenValsVecs(cov, eigenv);
}

#ifdef HAVE_OPENCL

static bool extractCovData(InputArray _src, UMat &Dx, UMat &Dy, int depth,
                           float scale, int aperture_size, int borderType)
{
    UMat src = _src.getUMat();

    Size wholeSize;
    Point ofs;
    src.locateROI(wholeSize, ofs);

    const int sobel_lsz = 16;
    if ((aperture_size == 3 || aperture_size == 5 || aperture_size == 7 || aperture_size == -1) &&
        wholeSize.height > sobel_lsz + (aperture_size >> 1) &&
        wholeSize.width > sobel_lsz + (aperture_size >> 1))
    {
        CV_Assert(depth == CV_8U || depth == CV_32F);

        Dx.create(src.size(), CV_32FC1);
        Dy.create(src.size(), CV_32FC1);

        size_t localsize[2] = {(size_t)sobel_lsz, (size_t)sobel_lsz};
        size_t globalsize[2] = {localsize[0] * (1 + (src.cols - 1) / localsize[0]),
                                localsize[1] * (1 + (src.rows - 1) / localsize[1])};

        int src_offset_x = (int)((src.offset % src.step) / src.elemSize());
        int src_offset_y = (int)(src.offset / src.step);

        const char *const borderTypes[] = {"BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
                                           "BORDER_WRAP", "BORDER_REFLECT101"};

        ocl::Kernel k(format("sobel%d", aperture_size).c_str(), ocl::imgproc::covardata_oclsrc,
                      cv::format("-D BLK_X=%d -D BLK_Y=%d -D %s -D SRCTYPE=%s%s",
                                 (int)localsize[0], (int)localsize[1], borderTypes[borderType], ocl::typeToStr(depth),
                                 aperture_size < 0 ? " -D SCHARR" : ""));
        if (k.empty())
            return false;

        k.args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, src_offset_x, src_offset_y,
               ocl::KernelArg::WriteOnlyNoSize(Dx), ocl::KernelArg::WriteOnly(Dy),
               wholeSize.height, wholeSize.width, scale);

        return k.run(2, globalsize, localsize, false);
    }
    else
    {
        if (aperture_size > 0)
        {
            Sobel(_src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType);
            Sobel(_src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType);
        }
        else
        {
            Scharr(_src, Dx, CV_32F, 1, 0, scale, 0, borderType);
            Scharr(_src, Dy, CV_32F, 0, 1, scale, 0, borderType);
        }
    }

    return true;
}

static bool ocl_cornerMinEigenValVecs(InputArray _src, OutputArray _dst, int block_size,
                                      int aperture_size, double k, int borderType, int op_type)
{
    CV_Assert(op_type == HARRIS || op_type == MINEIGENVAL);

    if (!(borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE ||
          borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101))
        return false;

    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    if (!(type == CV_8UC1 || type == CV_32FC1))
        return false;

    const char *const borderTypes[] = {"BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
                                       "BORDER_WRAP", "BORDER_REFLECT101"};
    const char *const cornerType[] = {"CORNER_MINEIGENVAL", "CORNER_HARRIS", 0};

    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if (aperture_size < 0)
        scale *= 2.0;
    if (depth == CV_8U)
        scale *= 255.0;
    scale = 1.0 / scale;

    UMat Dx, Dy;
    if (!extractCovData(_src, Dx, Dy, depth, (float)scale, aperture_size, borderType))
        return false;

    ocl::Kernel cornelKernel("corner", ocl::imgproc::corner_oclsrc,
                             format("-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s -D %s",
                                    block_size / 2, block_size / 2, block_size, block_size,
                                    borderTypes[borderType], cornerType[op_type]));
    if (cornelKernel.empty())
        return false;

    _dst.createSameSize(_src, CV_32FC1);
    UMat dst = _dst.getUMat();

    cornelKernel.args(ocl::KernelArg::ReadOnly(Dx), ocl::KernelArg::ReadOnly(Dy),
                      ocl::KernelArg::WriteOnly(dst), (float)k);

    size_t blockSizeX = 256, blockSizeY = 1;
    size_t gSize = blockSizeX - block_size / 2 * 2;
    size_t globalSizeX = (Dx.cols) % gSize == 0 ? Dx.cols / gSize * blockSizeX : (Dx.cols / gSize + 1) * blockSizeX;
    size_t rows_per_thread = 2;
    size_t globalSizeY = ((Dx.rows + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ? ((Dx.rows + rows_per_thread - 1) / rows_per_thread) : (((Dx.rows + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;

    size_t globalsize[2] = {globalSizeX, globalSizeY}, localsize[2] = {blockSizeX, blockSizeY};
    return cornelKernel.run(2, globalsize, localsize, false);
}

static bool ocl_preCornerDetect(InputArray _src, OutputArray _dst, int ksize, int borderType, int depth)
{
    UMat Dx, Dy, D2x, D2y, Dxy;

    if (!extractCovData(_src, Dx, Dy, depth, 1, ksize, borderType))
        return false;

    Sobel(_src, D2x, CV_32F, 2, 0, ksize, 1, 0, borderType);
    Sobel(_src, D2y, CV_32F, 0, 2, ksize, 1, 0, borderType);
    Sobel(_src, Dxy, CV_32F, 1, 1, ksize, 1, 0, borderType);

    _dst.create(_src.size(), CV_32FC1);
    UMat dst = _dst.getUMat();

    double factor = 1 << (ksize - 1);
    if (depth == CV_8U)
        factor *= 255;
    factor = 1. / (factor * factor * factor);

    ocl::Kernel k("preCornerDetect", ocl::imgproc::precornerdetect_oclsrc);
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(Dx), ocl::KernelArg::ReadOnlyNoSize(Dy),
           ocl::KernelArg::ReadOnlyNoSize(D2x), ocl::KernelArg::ReadOnlyNoSize(D2y),
           ocl::KernelArg::ReadOnlyNoSize(Dxy), ocl::KernelArg::WriteOnly(dst), (float)factor);

    size_t globalsize[2] = {(size_t)dst.cols, (size_t)dst.rows};
    return k.run(2, globalsize, NULL, false);
}

#endif

void yuchengcornerMinEigenVal(InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType)
{
    CV_INSTRUMENT_REGION()

//     CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
//                ocl_cornerMinEigenValVecs(_src, _dst, blockSize, ksize, 0.0, borderType, MINEIGENVAL))

// #ifdef HAVE_IPP
//     int kerSize = (ksize < 0) ? 3 : ksize;
//     bool isolated = (borderType & BORDER_ISOLATED) != 0;
//     int borderTypeNI = borderType & ~BORDER_ISOLATED;
// #endif
//     CV_IPP_RUN(((borderTypeNI == BORDER_REPLICATE && (!_src.isSubmatrix() || isolated)) &&
//                 (kerSize == 3 || kerSize == 5) && (blockSize == 3 || blockSize == 5)) &&
//                    IPP_VERSION_X100 >= 800,
//                ipp_cornerMinEigenVal(_src, _dst, blockSize, ksize, borderType));

    Mat src = _src.getMat();
    _dst.create(src.size(), CV_32FC1);
    Mat dst = _dst.getMat();

    cornerEigenValsVecs(src, dst, blockSize, ksize, MINEIGENVAL, 0, borderType);
}

void getGoodFeatures(InputArray _image, OutputArray _corners,
                     int maxCorners, double qualityLevel, double minDistance,
                     InputArray _mask, int blockSize, int gradientSize,
                     bool useHarrisDetector, double harrisK)
{
    CV_INSTRUMENT_REGION()

    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(_mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)));

    // CV_OCL_RUN(_image.dims() <= 2 && _image.isUMat(),
    //            ocl_goodFeaturesToTrack(_image, _corners, maxCorners, qualityLevel, minDistance,
    //                                    _mask, blockSize, gradientSize, useHarrisDetector, harrisK))

    Mat image = _image.getMat(), eig, tmp;
    if (image.empty())
    {
        _corners.release();
        return;
    }

    // Disabled due to bad accuracy
    // CV_OVX_RUN(false && useHarrisDetector && _mask.empty() &&
    //                !ovx::skipSmallImages<VX_KERNEL_HARRIS_CORNERS>(image.cols, image.rows),
    //            openvx_harris(image, _corners, maxCorners, qualityLevel, minDistance, blockSize, gradientSize, harrisK))

    if (useHarrisDetector)
        cornerHarris(image, eig, blockSize, gradientSize, harrisK);
    else
        yuchengcornerMinEigenVal(image, eig, blockSize, gradientSize);

    double maxVal = 0;
    minMaxLoc(eig, 0, &maxVal, 0, 0, _mask);
    threshold(eig, eig, maxVal * qualityLevel, 0, THRESH_TOZERO);
    dilate(eig, tmp, Mat());

    Size imgsize = image.size();
    std::vector<const float *> tmpCorners;

    // collect list of pointers to features - put them into temporary image
    Mat mask = _mask.getMat();
    for (int y = 1; y < imgsize.height - 1; y++)
    {
        const float *eig_data = (const float *)eig.ptr(y);
        const float *tmp_data = (const float *)tmp.ptr(y);
        const uchar *mask_data = mask.data ? mask.ptr(y) : 0;

        for (int x = 1; x < imgsize.width - 1; x++)
        {
            float val = eig_data[x];
            if (val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]))
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0)
    {
        _corners.release();
        return;
    }

    std::sort(tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f>> grid(grid_width * grid_height);

        minDistance *= minDistance;

        for (i = 0; i < total; i++)
        {
            int ofs = (int)((const uchar *)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y * eig.step) / sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for (int yy = y1; yy <= y2; yy++)
            {
                for (int xx = x1; xx <= x2; xx++)
                {
                    std::vector<Point2f> &m = grid[yy * grid_width + xx];

                    if (m.size())
                    {
                        for (j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if (dx * dx + dy * dy < minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

        break_out:

            if (good)
            {
                grid[y_cell * grid_width + x_cell].push_back(Point2f((float)x, (float)y));

                corners.push_back(Point2f((float)x, (float)y));
                ++ncorners;

                if (maxCorners > 0 && (int)ncorners == maxCorners)
                    break;
            }
        }
    }
    else
    {
        for (i = 0; i < total; i++)
        {
            int ofs = (int)((const uchar *)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y * eig.step) / sizeof(float));

            corners.push_back(Point2f((float)x, (float)y));
            ++ncorners;
            if (maxCorners > 0 && (int)ncorners == maxCorners)
                break;
        }
    }

    Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}

void getGoodFeatures(InputArray _image, OutputArray _corners,
                     int maxCorners, double qualityLevel, double minDistance,
                     InputArray _mask, int blockSize,
                     bool useHarrisDetector, double harrisK)
{
    getGoodFeatures(_image, _corners, maxCorners, qualityLevel, minDistance,
                    _mask, blockSize, 3, useHarrisDetector, harrisK);
}

// Computes a gradient orientation histogram at a specified pixel
static float yuchengcalcOrientationHist(const Mat &img, Point pt, int radius,
                                        float sigma, float *hist, int n)
{
    int i, j, k, len = (radius * 2 + 1) * (radius * 2 + 1);

    float expf_scale = -1.f / (2.f * sigma * sigma);
    AutoBuffer<float> buf(len * 4 + n + 4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float *temphist = W + len + 2;

    for (i = 0; i < n; i++)
        temphist[i] = 0.f;

    for (i = -radius, k = 0; i <= radius; i++)
    {
        int y = pt.y + i;
        if (y <= 0 || y >= img.rows - 1)
            continue;
        for (j = -radius; j <= radius; j++)
        {
            int x = pt.x + j;
            if (x <= 0 || x >= img.cols - 1)
                continue;

            float dx = (float)(img.at<char>(y, x + 1) - img.at<char>(y, x - 1));
            float dy = (float)(img.at<char>(y - 1, x) - img.at<char>(y + 1, x));

            X[k] = dx;
            Y[k] = dy;
            W[k] = (i * i + j * j) * expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    k = 0;
#if CV_AVX2
    if (USE_AVX2)
    {
        __m256 __nd360 = _mm256_set1_ps(n / 360.f);
        __m256i __n = _mm256_set1_epi32(n);
        int CV_DECL_ALIGNED(32) bin_buf[8];
        float CV_DECL_ALIGNED(32) w_mul_mag_buf[8];
        for (; k <= len - 8; k += 8)
        {
            __m256i __bin = _mm256_cvtps_epi32(_mm256_mul_ps(__nd360, _mm256_loadu_ps(&Ori[k])));

            __bin = _mm256_sub_epi32(__bin, _mm256_andnot_si256(_mm256_cmpgt_epi32(__n, __bin), __n));
            __bin = _mm256_add_epi32(__bin, _mm256_and_si256(__n, _mm256_cmpgt_epi32(_mm256_setzero_si256(), __bin)));

            __m256 __w_mul_mag = _mm256_mul_ps(_mm256_loadu_ps(&W[k]), _mm256_loadu_ps(&Mag[k]));

            _mm256_store_si256((__m256i *)bin_buf, __bin);
            _mm256_store_ps(w_mul_mag_buf, __w_mul_mag);

            temphist[bin_buf[0]] += w_mul_mag_buf[0];
            temphist[bin_buf[1]] += w_mul_mag_buf[1];
            temphist[bin_buf[2]] += w_mul_mag_buf[2];
            temphist[bin_buf[3]] += w_mul_mag_buf[3];
            temphist[bin_buf[4]] += w_mul_mag_buf[4];
            temphist[bin_buf[5]] += w_mul_mag_buf[5];
            temphist[bin_buf[6]] += w_mul_mag_buf[6];
            temphist[bin_buf[7]] += w_mul_mag_buf[7];
        }
    }
#endif
    for (; k < len; k++)
    {
        int bin = cvRound((n / 360.f) * Ori[k]);
        if (bin >= n)
            bin -= n;
        if (bin < 0)
            bin += n;
        temphist[bin] += W[k] * Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n - 1];
    temphist[-2] = temphist[n - 2];
    temphist[n] = temphist[0];
    temphist[n + 1] = temphist[1];

    i = 0;
#if CV_AVX2
    if (USE_AVX2)
    {
        __m256 __d_1_16 = _mm256_set1_ps(1.f / 16.f);
        __m256 __d_4_16 = _mm256_set1_ps(4.f / 16.f);
        __m256 __d_6_16 = _mm256_set1_ps(6.f / 16.f);
        for (; i <= n - 8; i += 8)
        {
#if CV_FMA3
            __m256 __hist = _mm256_fmadd_ps(
                _mm256_add_ps(_mm256_loadu_ps(&temphist[i - 2]), _mm256_loadu_ps(&temphist[i + 2])),
                __d_1_16,
                _mm256_fmadd_ps(
                    _mm256_add_ps(_mm256_loadu_ps(&temphist[i - 1]), _mm256_loadu_ps(&temphist[i + 1])),
                    __d_4_16,
                    _mm256_mul_ps(_mm256_loadu_ps(&temphist[i]), __d_6_16)));
#else
            __m256 __hist = _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_add_ps(_mm256_loadu_ps(&temphist[i - 2]), _mm256_loadu_ps(&temphist[i + 2])),
                    __d_1_16),
                _mm256_add_ps(
                    _mm256_mul_ps(
                        _mm256_add_ps(_mm256_loadu_ps(&temphist[i - 1]), _mm256_loadu_ps(&temphist[i + 1])),
                        __d_4_16),
                    _mm256_mul_ps(_mm256_loadu_ps(&temphist[i]), __d_6_16)));
#endif
            _mm256_storeu_ps(&hist[i], __hist);
        }
    }
#endif
    for (; i < n; i++)
    {
        hist[i] = (temphist[i - 2] + temphist[i + 2]) * (1.f / 16.f) +
                  (temphist[i - 1] + temphist[i + 1]) * (4.f / 16.f) +
                  temphist[i] * (6.f / 16.f);
    }

    float maxval = hist[0];
    for (i = 1; i < n; i++)
        maxval = std::max(maxval, hist[i]);

    return maxval;
}

void calKeyPointbyGoodFeatures(cv::InputArray _image,
                               std::vector<cv::Point2f> corners,
                               std::vector<cv::KeyPoint> &keypoints)
{
    if (_image.empty())
    {
        keypoints.clear();
        return;
    }

    cv::Mat original_image_gray = _image.getMat();

    //设置角点检测参数
    // std::vector<cv::Point2f> corners;
    int max_corners = 2000;
    double quality_level = 0.01;
    double min_distance = 3.0;
    int block_size = 3;
    bool use_harris = false;
    double k = 0.04;
    //角点检测
    getGoodFeatures(original_image_gray,
                    corners,
                    max_corners,
                    quality_level,
                    min_distance,
                    cv::Mat(),
                    block_size,
                    use_harris,
                    k);

    float hist[36];

    for (int i = 0; i < corners.size(); i++)
    {

        float omax = yuchengcalcOrientationHist(original_image_gray,
                                                corners[i],
                                                3,
                                                1,
                                                hist, 36);
        float mag_thr = (float)(omax * 0.8);
        float angle = 0.0;
        for (int j = 0; j < 36; j++)
        {
            int l = j > 0 ? j - 1 : 36 - 1;
            int r2 = j < 36 - 1 ? j + 1 : 0;

            if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
            {
                float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
                bin = bin < 0 ? 36 + bin : bin >= 36 ? bin - 36 : bin;

                                angle = 360.f - (float)((360.f / 36) * bin);
                if (std::abs(angle - 360.f) < FLT_EPSILON)
                    angle = 0.f;
            }
        }

        cv::KeyPoint temp_keypoint(corners[i], 10, angle);
        keypoints.push_back(temp_keypoint);
    }
}