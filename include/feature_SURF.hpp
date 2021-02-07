#ifndef FEATURE_SURF_HPP
#define FEATURE_SURF_HPP
#include "opencv2/core.hpp"
#include "xcv/xfeatures2d_precomp.hpp"
#include  <iostream>
static const int SURF_ORI_SEARCH_INC = 5;
static const float SURF_ORI_SIGMA = 2.5f;
static const float SURF_DESC_SIGMA = 3.3f;

// Wavelet size at first layer of first octave.
static const int SURF_HAAR_SIZE0 = 9;

// Wavelet size increment between layers. This should be an even number,
// such that the wavelet sizes in an octave are either all even or all odd.
// This ensures that when looking for the neighbours of a sample, the layers
// above and below are aligned correctly.
static const int SURF_HAAR_SIZE_INC = 6;

struct SurfHF
{
    int p0, p1, p2, p3;
    float w;

    SurfHF() : p0(0), p1(0), p2(0), p3(0), w(0) {}
};

#ifdef HAVE_OPENCL
namespace cv
{
    namespace xfeatures2d
    {
        class SURF_OCL
        {
        public:
            enum KeypointLayout
            {
                X_ROW = 0,
                Y_ROW,
                LAPLACIAN_ROW,
                OCTAVE_ROW,
                SIZE_ROW,
                ANGLE_ROW,
                HESSIAN_ROW,
                ROWS_COUNT
            };

            //! the full constructor taking all the necessary parameters
            SURF_OCL();

            bool init(double _hessianThreshold,
                      int _nOctaves,
                      int _nOctaveLayers,
                      bool _extended,
                      bool _upright);

            //! returns the descriptor size in float's (64 or 128)
            int descriptorSize() const { return extended ? 128 : 64; }

            void uploadKeypoints(const std::vector<KeyPoint> &keypoints, UMat &keypointsGPU);
            void downloadKeypoints(const UMat &keypointsGPU, std::vector<KeyPoint> &keypoints);

            //! finds the keypoints using fast hessian detector used in SURF
            //! supports CV_8UC1 images
            //! keypoints will have nFeature cols and 6 rows
            //! keypoints.ptr<float>(X_ROW)[i] will contain x coordinate of i'th feature
            //! keypoints.ptr<float>(Y_ROW)[i] will contain y coordinate of i'th feature
            //! keypoints.ptr<float>(LAPLACIAN_ROW)[i] will contain laplacian sign of i'th feature
            //! keypoints.ptr<float>(OCTAVE_ROW)[i] will contain octave of i'th feature
            //! keypoints.ptr<float>(SIZE_ROW)[i] will contain size of i'th feature
            //! keypoints.ptr<float>(ANGLE_ROW)[i] will contain orientation of i'th feature
            //! keypoints.ptr<float>(HESSIAN_ROW)[i] will contain response of i'th feature
            bool detect(cv::InputArray img, InputArray mask, UMat &keypoints);
            //! finds the keypoints and computes their descriptors.
            //! Optionally it can compute descriptors for the user-provided keypoints and recompute keypoints direction
            bool detectAndCompute(InputArray img, InputArray mask, UMat &keypoints,
                                  OutputArray descriptors, bool useProvidedKeypoints = false);

        protected:
            bool setImage(InputArray img, InputArray mask);

            // kernel callers declarations
            bool calcLayerDetAndTrace(int octave, int layer_rows);

            bool findMaximaInLayer(int counterOffset, int octave, int layer_rows, int layer_cols);

            bool interpolateKeypoint(int maxCounter, UMat &keypoints, int octave, int layer_rows, int maxFeatures);

            bool calcOrientation(UMat &keypoints);

            bool setUpRight(cv::UMat &keypoints);

            bool computeDescriptors(const UMat &keypoints, OutputArray descriptors);

            bool detectKeypoints(UMat &keypoints);

            // const SURF_Impl* params;
            double hessianThreshold;
            int nOctaves;
            int nOctaveLayers;
            bool extended;
            bool upright;

            //! max keypoints = min(keypointsRatio * img.size().area(), 65535)
            UMat sum, intBuffer;
            UMat det, trace;
            UMat maxPosBuffer;

            int img_cols, img_rows;

            int maxCandidates;
            int maxFeatures;

            UMat img, counters;

            // texture buffers
            ocl::Image2D imgTex, sumTex;
            bool haveImageSupport;
            String kerOpts;

            int status;
        };
#endif // HAVE_OPENCL
    }
}

void calKeypointswithDescriptorsbySURF(cv::InputArray _image, cv::InputArray _mask,
                                       std::vector<cv::KeyPoint> &keypoints,
                                       cv::OutputArray _descriptors);

#endif // !FEATURE_SURF_HPP
