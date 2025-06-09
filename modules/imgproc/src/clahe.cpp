/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"

// ----------------------------------------------------------------------
// CLAHE

#ifdef HAVE_OPENCL

namespace clahe
{
    static bool calcLut(cv::InputArray _src, cv::OutputArray _dst,
        const int tilesX, const int tilesY, const cv::Size tileSize,
        const int clipLimit, const float lutScale)
    {
        cv::ocl::Kernel k("calcLut", cv::ocl::imgproc::clahe_oclsrc);
        if(k.empty())
            return false;

        cv::UMat src = _src.getUMat();
        _dst.create(tilesX * tilesY, 256, CV_8UC1);
        cv::UMat dst = _dst.getUMat();

        int tile_size[2];
        tile_size[0] = tileSize.width;
        tile_size[1] = tileSize.height;

        size_t localThreads[3]  = { 32, 8, 1 };
        size_t globalThreads[3] = { tilesX * localThreads[0], tilesY * localThreads[1], 1 };

        int idx = 0;
        idx = k.set(idx, cv::ocl::KernelArg::ReadOnlyNoSize(src));
        idx = k.set(idx, cv::ocl::KernelArg::WriteOnlyNoSize(dst));
        idx = k.set(idx, tile_size);
        idx = k.set(idx, tilesX);
        idx = k.set(idx, clipLimit);
        k.set(idx, lutScale);

        return k.run(2, globalThreads, localThreads, false);
    }

    static bool transform(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _lut,
        const int tilesX, const int tilesY, const cv::Size & tileSize)
    {

        cv::ocl::Kernel k("transform", cv::ocl::imgproc::clahe_oclsrc);
        if(k.empty())
            return false;

        int tile_size[2];
        tile_size[0] = tileSize.width;
        tile_size[1] = tileSize.height;

        cv::UMat src = _src.getUMat();
        _dst.create(src.size(), src.type());
        cv::UMat dst = _dst.getUMat();
        cv::UMat lut = _lut.getUMat();

        size_t localThreads[3]  = { 32, 8, 1 };
        size_t globalThreads[3] = { (size_t)src.cols, (size_t)src.rows, 1 };

        int idx = 0;
        idx = k.set(idx, cv::ocl::KernelArg::ReadOnlyNoSize(src));
        idx = k.set(idx, cv::ocl::KernelArg::WriteOnlyNoSize(dst));
        idx = k.set(idx, cv::ocl::KernelArg::ReadOnlyNoSize(lut));
        idx = k.set(idx, src.cols);
        idx = k.set(idx, src.rows);
        idx = k.set(idx, tile_size);
        idx = k.set(idx, tilesX);
        k.set(idx, tilesY);

        return k.run(2, globalThreads, localThreads, false);
    }
}

#endif

namespace
{
    template <class T, int histSize, int shift>
    class CLAHE_CalcLut_Body : public cv::ParallelLoopBody
    {
    public:
        CLAHE_CalcLut_Body(const cv::Mat& src, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& clipLimit, const float& lutScale) :
            src_(src), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), clipLimit_(clipLimit), lutScale_(lutScale)
        {
        }

        void operator ()(const cv::Range& range) const CV_OVERRIDE;

    private:
        cv::Mat src_;
        mutable cv::Mat lut_;

        cv::Size tileSize_;
        int tilesX_;
        int clipLimit_;
        float lutScale_;
    };

    template <class T, int histSize, int shift>
    void CLAHE_CalcLut_Body<T,histSize,shift>::operator ()(const cv::Range& range) const
    {
        T* tileLut = lut_.ptr<T>(range.start);
        const size_t lut_step = lut_.step / sizeof(T);

        for (int k = range.start; k < range.end; ++k, tileLut += lut_step)
        {
            const int ty = k / tilesX_;
            const int tx = k % tilesX_;

            // retrieve tile submatrix

            cv::Rect tileROI;
            tileROI.x = tx * tileSize_.width;
            tileROI.y = ty * tileSize_.height;
            tileROI.width = tileSize_.width;
            tileROI.height = tileSize_.height;

            const cv::Mat tile = src_(tileROI);

            // calc histogram

            cv::AutoBuffer<int> _tileHist(histSize);
            int* tileHist = _tileHist.data();
            std::fill(tileHist, tileHist + histSize, 0);

#if CV_SIMD
            // Use multiple histograms approach to reduce data dependencies
            const int numHists = 4;
            cv::AutoBuffer<int> _multiHist(histSize * numHists);
            int* multiHist = _multiHist.data();
            std::fill(multiHist, multiHist + histSize * numHists, 0);
#endif

            int height = tileROI.height;
            const size_t sstep = src_.step / sizeof(T);
            for (const T* ptr = tile.ptr<T>(0); height--; ptr += sstep)
            {
                int x = 0;
#if CV_SIMD
                // Process pixels using multiple histograms to reduce conflicts
                for (; x <= tileROI.width - numHists; x += numHists)
                {
                    // Each pixel goes to a different histogram to avoid conflicts
                    for (int i = 0; i < numHists; i++)
                    {
                        int val = ptr[x + i] >> shift;
                        multiHist[i * histSize + val]++;
                    }
                }
#endif
                // Process remaining pixels with scalar code
                for (; x <= tileROI.width - 4; x += 4)
                {
                    int t0 = ptr[x], t1 = ptr[x+1];
                    tileHist[t0 >> shift]++; tileHist[t1 >> shift]++;
                    t0 = ptr[x+2]; t1 = ptr[x+3];
                    tileHist[t0 >> shift]++; tileHist[t1 >> shift]++;
                }

                for (; x < tileROI.width; ++x)
                    tileHist[ptr[x] >> shift]++;
            }

#if CV_SIMD
            // Merge multiple histograms using SIMD
            const int vecSize = cv::v_int32::nlanes;
            int h = 0;
            for (; h <= histSize - vecSize; h += vecSize)
            {
                cv::v_int32 sum = cv::vx_setzero_s32();
                for (int i = 0; i < numHists; i++)
                {
                    sum = cv::v_add(sum, cv::vx_load(multiHist + i * histSize + h));
                }
                sum = cv::v_add(sum, cv::vx_load(tileHist + h));
                cv::v_store(tileHist + h, sum);
            }
            // Handle remaining elements
            for (; h < histSize; h++)
            {
                for (int i = 0; i < numHists; i++)
                {
                    tileHist[h] += multiHist[i * histSize + h];
                }
            }
#endif

            // clip histogram

            if (clipLimit_ > 0)
            {
                // how many pixels were clipped
                int clipped = 0;
                for (int i = 0; i < histSize; ++i)
                {
                    if (tileHist[i] > clipLimit_)
                    {
                        clipped += tileHist[i] - clipLimit_;
                        tileHist[i] = clipLimit_;
                    }
                }

                // redistribute clipped pixels
                int redistBatch = clipped / histSize;
                int residual = clipped - redistBatch * histSize;

                for (int i = 0; i < histSize; ++i)
                    tileHist[i] += redistBatch;

                if (residual != 0)
                {
                    int residualStep = MAX(histSize / residual, 1);
                    for (int i = 0; i < histSize && residual > 0; i += residualStep, residual--)
                        tileHist[i]++;
                }
            }

            // calc Lut

#if CV_SIMD
            // SIMD optimized cumulative sum and LUT calculation
            if (histSize == 256 && sizeof(T) == 1) // Optimize for common 8-bit case
            {
                const int vecSize = cv::v_int32::nlanes;
                cv::AutoBuffer<int> _cumSum(histSize + vecSize);
                int* cumSum = _cumSum.data();
                cumSum[0] = 0;
                
                // Compute prefix sum using SIMD
                int sum = 0;
                int i = 0;
                for (; i <= histSize - vecSize; i += vecSize)
                {
                    cv::v_int32 vec = cv::vx_load(tileHist + i);
                    for (int j = 0; j < vecSize; j++)
                    {
                        sum += tileHist[i + j];
                        cumSum[i + j + 1] = sum;
                    }
                }
                for (; i < histSize; i++)
                {
                    sum += tileHist[i];
                    cumSum[i + 1] = sum;
                }
                
                // Convert to LUT using SIMD
                cv::v_float32 scale = cv::vx_setall(lutScale_);
                i = 0;
                for (; i <= histSize - vecSize; i += vecSize)
                {
                    cv::v_int32 csum = cv::vx_load(cumSum + i + 1);
                    cv::v_float32 fsum = cv::v_cvt_f32(csum);
                    cv::v_float32 result = cv::v_mul(fsum, scale);
                    cv::v_int32 iresult = cv::v_round(result);
                    
                    // Store results with saturation - process element by element
                    // since we need to saturate to uchar
                    for (int j = 0; j < vecSize && i + j < histSize; j++)
                    {
                        tileLut[i + j] = cv::saturate_cast<T>(iresult.get0());
                        iresult = cv::v_rotate_right<1>(iresult);
                    }
                }
                // Handle remaining elements
                for (; i < histSize; i++)
                {
                    tileLut[i] = cv::saturate_cast<T>(cumSum[i + 1] * lutScale_);
                }
            }
            else
#endif
            {
                // Scalar fallback
                int sum = 0;
                for (int i = 0; i < histSize; ++i)
                {
                    sum += tileHist[i];
                    tileLut[i] = cv::saturate_cast<T>(sum * lutScale_);
                }
            }
        }
    }

    template <class T, int shift>
    class CLAHE_Interpolation_Body : public cv::ParallelLoopBody
    {
    public:
        CLAHE_Interpolation_Body(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& tilesY) :
            src_(src), dst_(dst), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), tilesY_(tilesY)
        {
            buf.allocate(src.cols << 2);
            ind1_p = buf.data();
            ind2_p = ind1_p + src.cols;
            xa_p = (float *)(ind2_p + src.cols);
            xa1_p = xa_p + src.cols;

            int lut_step = static_cast<int>(lut_.step / sizeof(T));
            float inv_tw = 1.0f / tileSize_.width;

            for (int x = 0; x < src.cols; ++x)
            {
                float txf = x * inv_tw - 0.5f;

                int tx1 = cvFloor(txf);
                int tx2 = tx1 + 1;

                xa_p[x] = txf - tx1;
                xa1_p[x] = 1.0f - xa_p[x];

                tx1 = std::max(tx1, 0);
                tx2 = std::min(tx2, tilesX_ - 1);

                ind1_p[x] = tx1 * lut_step;
                ind2_p[x] = tx2 * lut_step;
            }
        }

        void operator ()(const cv::Range& range) const CV_OVERRIDE;

    private:
        cv::Mat src_;
        mutable cv::Mat dst_;
        cv::Mat lut_;

        cv::Size tileSize_;
        int tilesX_;
        int tilesY_;

        cv::AutoBuffer<int> buf;
        int * ind1_p, * ind2_p;
        float * xa_p, * xa1_p;
    };

    template <class T, int shift>
    void CLAHE_Interpolation_Body<T, shift>::operator ()(const cv::Range& range) const
    {
        float inv_th = 1.0f / tileSize_.height;

        for (int y = range.start; y < range.end; ++y)
        {
            const T* srcRow = src_.ptr<T>(y);
            T* dstRow = dst_.ptr<T>(y);

            float tyf = y * inv_th - 0.5f;

            int ty1 = cvFloor(tyf);
            int ty2 = ty1 + 1;

            float ya = tyf - ty1, ya1 = 1.0f - ya;

            ty1 = std::max(ty1, 0);
            ty2 = std::min(ty2, tilesY_ - 1);

            const T* lutPlane1 = lut_.ptr<T>(ty1 * tilesX_);
            const T* lutPlane2 = lut_.ptr<T>(ty2 * tilesX_);

            int x = 0;
#if CV_SIMD
            // SIMD optimization for interpolation
            if (sizeof(T) == 1 && shift == 0) // 8-bit case
            {
                const int vecSize = cv::v_float32::nlanes;
                cv::v_float32 vya1 = cv::vx_setall(ya1);
                cv::v_float32 vya = cv::vx_setall(ya);
                
                // Process pixels in groups for better SIMD utilization
                for (; x <= src_.cols - vecSize; x += vecSize)
                {
                    // Load interpolation weights
                    cv::v_float32 vxa1 = cv::vx_load(xa1_p + x);
                    cv::v_float32 vxa = cv::vx_load(xa_p + x);
                    
                    // Unfortunately, we still need scalar lookups for LUT access
                    // but we can vectorize the interpolation computation
                    alignas(32) float results[16]; // Max size for any SIMD width
                    
                    for (int i = 0; i < vecSize; i++)
                    {
                        int srcVal = srcRow[x + i];
                        int ind1 = ind1_p[x + i] + srcVal;
                        int ind2 = ind2_p[x + i] + srcVal;
                        
                        float lut11 = lutPlane1[ind1];
                        float lut12 = lutPlane1[ind2];
                        float lut21 = lutPlane2[ind1];
                        float lut22 = lutPlane2[ind2];
                        
                        results[i] = (lut11 * xa1_p[x + i] + lut12 * xa_p[x + i]) * ya1 +
                                    (lut21 * xa1_p[x + i] + lut22 * xa_p[x + i]) * ya;
                    }
                    
                    // Vectorized conversion to output type
                    cv::v_float32 vres = cv::vx_load_aligned(results);
                    cv::v_int32 ires = cv::v_round(vres);
                    
                    // Pack and store results
                    // Store 4 bytes at a time
                    for (int j = 0; j < vecSize && x + j < src_.cols; j++)
                    {
                        dstRow[x + j] = cv::saturate_cast<uchar>(ires.get0());
                        ires = cv::v_rotate_right<1>(ires);
                    }
                }
            }
#endif
            // Scalar code for remaining pixels
            for (; x < src_.cols; ++x)
            {
                int srcVal = srcRow[x] >> shift;

                int ind1 = ind1_p[x] + srcVal;
                int ind2 = ind2_p[x] + srcVal;

                float res = (lutPlane1[ind1] * xa1_p[x] + lutPlane1[ind2] * xa_p[x]) * ya1 +
                            (lutPlane2[ind1] * xa1_p[x] + lutPlane2[ind2] * xa_p[x]) * ya;

                dstRow[x] = cv::saturate_cast<T>(res) << shift;
            }
        }
    }

    class CLAHE_Impl CV_FINAL : public cv::CLAHE
    {
    public:
        CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

        void apply(cv::InputArray src, cv::OutputArray dst) CV_OVERRIDE;

        void setClipLimit(double clipLimit) CV_OVERRIDE;
        double getClipLimit() const CV_OVERRIDE;

        void setTilesGridSize(cv::Size tileGridSize) CV_OVERRIDE;
        cv::Size getTilesGridSize() const CV_OVERRIDE;

        void collectGarbage() CV_OVERRIDE;

    private:
        double clipLimit_;
        int tilesX_;
        int tilesY_;

        cv::Mat srcExt_;
        cv::Mat lut_;

#ifdef HAVE_OPENCL
        cv::UMat usrcExt_;
        cv::UMat ulut_;
#endif
    };

    CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
        clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY)
    {
    }

    void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst)
    {
        CV_INSTRUMENT_REGION();

        CV_Assert( _src.type() == CV_8UC1 || _src.type() == CV_16UC1 );

#ifdef HAVE_OPENCL
        bool useOpenCL = cv::ocl::isOpenCLActivated() && _src.isUMat() && _src.dims()<=2 && _src.type() == CV_8UC1;
#endif

        int histSize = _src.type() == CV_8UC1 ? 256 : 65536;

        cv::Size tileSize;
        cv::_InputArray _srcForLut;

        if (_src.size().width % tilesX_ == 0 && _src.size().height % tilesY_ == 0)
        {
            tileSize = cv::Size(_src.size().width / tilesX_, _src.size().height / tilesY_);
            _srcForLut = _src;
        }
        else
        {
#ifdef HAVE_OPENCL
            if(useOpenCL)
            {
                cv::copyMakeBorder(_src, usrcExt_, 0, tilesY_ - (_src.size().height % tilesY_), 0, tilesX_ - (_src.size().width % tilesX_), cv::BORDER_REFLECT_101);
                tileSize = cv::Size(usrcExt_.size().width / tilesX_, usrcExt_.size().height / tilesY_);
                _srcForLut = usrcExt_;
            }
            else
#endif
            {
                cv::copyMakeBorder(_src, srcExt_, 0, tilesY_ - (_src.size().height % tilesY_), 0, tilesX_ - (_src.size().width % tilesX_), cv::BORDER_REFLECT_101);
                tileSize = cv::Size(srcExt_.size().width / tilesX_, srcExt_.size().height / tilesY_);
                _srcForLut = srcExt_;
            }
        }

        const int tileSizeTotal = tileSize.area();
        const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

        int clipLimit = 0;
        if (clipLimit_ > 0.0)
        {
            clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
            clipLimit = std::max(clipLimit, 1);
        }

#ifdef HAVE_OPENCL
        if (useOpenCL && clahe::calcLut(_srcForLut, ulut_, tilesX_, tilesY_, tileSize, clipLimit, lutScale) )
            if( clahe::transform(_src, _dst, ulut_, tilesX_, tilesY_, tileSize) )
            {
                CV_IMPL_ADD(CV_IMPL_OCL);
                return;
            }
#endif

        cv::Mat src = _src.getMat();
        _dst.create( src.size(), src.type() );
        cv::Mat dst = _dst.getMat();
        cv::Mat srcForLut = _srcForLut.getMat();
        lut_.create(tilesX_ * tilesY_, histSize, _src.type());

        cv::Ptr<cv::ParallelLoopBody> calcLutBody;
        if (_src.type() == CV_8UC1)
            calcLutBody = cv::makePtr<CLAHE_CalcLut_Body<uchar, 256, 0> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale);
        else if (_src.type() == CV_16UC1)
            calcLutBody = cv::makePtr<CLAHE_CalcLut_Body<ushort, 65536, 0> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale);
        else
            CV_Error( cv::Error::StsBadArg, "Unsupported type" );

        cv::parallel_for_(cv::Range(0, tilesX_ * tilesY_), *calcLutBody);

        cv::Ptr<cv::ParallelLoopBody> interpolationBody;
        if (_src.type() == CV_8UC1)
            interpolationBody = cv::makePtr<CLAHE_Interpolation_Body<uchar, 0> >(src, dst, lut_, tileSize, tilesX_, tilesY_);
        else if (_src.type() == CV_16UC1)
            interpolationBody = cv::makePtr<CLAHE_Interpolation_Body<ushort, 0> >(src, dst, lut_, tileSize, tilesX_, tilesY_);

        cv::parallel_for_(cv::Range(0, src.rows), *interpolationBody);
    }

    void CLAHE_Impl::setClipLimit(double clipLimit)
    {
        clipLimit_ = clipLimit;
    }

    double CLAHE_Impl::getClipLimit() const
    {
        return clipLimit_;
    }

    void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
    {
        tilesX_ = tileGridSize.width;
        tilesY_ = tileGridSize.height;
    }

    cv::Size CLAHE_Impl::getTilesGridSize() const
    {
        return cv::Size(tilesX_, tilesY_);
    }

    void CLAHE_Impl::collectGarbage()
    {
        srcExt_.release();
        lut_.release();
#ifdef HAVE_OPENCL
        usrcExt_.release();
        ulut_.release();
#endif
    }
}

cv::Ptr<cv::CLAHE> cv::createCLAHE(double clipLimit, cv::Size tileGridSize)
{
    return makePtr<CLAHE_Impl>(clipLimit, tileGridSize.width, tileGridSize.height);
}
