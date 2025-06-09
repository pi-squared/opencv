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
// Copyright (C) 2025, OpenCV contributors, all rights reserved.
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

#ifndef OPENCV_IMGPROC_HISTOGRAM_SIMD_HPP
#define OPENCV_IMGPROC_HISTOGRAM_SIMD_HPP

#include "opencv2/core/hal/intrin.hpp"

namespace cv {

#if CV_SIMD

// SIMD-optimized histogram calculation using multiple sub-histograms to reduce conflicts
class EqualizeHistCalcHist_SIMD_Invoker : public cv::ParallelLoopBody
{
public:
    enum { HIST_SZ = 256, SUB_HIST_COUNT = 4 };

    EqualizeHistCalcHist_SIMD_Invoker(cv::Mat& src, int* histogram, cv::Mutex* histogramLock)
        : src_(src), globalHistogram_(histogram), histogramLock_(histogramLock)
    { }

    void operator()(const cv::Range& rowRange) const CV_OVERRIDE
    {
        // Use multiple sub-histograms to reduce conflicts and improve parallelism
        CV_DECL_ALIGNED(64) int subHistograms[SUB_HIST_COUNT][HIST_SZ];
        memset(subHistograms, 0, sizeof(subHistograms));

        const size_t sstep = src_.step;
        int width = src_.cols;
        int height = rowRange.end - rowRange.start;

        if (src_.isContinuous())
        {
            width *= height;
            height = 1;
        }

        const int vlen = cv::v_uint8::nlanes;
        const int vlen4 = vlen * 4;

        for (const uchar* ptr = src_.ptr<uchar>(rowRange.start); height--; ptr += sstep)
        {
            int x = 0;

            // Process 4 vectors at a time for better ILP
            for (; x <= width - vlen4; x += vlen4)
            {
                cv::v_uint8 v0 = cv::vx_load(ptr + x);
                cv::v_uint8 v1 = cv::vx_load(ptr + x + vlen);
                cv::v_uint8 v2 = cv::vx_load(ptr + x + vlen*2);
                cv::v_uint8 v3 = cv::vx_load(ptr + x + vlen*3);

                // Prefetch next cache line
#ifdef CV_CPU_OPTIMIZATION_HINTS_AVAILABLE
                CV_CPU_OPTIMIZATION_HINTS_PREFETCH_READ(ptr + x + vlen4);
#endif

                // Process each vector into different sub-histograms
                updateSubHistogram(v0, subHistograms[0]);
                updateSubHistogram(v1, subHistograms[1]);
                updateSubHistogram(v2, subHistograms[2]);
                updateSubHistogram(v3, subHistograms[3]);
            }

            // Process remaining elements with single vector
            for (; x <= width - vlen; x += vlen)
            {
                cv::v_uint8 v = cv::vx_load(ptr + x);
                updateSubHistogram(v, subHistograms[x % SUB_HIST_COUNT]);
            }

            // Process remaining scalar elements
            for (; x < width; ++x)
            {
                subHistograms[x % SUB_HIST_COUNT][ptr[x]]++;
            }
        }

        // Merge sub-histograms into local histogram
        CV_DECL_ALIGNED(64) int localHistogram[HIST_SZ];
        mergeSubHistograms(subHistograms, localHistogram);

        // Update global histogram
        cv::AutoLock lock(*histogramLock_);
        addHistograms(localHistogram, globalHistogram_);
    }

    static bool isWorthParallel(const cv::Mat& src)
    {
        return (src.total() >= 320*240);  // Lower threshold for SIMD version
    }

private:
    // Update sub-histogram from a SIMD vector
    void updateSubHistogram(const cv::v_uint8& v, int* subHist) const
    {
        // Extract bytes and update histogram
        // This is still a serial operation but distributed across sub-histograms
        uchar CV_DECL_ALIGNED(64) buf[cv::v_uint8::nlanes];
        cv::vx_store_aligned(buf, v);
        
        // Unroll by 4 for better performance
        int i = 0;
        for (; i <= cv::v_uint8::nlanes - 4; i += 4)
        {
            subHist[buf[i]]++;
            subHist[buf[i+1]]++;
            subHist[buf[i+2]]++;
            subHist[buf[i+3]]++;
        }
        for (; i < cv::v_uint8::nlanes; i++)
        {
            subHist[buf[i]]++;
        }
    }

    // Merge sub-histograms using SIMD
    void mergeSubHistograms(const int subHists[SUB_HIST_COUNT][HIST_SZ], int* result) const
    {
        const int vlen = cv::v_int32::nlanes;
        
        int i = 0;
        for (; i <= HIST_SZ - vlen; i += vlen)
        {
            cv::v_int32 sum = cv::vx_setzero_s32();
            for (int j = 0; j < SUB_HIST_COUNT; j++)
            {
                sum = cv::v_add(sum, cv::vx_load(subHists[j] + i));
            }
            cv::v_store(result + i, sum);
        }
        
        // Handle remaining elements
        for (; i < HIST_SZ; i++)
        {
            int sum = 0;
            for (int j = 0; j < SUB_HIST_COUNT; j++)
            {
                sum += subHists[j][i];
            }
            result[i] = sum;
        }
    }

    // Add two histograms using SIMD
    void addHistograms(const int* src, int* dst) const
    {
        const int vlen = cv::v_int32::nlanes;
        
        int i = 0;
        for (; i <= HIST_SZ - vlen; i += vlen)
        {
            cv::v_int32 v1 = cv::vx_load(src + i);
            cv::v_int32 v2 = cv::vx_load(dst + i);
            cv::v_store(dst + i, cv::v_add(v1, v2));
        }
        
        for (; i < HIST_SZ; i++)
        {
            dst[i] += src[i];
        }
    }

    cv::Mat& src_;
    int* globalHistogram_;
    cv::Mutex* histogramLock_;
};

// SIMD-optimized LUT application
class EqualizeHistLut_SIMD_Invoker : public cv::ParallelLoopBody
{
public:
    EqualizeHistLut_SIMD_Invoker(cv::Mat& src, cv::Mat& dst, int* lut)
        : src_(src), dst_(dst), lut_(lut)
    { }

    void operator()(const cv::Range& rowRange) const CV_OVERRIDE
    {
        const size_t sstep = src_.step;
        const size_t dstep = dst_.step;
        int width = src_.cols;
        int height = rowRange.end - rowRange.start;

        if (src_.isContinuous() && dst_.isContinuous())
        {
            width *= height;
            height = 1;
        }

        const int vlen = cv::v_uint8::nlanes;
        const int vlen4 = vlen * 4;

        const uchar* sptr = src_.ptr<uchar>(rowRange.start);
        uchar* dptr = dst_.ptr<uchar>(rowRange.start);

        // Create LUT in uchar format for direct use
        uchar CV_DECL_ALIGNED(64) lut_u8[256];
        for (int i = 0; i < 256; i++)
        {
            lut_u8[i] = saturate_cast<uchar>(lut_[i]);
        }

        for (; height--; sptr += sstep, dptr += dstep)
        {
            int x = 0;

            // Process 4 vectors at a time
            for (; x <= width - vlen4; x += vlen4)
            {
                // Load input values
                cv::v_uint8 v0 = cv::vx_load(sptr + x);
                cv::v_uint8 v1 = cv::vx_load(sptr + x + vlen);
                cv::v_uint8 v2 = cv::vx_load(sptr + x + vlen*2);
                cv::v_uint8 v3 = cv::vx_load(sptr + x + vlen*3);

                // Prefetch next cache line
#ifdef CV_CPU_OPTIMIZATION_HINTS_AVAILABLE
                CV_CPU_OPTIMIZATION_HINTS_PREFETCH_READ(sptr + x + vlen4);
                CV_CPU_OPTIMIZATION_HINTS_PREFETCH_WRITE(dptr + x + vlen4);
#endif

                // Apply LUT - we need to extract and process
                // This is still somewhat serial but organized for better cache usage
                cv::v_uint8 r0 = applyLutVector(v0, lut_u8);
                cv::v_uint8 r1 = applyLutVector(v1, lut_u8);
                cv::v_uint8 r2 = applyLutVector(v2, lut_u8);
                cv::v_uint8 r3 = applyLutVector(v3, lut_u8);

                // Store results
                cv::v_store(dptr + x, r0);
                cv::v_store(dptr + x + vlen, r1);
                cv::v_store(dptr + x + vlen*2, r2);
                cv::v_store(dptr + x + vlen*3, r3);
            }

            // Process single vector
            for (; x <= width - vlen; x += vlen)
            {
                cv::v_uint8 v = cv::vx_load(sptr + x);
                cv::v_uint8 r = applyLutVector(v, lut_u8);
                cv::v_store(dptr + x, r);
            }

            // Process remaining scalar elements with 4x unrolling
            for (; x <= width - 4; x += 4)
            {
                dptr[x] = lut_u8[sptr[x]];
                dptr[x+1] = lut_u8[sptr[x+1]];
                dptr[x+2] = lut_u8[sptr[x+2]];
                dptr[x+3] = lut_u8[sptr[x+3]];
            }

            for (; x < width; ++x)
            {
                dptr[x] = lut_u8[sptr[x]];
            }
        }
    }

    static bool isWorthParallel(const cv::Mat& src)
    {
        return (src.total() >= 320*240);  // Lower threshold for SIMD version
    }

private:
    // Apply LUT to a vector of values
    cv::v_uint8 applyLutVector(const cv::v_uint8& indices, const uchar* lut) const
    {
        // Extract indices and apply LUT
        // This is still a gather operation but organized for cache efficiency
        uchar CV_DECL_ALIGNED(64) idx_buf[cv::v_uint8::nlanes];
        uchar CV_DECL_ALIGNED(64) res_buf[cv::v_uint8::nlanes];
        
        cv::vx_store_aligned(idx_buf, indices);
        
        // Unroll by 4 for better performance
        int i = 0;
        for (; i <= cv::v_uint8::nlanes - 4; i += 4)
        {
            res_buf[i] = lut[idx_buf[i]];
            res_buf[i+1] = lut[idx_buf[i+1]];
            res_buf[i+2] = lut[idx_buf[i+2]];
            res_buf[i+3] = lut[idx_buf[i+3]];
        }
        for (; i < cv::v_uint8::nlanes; i++)
        {
            res_buf[i] = lut[idx_buf[i]];
        }
        
        return cv::vx_load_aligned(res_buf);
    }

    cv::Mat& src_;
    cv::Mat& dst_;
    int* lut_;
};

#endif // CV_SIMD

} // namespace cv

#endif // OPENCV_IMGPROC_HISTOGRAM_SIMD_HPP