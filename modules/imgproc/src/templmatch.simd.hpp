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
// Copyright (C) 2025, OpenCV Foundation, all rights reserved.
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
//   * The name of OpenCV Foundation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_IMGPROC_TEMPLMATCH_SIMD_HPP
#define OPENCV_IMGPROC_TEMPLMATCH_SIMD_HPP

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace hal {

// SIMD-optimized template matching functions for direct correlation methods
// These are used for small to medium-sized templates where FFT overhead isn't worth it

#if CV_SIMD

// Optimized squared difference calculation for TM_SQDIFF
inline float matchTemplateSqDiff_SIMD(const float* img, const float* templ, int length)
{
    const int step = v_float32::nlanes;
    v_float32 vsum = vx_setzero<v_float32>();
    
    int i = 0;
    for (; i <= length - step; i += step)
    {
        v_float32 vimg = vx_load(img + i);
        v_float32 vtempl = vx_load(templ + i);
        v_float32 vdiff = v_sub(vimg, vtempl);
        vsum = v_fma(vdiff, vdiff, vsum);
    }
    
    float sum = v_reduce_sum(vsum);
    
    // Process remaining elements
    for (; i < length; i++)
    {
        float diff = img[i] - templ[i];
        sum += diff * diff;
    }
    
    return sum;
}

// Optimized cross-correlation for TM_CCORR
inline float matchTemplateCCorr_SIMD(const float* img, const float* templ, int length)
{
    const int step = v_float32::nlanes;
    v_float32 vsum = vx_setzero<v_float32>();
    
    int i = 0;
    for (; i <= length - step; i += step)
    {
        v_float32 vimg = vx_load(img + i);
        v_float32 vtempl = vx_load(templ + i);
        vsum = v_fma(vimg, vtempl, vsum);
    }
    
    float sum = v_reduce_sum(vsum);
    
    // Process remaining elements
    for (; i < length; i++)
    {
        sum += img[i] * templ[i];
    }
    
    return sum;
}

// Compute mean using SIMD
inline float computeMean_SIMD(const float* data, int length)
{
    const int step = v_float32::nlanes;
    v_float32 vsum = vx_setzero<v_float32>();
    
    int i = 0;
    for (; i <= length - step; i += step)
    {
        vsum = v_add(vsum, vx_load(data + i));
    }
    
    float sum = v_reduce_sum(vsum);
    
    // Process remaining elements
    for (; i < length; i++)
    {
        sum += data[i];
    }
    
    return sum / length;
}

// Optimized correlation coefficient calculation for TM_CCOEFF
inline float matchTemplateCCoeff_SIMD(const float* img, const float* templ, int length, 
                                      float imgMean, float templMean)
{
    const int step = v_float32::nlanes;
    v_float32 vsum = vx_setzero<v_float32>();
    v_float32 vimgMean = vx_setall<v_float32>(imgMean);
    v_float32 vtemplMean = vx_setall<v_float32>(templMean);
    
    int i = 0;
    for (; i <= length - step; i += step)
    {
        v_float32 vimg = vx_load(img + i);
        v_float32 vtempl = vx_load(templ + i);
        v_float32 vimg_centered = v_sub(vimg, vimgMean);
        v_float32 vtempl_centered = v_sub(vtempl, vtemplMean);
        vsum = v_fma(vimg_centered, vtempl_centered, vsum);
    }
    
    float sum = v_reduce_sum(vsum);
    
    // Process remaining elements
    for (; i < length; i++)
    {
        sum += (img[i] - imgMean) * (templ[i] - templMean);
    }
    
    return sum;
}

// Compute variance (for normalization) using SIMD
inline float computeVariance_SIMD(const float* data, int length, float mean)
{
    const int step = v_float32::nlanes;
    v_float32 vsum = vx_setzero<v_float32>();
    v_float32 vmean = vx_setall<v_float32>(mean);
    
    int i = 0;
    for (; i <= length - step; i += step)
    {
        v_float32 vdata = vx_load(data + i);
        v_float32 vdiff = v_sub(vdata, vmean);
        vsum = v_fma(vdiff, vdiff, vsum);
    }
    
    float sum = v_reduce_sum(vsum);
    
    // Process remaining elements
    for (; i < length; i++)
    {
        float diff = data[i] - mean;
        sum += diff * diff;
    }
    
    return sum;
}

// Main SIMD-optimized template matching for direct methods
// This is called for each position in the result matrix
inline void matchTemplateNaive_SIMD(const Mat& img, const Mat& templ, Mat& result, int method)
{
    CV_Assert(img.depth() == CV_32F && templ.depth() == CV_32F);
    
    int rows = result.rows;
    int cols = result.cols;
    int trows = templ.rows;
    int tcols = templ.cols;
    
    // Pre-compute template statistics if needed
    float templMean = 0, templVar = 0;
    if (method == TM_CCOEFF || method == TM_CCOEFF_NORMED)
    {
        templMean = computeMean_SIMD((const float*)templ.data, trows * tcols);
        if (method == TM_CCOEFF_NORMED)
        {
            templVar = computeVariance_SIMD((const float*)templ.data, trows * tcols, templMean);
        }
    }
    
    // Process each position in the result
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float value = 0;
            
            // Extract window from image
            Mat window = img(Rect(j, i, tcols, trows));
            
            switch (method)
            {
                case TM_SQDIFF:
                    value = matchTemplateSqDiff_SIMD((const float*)window.data, 
                                                     (const float*)templ.data, 
                                                     trows * tcols);
                    break;
                    
                case TM_CCORR:
                    value = matchTemplateCCorr_SIMD((const float*)window.data, 
                                                    (const float*)templ.data, 
                                                    trows * tcols);
                    break;
                    
                case TM_CCOEFF:
                {
                    float imgMean = computeMean_SIMD((const float*)window.data, trows * tcols);
                    value = matchTemplateCCoeff_SIMD((const float*)window.data, 
                                                     (const float*)templ.data, 
                                                     trows * tcols, 
                                                     imgMean, templMean);
                    break;
                }
                
                case TM_SQDIFF_NORMED:
                {
                    float sqdiff = matchTemplateSqDiff_SIMD((const float*)window.data, 
                                                            (const float*)templ.data, 
                                                            trows * tcols);
                    float imgVar = computeVariance_SIMD((const float*)window.data, 
                                                       trows * tcols, 
                                                       computeMean_SIMD((const float*)window.data, 
                                                                       trows * tcols));
                    float denom = std::sqrt(imgVar * templVar);
                    value = denom > FLT_EPSILON ? sqdiff / denom : 1.0f;
                    break;
                }
                
                case TM_CCORR_NORMED:
                {
                    float corr = matchTemplateCCorr_SIMD((const float*)window.data, 
                                                         (const float*)templ.data, 
                                                         trows * tcols);
                    float imgNorm = std::sqrt(matchTemplateCCorr_SIMD((const float*)window.data, 
                                                                      (const float*)window.data, 
                                                                      trows * tcols));
                    float templNorm = std::sqrt(matchTemplateCCorr_SIMD((const float*)templ.data, 
                                                                        (const float*)templ.data, 
                                                                        trows * tcols));
                    float denom = imgNorm * templNorm;
                    value = denom > FLT_EPSILON ? corr / denom : 0.0f;
                    break;
                }
                
                case TM_CCOEFF_NORMED:
                {
                    float imgMean = computeMean_SIMD((const float*)window.data, trows * tcols);
                    float imgVar = computeVariance_SIMD((const float*)window.data, 
                                                       trows * tcols, imgMean);
                    float coeff = matchTemplateCCoeff_SIMD((const float*)window.data, 
                                                          (const float*)templ.data, 
                                                          trows * tcols, 
                                                          imgMean, templMean);
                    float denom = std::sqrt(imgVar * templVar);
                    value = denom > FLT_EPSILON ? coeff / denom : 0.0f;
                    break;
                }
                
                default:
                    CV_Error(cv::Error::StsBadArg, "Unknown comparison method");
            }
            
            result.at<float>(i, j) = value;
        }
    }
}

#endif // CV_SIMD

// Optimized version for small templates using cache-friendly access patterns
inline void matchTemplateNaive_CacheFriendly(const Mat& img, const Mat& templ, Mat& result, int method)
{
    CV_Assert(img.depth() == CV_32F && templ.depth() == CV_32F);
    
    int rows = result.rows;
    int cols = result.cols;
    int trows = templ.rows;
    int tcols = templ.cols;
    int imgStep = (int)(img.step / sizeof(float));
    int templStep = (int)(templ.step / sizeof(float));
    
    const float* imgData = (const float*)img.data;
    const float* templData = (const float*)templ.data;
    float* resultData = (float*)result.data;
    
    // Process blocks for better cache utilization
    const int blockSize = 32; // Tune based on L1 cache size
    
    for (int i0 = 0; i0 < rows; i0 += blockSize)
    {
        for (int j0 = 0; j0 < cols; j0 += blockSize)
        {
            int iMax = std::min(i0 + blockSize, rows);
            int jMax = std::min(j0 + blockSize, cols);
            
            for (int i = i0; i < iMax; i++)
            {
                for (int j = j0; j < jMax; j++)
                {
                    float value = 0;
                    
                    // Direct computation for each method
                    if (method == TM_SQDIFF)
                    {
                        for (int ti = 0; ti < trows; ti++)
                        {
                            const float* imgRow = imgData + (i + ti) * imgStep + j;
                            const float* templRow = templData + ti * templStep;
                            
                            for (int tj = 0; tj < tcols; tj++)
                            {
                                float diff = imgRow[tj] - templRow[tj];
                                value += diff * diff;
                            }
                        }
                    }
                    else if (method == TM_CCORR)
                    {
                        for (int ti = 0; ti < trows; ti++)
                        {
                            const float* imgRow = imgData + (i + ti) * imgStep + j;
                            const float* templRow = templData + ti * templStep;
                            
                            for (int tj = 0; tj < tcols; tj++)
                            {
                                value += imgRow[tj] * templRow[tj];
                            }
                        }
                    }
                    
                    resultData[i * cols + j] = value;
                }
            }
        }
    }
}

} // namespace hal
} // namespace cv

#endif // OPENCV_IMGPROC_TEMPLMATCH_SIMD_HPP