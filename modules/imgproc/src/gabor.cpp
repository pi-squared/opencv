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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2012, Willow Garage Inc., all rights reserved.
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

/*
 Gabor filters and such. To be greatly extended to have full texture analysis.
 For the formulas and the explanation of the parameters see:
 http://en.wikipedia.org/wiki/Gabor_filter
*/

cv::Mat cv::getGaborKernel( Size ksize, double sigma, double theta,
                            double lambd, double gamma, double psi, int ktype )
{
    double sigma_x = sigma;
    double sigma_y = sigma/gamma;
    int nstds = 3;
    int xmin, xmax, ymin, ymax;
    double c = cos(theta), s = sin(theta);

    if( ksize.width > 0 )
        xmax = ksize.width/2;
    else
        xmax = cvRound(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));

    if( ksize.height > 0 )
        ymax = ksize.height/2;
    else
        ymax = cvRound(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));

    xmin = -xmax;
    ymin = -ymax;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );

    Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
    double scale = 1;
    double ex = -0.5/(sigma_x*sigma_x);
    double ey = -0.5/(sigma_y*sigma_y);
    double cscale = CV_PI*2/lambd;

#if CV_SIMD
    // SIMD optimization for float type
    if( ktype == CV_32F )
    {
        const int v_width = v_float32::nlanes;
        v_float32 v_c = vx_setall_f32((float)c);
        v_float32 v_s = vx_setall_f32((float)s);
        v_float32 v_ex = vx_setall_f32((float)ex);
        v_float32 v_ey = vx_setall_f32((float)ey);
        v_float32 v_cscale = vx_setall_f32((float)cscale);
        v_float32 v_psi = vx_setall_f32((float)psi);
        v_float32 v_scale = vx_setall_f32((float)scale);

        for( int y = ymin; y <= ymax; y++ )
        {
            float* kptr = kernel.ptr<float>(ymax - y);
            v_float32 v_y = vx_setall_f32((float)y);
            
            int x = xmin;
            for( ; x <= xmax - v_width + 1; x += v_width )
            {
                // Create vector of x values (max SIMD width is 16 for AVX-512)
                CV_DECL_ALIGNED(32) float x_vals[16];
                CV_DECL_ALIGNED(32) float cos_vals[16];
                for(int i = 0; i < v_width; i++)
                    x_vals[i] = (float)(x + i);
                v_float32 v_x = vx_load_aligned(x_vals);
                
                // Compute rotated coordinates
                v_float32 v_xr = v_fma(v_x, v_c, v_mul(v_y, v_s));
                v_float32 v_yr = v_fma(v_x, v_sub(vx_setzero_f32(), v_s), v_mul(v_y, v_c));
                
                // Compute Gaussian envelope
                v_float32 v_gauss_x = v_mul(v_mul(v_ex, v_xr), v_xr);
                v_float32 v_gauss_y = v_mul(v_mul(v_ey, v_yr), v_yr);
                v_float32 v_gauss = v_exp(v_add(v_gauss_x, v_gauss_y));
                
                // Compute cosine modulation
                v_float32 v_cos_arg = v_fma(v_cscale, v_xr, v_psi);
                // Note: OpenCV doesn't have vectorized cos, so we'll compute it element-wise
                v_store_aligned(cos_vals, v_cos_arg);
                for(int i = 0; i < v_width; i++)
                    cos_vals[i] = std::cos(cos_vals[i]);
                v_float32 v_cos = vx_load_aligned(cos_vals);
                
                // Final result
                v_float32 v_result = v_mul(v_mul(v_scale, v_gauss), v_cos);
                
                // Store results - need to reverse the order since kernel is indexed as (xmax - x)
                CV_DECL_ALIGNED(32) float result_vals[16];
                v_store_aligned(result_vals, v_result);
                for(int i = 0; i < v_width; i++)
                    kptr[xmax - (x + i)] = result_vals[i];
            }
            
            // Process remaining elements
            for( ; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = -x*s + y*c;
                double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
                kptr[xmax - x] = (float)v;
            }
        }
    }
    else if( ktype == CV_64F )
    {
#if CV_SIMD_64F
        const int v_width = v_float64::nlanes;
        v_float64 v_c = vx_setall<double>(c);
        v_float64 v_s = vx_setall<double>(s);
        v_float64 v_ex = vx_setall<double>(ex);
        v_float64 v_ey = vx_setall<double>(ey);
        v_float64 v_cscale = vx_setall<double>(cscale);
        v_float64 v_psi = vx_setall<double>(psi);
        v_float64 v_scale = vx_setall<double>(scale);

        for( int y = ymin; y <= ymax; y++ )
        {
            double* kptr = kernel.ptr<double>(ymax - y);
            v_float64 v_y = vx_setall<double>((double)y);
            
            int x = xmin;
            for( ; x <= xmax - v_width + 1; x += v_width )
            {
                // Create vector of x values (max SIMD width for double is 8 for AVX-512)
                CV_DECL_ALIGNED(32) double x_vals[8];
                CV_DECL_ALIGNED(32) double cos_vals[8];
                for(int i = 0; i < v_width; i++)
                    x_vals[i] = (double)(x + i);
                v_float64 v_x = vx_load_aligned(x_vals);
                
                // Compute rotated coordinates
                v_float64 v_xr = v_fma(v_x, v_c, v_mul(v_y, v_s));
                v_float64 v_yr = v_fma(v_x, v_sub(vx_setzero_f64(), v_s), v_mul(v_y, v_c));
                
                // Compute Gaussian envelope
                v_float64 v_gauss_x = v_mul(v_mul(v_ex, v_xr), v_xr);
                v_float64 v_gauss_y = v_mul(v_mul(v_ey, v_yr), v_yr);
                v_float64 v_gauss = v_exp(v_add(v_gauss_x, v_gauss_y));
                
                // Compute cosine modulation
                v_float64 v_cos_arg = v_fma(v_cscale, v_xr, v_psi);
                // Note: OpenCV doesn't have vectorized cos, so we'll compute it element-wise
                v_store_aligned(cos_vals, v_cos_arg);
                for(int i = 0; i < v_width; i++)
                    cos_vals[i] = std::cos(cos_vals[i]);
                v_float64 v_cos = vx_load_aligned(cos_vals);
                
                // Final result
                v_float64 v_result = v_mul(v_mul(v_scale, v_gauss), v_cos);
                
                // Store results - need to reverse the order since kernel is indexed as (xmax - x)
                CV_DECL_ALIGNED(32) double result_vals[8];
                v_store_aligned(result_vals, v_result);
                for(int i = 0; i < v_width; i++)
                    kptr[xmax - (x + i)] = result_vals[i];
            }
            
            // Process remaining elements
            for( ; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = -x*s + y*c;
                double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
                kptr[xmax - x] = v;
            }
        }
#else
        // Fallback to scalar code for double when SIMD double is not available
        for( int y = ymin; y <= ymax; y++ )
            for( int x = xmin; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = -x*s + y*c;

                double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
                kernel.at<double>(ymax - y, xmax - x) = v;
            }
#endif
    }
#else
    // Non-SIMD fallback
    for( int y = ymin; y <= ymax; y++ )
        for( int x = xmin; x <= xmax; x++ )
        {
            double xr = x*c + y*s;
            double yr = -x*s + y*c;

            double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
            if( ktype == CV_32F )
                kernel.at<float>(ymax - y, xmax - x) = (float)v;
            else
                kernel.at<double>(ymax - y, xmax - x) = v;
        }
#endif

    return kernel;
}


/* End of file. */
