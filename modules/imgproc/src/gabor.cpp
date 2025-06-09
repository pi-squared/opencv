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

namespace cv {

#if CV_SIMD || CV_SIMD_SCALABLE
// SIMD-optimized Gabor kernel generation for float type
static void getGaborKernel_SIMD_32F(float* kernel_data, int kernel_step,
                                    int xmin, int xmax, int ymin, int ymax,
                                    float c, float s, float ex, float ey,
                                    float cscale, float psi, float scale)
{
    const int width = xmax - xmin + 1;
    const int vlen = VTraits<v_float32>::vlanes();
    
    // Pre-compute constants
    v_float32 v_c = vx_setall_f32(c);
    v_float32 v_s = vx_setall_f32(s);
    v_float32 v_neg_s = vx_setall_f32(-s);
    v_float32 v_ex = vx_setall_f32(ex);
    v_float32 v_ey = vx_setall_f32(ey);
    v_float32 v_cscale = vx_setall_f32(cscale);
    v_float32 v_psi = vx_setall_f32(psi);
    v_float32 v_scale = vx_setall_f32(scale);
    
    for (int y = ymin; y <= ymax; y++)
    {
        v_float32 v_y = vx_setall_f32((float)y);
        float* row = kernel_data + (ymax - y) * kernel_step;
        
        int x = xmin;
        for (; x <= xmax - vlen + 1; x += vlen)
        {
            // Create vector of x values: [x, x+1, x+2, ...]
            float x_vals[VTraits<v_float32>::max_nlanes];
            for (int i = 0; i < vlen; i++)
            {
                x_vals[i] = (float)(x + i);
            }
            v_float32 v_x = vx_load(x_vals);
            
            // Compute rotated coordinates
            // xr = x*c + y*s
            v_float32 v_xr = v_muladd(v_x, v_c, v_mul(v_y, v_s));
            // yr = -x*s + y*c
            v_float32 v_yr = v_muladd(v_x, v_neg_s, v_mul(v_y, v_c));
            
            // Compute Gabor values
            // exp(ex*xr*xr + ey*yr*yr)
            v_float32 v_xr_sq = v_mul(v_xr, v_xr);
            v_float32 v_yr_sq = v_mul(v_yr, v_yr);
            v_float32 v_exp_arg = v_muladd(v_ex, v_xr_sq, v_mul(v_ey, v_yr_sq));
            v_float32 v_exp_val = v_exp(v_exp_arg);
            
            // cos(cscale*xr + psi)
            v_float32 v_cos_arg = v_muladd(v_cscale, v_xr, v_psi);
            v_float32 v_cos_val = v_cos(v_cos_arg);
            
            // Final value: scale * exp(...) * cos(...)
            v_float32 v_result = v_mul(v_scale, v_mul(v_exp_val, v_cos_val));
            
            // Store result
            v_store(row + (xmax - x - vlen + 1), v_reverse(v_result));
        }
        
        // Process remaining elements
        for (; x <= xmax; x++)
        {
            float xr = x*c + y*s;
            float yr = -x*s + y*c;
            float v = scale * std::exp(ex*xr*xr + ey*yr*yr) * std::cos(cscale*xr + psi);
            row[xmax - x] = v;
        }
    }
    vx_cleanup();
}

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
// SIMD-optimized Gabor kernel generation for double type
static void getGaborKernel_SIMD_64F(double* kernel_data, int kernel_step,
                                    int xmin, int xmax, int ymin, int ymax,
                                    double c, double s, double ex, double ey,
                                    double cscale, double psi, double scale)
{
    const int width = xmax - xmin + 1;
    const int vlen = VTraits<v_float64>::vlanes();
    
    // Pre-compute constants
    v_float64 v_c = vx_setall_f64(c);
    v_float64 v_s = vx_setall_f64(s);
    v_float64 v_neg_s = vx_setall_f64(-s);
    v_float64 v_ex = vx_setall_f64(ex);
    v_float64 v_ey = vx_setall_f64(ey);
    v_float64 v_cscale = vx_setall_f64(cscale);
    v_float64 v_psi = vx_setall_f64(psi);
    v_float64 v_scale = vx_setall_f64(scale);
    
    for (int y = ymin; y <= ymax; y++)
    {
        v_float64 v_y = vx_setall_f64((double)y);
        double* row = kernel_data + (ymax - y) * kernel_step;
        
        int x = xmin;
        for (; x <= xmax - vlen + 1; x += vlen)
        {
            // Create vector of x values: [x, x+1, x+2, ...]
            double x_vals[VTraits<v_float64>::max_nlanes];
            for (int i = 0; i < vlen; i++)
            {
                x_vals[i] = (double)(x + i);
            }
            v_float64 v_x = vx_load(x_vals);
            
            // Compute rotated coordinates
            // xr = x*c + y*s
            v_float64 v_xr = v_muladd(v_x, v_c, v_mul(v_y, v_s));
            // yr = -x*s + y*c
            v_float64 v_yr = v_muladd(v_x, v_neg_s, v_mul(v_y, v_c));
            
            // Compute Gabor values
            // exp(ex*xr*xr + ey*yr*yr)
            v_float64 v_xr_sq = v_mul(v_xr, v_xr);
            v_float64 v_yr_sq = v_mul(v_yr, v_yr);
            v_float64 v_exp_arg = v_muladd(v_ex, v_xr_sq, v_mul(v_ey, v_yr_sq));
            v_float64 v_exp_val = v_exp(v_exp_arg);
            
            // cos(cscale*xr + psi)
            v_float64 v_cos_arg = v_muladd(v_cscale, v_xr, v_psi);
            v_float64 v_cos_val = v_cos(v_cos_arg);
            
            // Final value: scale * exp(...) * cos(...)
            v_float64 v_result = v_mul(v_scale, v_mul(v_exp_val, v_cos_val));
            
            // Store result
            v_store(row + (xmax - x - vlen + 1), v_reverse(v_result));
        }
        
        // Process remaining elements
        for (; x <= xmax; x++)
        {
            double xr = x*c + y*s;
            double yr = -x*s + y*c;
            double v = scale * std::exp(ex*xr*xr + ey*yr*yr) * std::cos(cscale*xr + psi);
            row[xmax - x] = v;
        }
    }
    vx_cleanup();
}
#endif // CV_SIMD_64F
#endif // CV_SIMD

Mat getGaborKernel( Size ksize, double sigma, double theta,
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

#if CV_SIMD || CV_SIMD_SCALABLE
    // Use SIMD optimized version when available
    if( ktype == CV_32F )
    {
        getGaborKernel_SIMD_32F((float*)kernel.data, kernel.cols,
                                xmin, xmax, ymin, ymax,
                                (float)c, (float)s, (float)ex, (float)ey,
                                (float)cscale, (float)psi, (float)scale);
    }
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    else if( ktype == CV_64F )
    {
        getGaborKernel_SIMD_64F((double*)kernel.data, kernel.cols,
                                xmin, xmax, ymin, ymax,
                                c, s, ex, ey, cscale, psi, scale);
    }
#endif
    else
#endif
    {
        // Fallback to scalar implementation
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
    }

    return kernel;
}

} // namespace cv

/* End of file. */
