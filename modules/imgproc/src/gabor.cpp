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

    if( ktype == CV_32F )
    {
        // SIMD optimized path for CV_32F
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vecsize = VTraits<v_float32>::vlanes();
        v_float32 v_c = vx_setall_f32((float)c);
        v_float32 v_s = vx_setall_f32((float)s);
        v_float32 v_ex = vx_setall_f32((float)ex);
        v_float32 v_ey = vx_setall_f32((float)ey);
        v_float32 v_scale = vx_setall_f32((float)scale);
        v_float32 v_cscale = vx_setall_f32((float)cscale);
        v_float32 v_psi = vx_setall_f32((float)psi);
        
        // Create increment vector [0, 1, 2, 3, ...]
        float inc_vals[VTraits<v_float32>::max_nlanes];
        for (int i = 0; i < vecsize; i++)
            inc_vals[i] = (float)i;
        v_float32 v_inc = vx_load(inc_vals);
#endif

        for( int y = ymin; y <= ymax; y++ )
        {
            float* kptr = kernel.ptr<float>(ymax - y);
            float fy = (float)y;
            int x = xmin;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 v_y = vx_setall_f32(fy);
            v_float32 v_xmin = vx_setall_f32((float)xmin);
            
            // Process multiple pixels using SIMD
            for( ; x <= xmax - vecsize + 1; x += vecsize )
            {
                // Create vector of x values [x, x+1, x+2, ...]
                v_float32 v_x = v_add(v_xmin, v_inc);
                
                // Compute rotated coordinates
                v_float32 v_xr = v_fma(v_x, v_c, v_mul(v_y, v_s));  // xr = x*c + y*s
                v_float32 v_yr = v_sub(v_mul(v_y, v_c), v_mul(v_x, v_s));  // yr = -x*s + y*c
                
                // Compute Gaussian part: exp(ex*xr*xr + ey*yr*yr)
                v_float32 v_xr2 = v_mul(v_xr, v_xr);
                v_float32 v_yr2 = v_mul(v_yr, v_yr);
                v_float32 v_exp_arg = v_fma(v_ex, v_xr2, v_mul(v_ey, v_yr2));
                v_float32 v_gauss = v_exp(v_exp_arg);
                
                // Compute cosine part: cos(cscale*xr + psi)
                v_float32 v_cos_arg = v_fma(v_cscale, v_xr, v_psi);
                v_float32 v_cosine = v_cos(v_cos_arg);
                
                // Combine: scale * gaussian * cosine
                v_float32 v_result = v_mul(v_scale, v_mul(v_gauss, v_cosine));
                
                // Store results with proper indexing
                // We need to handle the reversed indexing manually
                float CV_DECL_ALIGNED(32) temp[VTraits<v_float32>::max_nlanes];
                v_store_aligned(temp, v_result);
                for (int i = 0; i < vecsize; i++)
                    kptr[xmax - x - i] = temp[i];
                
                // Update xmin for next iteration
                v_xmin = v_add(v_xmin, vx_setall_f32((float)vecsize));
            }
#endif

            // Process remaining pixels
            for( ; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = -x*s + y*c;

                double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
                kptr[xmax - x] = (float)v;
            }
        }
    }
    else // CV_64F
    {
        // SIMD optimized path for CV_64F
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        const int vecsize = VTraits<v_float64>::vlanes();
        v_float64 v_c = vx_setall_f64(c);
        v_float64 v_s = vx_setall_f64(s);
        v_float64 v_ex = vx_setall_f64(ex);
        v_float64 v_ey = vx_setall_f64(ey);
        v_float64 v_scale = vx_setall_f64(scale);
        v_float64 v_cscale = vx_setall_f64(cscale);
        v_float64 v_psi = vx_setall_f64(psi);
        
        // Create increment vector [0, 1, 2, 3, ...]
        double inc_vals[VTraits<v_float64>::max_nlanes];
        for (int i = 0; i < vecsize; i++)
            inc_vals[i] = (double)i;
        v_float64 v_inc = vx_load(inc_vals);
#endif

        for( int y = ymin; y <= ymax; y++ )
        {
            double* kptr = kernel.ptr<double>(ymax - y);
            double dy = (double)y;
            int x = xmin;

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
            v_float64 v_y = vx_setall_f64(dy);
            v_float64 v_xmin = vx_setall_f64((double)xmin);
            
            // Process multiple pixels using SIMD
            for( ; x <= xmax - vecsize + 1; x += vecsize )
            {
                // Create vector of x values [x, x+1, x+2, ...]
                v_float64 v_x = v_add(v_xmin, v_inc);
                
                // Compute rotated coordinates
                v_float64 v_xr = v_fma(v_x, v_c, v_mul(v_y, v_s));  // xr = x*c + y*s
                v_float64 v_yr = v_sub(v_mul(v_y, v_c), v_mul(v_x, v_s));  // yr = -x*s + y*c
                
                // Compute Gaussian part: exp(ex*xr*xr + ey*yr*yr)
                v_float64 v_xr2 = v_mul(v_xr, v_xr);
                v_float64 v_yr2 = v_mul(v_yr, v_yr);
                v_float64 v_exp_arg = v_fma(v_ex, v_xr2, v_mul(v_ey, v_yr2));
                v_float64 v_gauss = v_exp(v_exp_arg);
                
                // Compute cosine part: cos(cscale*xr + psi)
                v_float64 v_cos_arg = v_fma(v_cscale, v_xr, v_psi);
                v_float64 v_cosine = v_cos(v_cos_arg);
                
                // Combine: scale * gaussian * cosine
                v_float64 v_result = v_mul(v_scale, v_mul(v_gauss, v_cosine));
                
                // Store results with proper indexing
                // We need to handle the reversed indexing manually
                double CV_DECL_ALIGNED(32) temp[VTraits<v_float64>::max_nlanes];
                v_store_aligned(temp, v_result);
                for (int i = 0; i < vecsize; i++)
                    kptr[xmax - x - i] = temp[i];
                
                // Update xmin for next iteration
                v_xmin = v_add(v_xmin, vx_setall_f64((double)vecsize));
            }
#endif

            // Process remaining pixels
            for( ; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = -x*s + y*c;

                double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
                kptr[xmax - x] = v;
            }
        }
    }

    return kernel;
}


/* End of file. */
