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
        // SIMD optimized path for float32
#if CV_SIMD
        const int vecsize = v_float32::nlanes;
        int width = xmax - xmin + 1;
        
        // Pre-compute constants
        v_float32 v_c = v_setall((float)c);
        v_float32 v_s = v_setall((float)s);
        v_float32 v_ex = v_setall((float)ex);
        v_float32 v_ey = v_setall((float)ey);
        v_float32 v_cscale = v_setall((float)cscale);
        v_float32 v_psi = v_setall((float)psi);
        v_float32 v_scale = v_setall((float)scale);
        
        // Create x coordinate increment vector
        v_float32 v_xinc;
        for( int i = 0; i < vecsize; i++ )
            v_xinc.s[i] = (float)i;
        
        for( int y = ymin; y <= ymax; y++ )
        {
            float* ptr = kernel.ptr<float>(ymax - y);
            v_float32 v_ys = v_setall((float)(y * s));
            v_float32 v_yc = v_setall((float)(y * c));
            
            int x = xmin;
            
            // Process 2 vectors at a time for better ILP
            for( ; x <= xmax - 2*vecsize + 1; x += 2*vecsize )
            {
                // First vector
                v_float32 vx1 = v_setall((float)x) + v_xinc;
                v_float32 vxr1 = vx1 * v_c + v_ys;
                v_float32 vyr1 = v_yc - vx1 * v_s;
                
                // Second vector
                v_float32 vx2 = v_setall((float)(x + vecsize)) + v_xinc;
                v_float32 vxr2 = vx2 * v_c + v_ys;
                v_float32 vyr2 = v_yc - vx2 * v_s;
                
                // Compute Gaussian parts
                v_float32 gaussian_arg1 = vxr1 * vxr1 * v_ex + vyr1 * vyr1 * v_ey;
                v_float32 gaussian_arg2 = vxr2 * vxr2 * v_ex + vyr2 * vyr2 * v_ey;
                v_float32 gaussian1 = v_exp(gaussian_arg1);
                v_float32 gaussian2 = v_exp(gaussian_arg2);
                
                // Compute cosine parts
                v_float32 cos_arg1 = vxr1 * v_cscale + v_psi;
                v_float32 cos_arg2 = vxr2 * v_cscale + v_psi;
                v_float32 cosine1 = v_cos(cos_arg1);
                v_float32 cosine2 = v_cos(cos_arg2);
                
                // Combine results
                v_float32 result1 = gaussian1 * cosine1 * v_scale;
                v_float32 result2 = gaussian2 * cosine2 * v_scale;
                
                // Store results (reversed order)
                v_store(ptr + (xmax - x - vecsize + 1), result1);
                v_store(ptr + (xmax - x - 2*vecsize + 1), result2);
            }
            
            // Process single vector
            for( ; x <= xmax - vecsize + 1; x += vecsize )
            {
                v_float32 vx = v_setall((float)x) + v_xinc;
                v_float32 vxr = vx * v_c + v_ys;
                v_float32 vyr = v_yc - vx * v_s;
                
                v_float32 gaussian_arg = vxr * vxr * v_ex + vyr * vyr * v_ey;
                v_float32 gaussian = v_exp(gaussian_arg);
                
                v_float32 cos_arg = vxr * v_cscale + v_psi;
                v_float32 cosine = v_cos(cos_arg);
                
                v_float32 result = gaussian * cosine * v_scale;
                v_store(ptr + (xmax - x - vecsize + 1), result);
            }
            
            // Process remaining elements
            for( ; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = y*c - x*s;
                float v = (float)(scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi));
                ptr[xmax - x] = v;
            }
        }
#else
        // Fallback to scalar implementation
        for( int y = ymin; y <= ymax; y++ )
            for( int x = xmin; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = -x*s + y*c;
                float v = (float)(scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi));
                kernel.at<float>(ymax - y, xmax - x) = v;
            }
#endif
    }
    else
    {
        // Double precision path with SIMD optimization
#if CV_SIMD_64F
        const int vecsize = v_float64::nlanes;
        int width = xmax - xmin + 1;
        
        // Pre-compute constants
        v_float64 v_c = v_setall(c);
        v_float64 v_s = v_setall(s);
        v_float64 v_ex = v_setall(ex);
        v_float64 v_ey = v_setall(ey);
        v_float64 v_cscale = v_setall(cscale);
        v_float64 v_psi = v_setall(psi);
        v_float64 v_scale = v_setall(scale);
        
        // Create x coordinate increment vector
        v_float64 v_xinc;
        for( int i = 0; i < vecsize; i++ )
            v_xinc.s[i] = (double)i;
        
        for( int y = ymin; y <= ymax; y++ )
        {
            double* ptr = kernel.ptr<double>(ymax - y);
            v_float64 v_ys = v_setall(y * s);
            v_float64 v_yc = v_setall(y * c);
            
            int x = xmin;
            
            // Process 2 vectors at a time for better ILP
            for( ; x <= xmax - 2*vecsize + 1; x += 2*vecsize )
            {
                // First vector
                v_float64 vx1 = v_setall((double)x) + v_xinc;
                v_float64 vxr1 = vx1 * v_c + v_ys;
                v_float64 vyr1 = v_yc - vx1 * v_s;
                
                // Second vector
                v_float64 vx2 = v_setall((double)(x + vecsize)) + v_xinc;
                v_float64 vxr2 = vx2 * v_c + v_ys;
                v_float64 vyr2 = v_yc - vx2 * v_s;
                
                // Compute Gaussian parts
                v_float64 gaussian_arg1 = vxr1 * vxr1 * v_ex + vyr1 * vyr1 * v_ey;
                v_float64 gaussian_arg2 = vxr2 * vxr2 * v_ex + vyr2 * vyr2 * v_ey;
                v_float64 gaussian1 = v_exp(gaussian_arg1);
                v_float64 gaussian2 = v_exp(gaussian_arg2);
                
                // Compute cosine parts
                v_float64 cos_arg1 = vxr1 * v_cscale + v_psi;
                v_float64 cos_arg2 = vxr2 * v_cscale + v_psi;
                v_float64 cosine1 = v_cos(cos_arg1);
                v_float64 cosine2 = v_cos(cos_arg2);
                
                // Combine results
                v_float64 result1 = gaussian1 * cosine1 * v_scale;
                v_float64 result2 = gaussian2 * cosine2 * v_scale;
                
                // Store results (reversed order)
                v_store(ptr + (xmax - x - vecsize + 1), result1);
                v_store(ptr + (xmax - x - 2*vecsize + 1), result2);
            }
            
            // Process single vector
            for( ; x <= xmax - vecsize + 1; x += vecsize )
            {
                v_float64 vx = v_setall((double)x) + v_xinc;
                v_float64 vxr = vx * v_c + v_ys;
                v_float64 vyr = v_yc - vx * v_s;
                
                v_float64 gaussian_arg = vxr * vxr * v_ex + vyr * vyr * v_ey;
                v_float64 gaussian = v_exp(gaussian_arg);
                
                v_float64 cos_arg = vxr * v_cscale + v_psi;
                v_float64 cosine = v_cos(cos_arg);
                
                v_float64 result = gaussian * cosine * v_scale;
                v_store(ptr + (xmax - x - vecsize + 1), result);
            }
            
            // Process remaining elements
            for( ; x <= xmax; x++ )
            {
                double xr = x*c + y*s;
                double yr = y*c - x*s;
                double v = scale*std::exp(ex*xr*xr + ey*yr*yr)*cos(cscale*xr + psi);
                ptr[xmax - x] = v;
            }
        }
#else
        // Fallback to scalar implementation
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

    return kernel;
}


/* End of file. */
