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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

void cv::cornerSubPix( InputArray _image, InputOutputArray _corners,
                       Size win, Size zeroZone, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    const int MAX_ITERS = 100;
    int win_w = win.width * 2 + 1, win_h = win.height * 2 + 1;
    int i, j, k;
    int max_iters = (criteria.type & TermCriteria::MAX_ITER) ? MIN(MAX(criteria.maxCount, 1), MAX_ITERS) : MAX_ITERS;
    double eps = (criteria.type & TermCriteria::EPS) ? MAX(criteria.epsilon, 0.) : 0;
    eps *= eps; // use square of error in comparison operations

    cv::Mat src = _image.getMat(), cornersmat = _corners.getMat();
    int count = cornersmat.checkVector(2, CV_32F);
    CV_Assert( count >= 0 );
    Point2f* corners = cornersmat.ptr<Point2f>();

    if( count == 0 )
        return;

    CV_Assert( win.width > 0 && win.height > 0 );
    CV_Assert( src.cols >= win.width*2 + 5 && src.rows >= win.height*2 + 5 );
    CV_Assert( src.channels() == 1 );

    Mat maskm(win_h, win_w, CV_32F), subpix_buf(win_h+2, win_w+2, CV_32F);
    float* mask = maskm.ptr<float>();

    for( i = 0; i < win_h; i++ )
    {
        float y = (float)(i - win.height)/win.height;
        float vy = std::exp(-y*y);
        for( j = 0; j < win_w; j++ )
        {
            float x = (float)(j - win.width)/win.width;
            mask[i * win_w + j] = (float)(vy*std::exp(-x*x));
        }
    }

    // make zero_zone
    if( zeroZone.width >= 0 && zeroZone.height >= 0 &&
        zeroZone.width * 2 + 1 < win_w && zeroZone.height * 2 + 1 < win_h )
    {
        for( i = win.height - zeroZone.height; i <= win.height + zeroZone.height; i++ )
        {
            for( j = win.width - zeroZone.width; j <= win.width + zeroZone.width; j++ )
            {
                mask[i * win_w + j] = 0;
            }
        }
    }

    // do optimization loop for all the points
    for( int pt_i = 0; pt_i < count; pt_i++ )
    {
        Point2f cT = corners[pt_i], cI = cT;
        CV_Assert( Rect(0, 0, src.cols, src.rows).contains(cT) );
        int iter = 0;
        double err = 0;

        do
        {
            Point2f cI2;
            double a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;

            getRectSubPix(src, Size(win_w+2, win_h+2), cI, subpix_buf, subpix_buf.type());
            const float* subpix = &subpix_buf.at<float>(1,1);

            // process gradient
            for( i = 0, k = 0; i < win_h; i++, subpix += win_w + 2 )
            {
                double py = i - win.height;

#if CV_SIMD
                v_float32 v_a = vx_setzero_f32();
                v_float32 v_b = vx_setzero_f32();
                v_float32 v_c = vx_setzero_f32();
                v_float32 v_bb1 = vx_setzero_f32();
                v_float32 v_bb2 = vx_setzero_f32();
                v_float32 v_py = vx_setall_f32((float)py);

                j = 0;
                const int vstep = VTraits<v_float32>::vlanes();
                
#if CV_AVX512_SKX && CV_SIMD_WIDTH == 64
                // AVX-512 specific optimization - process 16 values at once
                // Prefetch ahead for better cache utilization
                if (win_w >= 32) {
                    const float* prefetch_ptr = subpix + 16;
                    _mm_prefetch((const char*)prefetch_ptr, _MM_HINT_T0);
                    _mm_prefetch((const char*)(prefetch_ptr + win_w + 2), _MM_HINT_T0);
                }
#endif
                
                for( ; j <= win_w - vstep; j += vstep, k += vstep )
                {
                    // Load mask values
                    v_float32 v_mask = vx_load(mask + k);
                    
                    // Calculate gradients
                    v_float32 v_tgx = v_sub(vx_load(subpix + j + 1), vx_load(subpix + j - 1));
                    v_float32 v_tgy = v_sub(vx_load(subpix + j + win_w + 2), vx_load(subpix + j - win_w - 2));
                    
                    // Calculate gxx, gxy, gyy
                    v_float32 v_gxx = v_mul(v_mul(v_tgx, v_tgx), v_mask);
                    v_float32 v_gxy = v_mul(v_mul(v_tgx, v_tgy), v_mask);
                    v_float32 v_gyy = v_mul(v_mul(v_tgy, v_tgy), v_mask);
                    
                    // Calculate px values
                    float px_base = (float)(j - win.width);
                    float px_vals[VTraits<v_float32>::max_nlanes];
                    for (int idx = 0; idx < vstep; idx++)
                        px_vals[idx] = px_base + idx;
                    v_float32 v_px = vx_load(px_vals);
                    
                    // Accumulate matrix elements
                    v_a = v_add(v_a, v_gxx);
                    v_b = v_add(v_b, v_gxy);
                    v_c = v_add(v_c, v_gyy);
                    
                    // Calculate bb1 and bb2 using FMA if available
#if CV_FMA3
                    v_bb1 = v_fma(v_gxx, v_px, v_bb1);
                    v_bb1 = v_fma(v_gxy, v_py, v_bb1);
                    v_bb2 = v_fma(v_gxy, v_px, v_bb2);
                    v_bb2 = v_fma(v_gyy, v_py, v_bb2);
#else
                    v_bb1 = v_add(v_bb1, v_add(v_mul(v_gxx, v_px), v_mul(v_gxy, v_py)));
                    v_bb2 = v_add(v_bb2, v_add(v_mul(v_gxy, v_px), v_mul(v_gyy, v_py)));
#endif
                    
#if CV_AVX512_SKX && CV_SIMD_WIDTH == 64
                    // Prefetch for next iteration
                    if (j + vstep < win_w - vstep) {
                        _mm_prefetch((const char*)(subpix + j + vstep + 16), _MM_HINT_T0);
                        _mm_prefetch((const char*)(subpix + j + vstep + 16 + win_w + 2), _MM_HINT_T0);
                    }
#endif
                }
                
                // Reduce SIMD accumulators
                a += v_reduce_sum(v_a);
                b += v_reduce_sum(v_b);
                c += v_reduce_sum(v_c);
                bb1 += v_reduce_sum(v_bb1);
                bb2 += v_reduce_sum(v_bb2);
#else
                j = 0;
#endif
                // Process remaining elements
                for( ; j < win_w; j++, k++ )
                {
                    double m = mask[k];
                    double tgx = subpix[j+1] - subpix[j-1];
                    double tgy = subpix[j+win_w+2] - subpix[j-win_w-2];
                    double gxx = tgx * tgx * m;
                    double gxy = tgx * tgy * m;
                    double gyy = tgy * tgy * m;
                    double px = j - win.width;

                    a += gxx;
                    b += gxy;
                    c += gyy;

                    bb1 += gxx * px + gxy * py;
                    bb2 += gxy * px + gyy * py;
                }
            }

            double det=a*c-b*b;
            if( fabs( det ) <= DBL_EPSILON*DBL_EPSILON )
                break;

            // 2x2 matrix inversion
            double scale=1.0/det;
            cI2.x = (float)(cI.x + c*scale*bb1 - b*scale*bb2);
            cI2.y = (float)(cI.y - b*scale*bb1 + a*scale*bb2);
            err = (cI2.x - cI.x) * (cI2.x - cI.x) + (cI2.y - cI.y) * (cI2.y - cI.y);
            // if new point is out of image, leave previous point as the result
            if( !Rect(0, 0, src.cols, src.rows).contains(cI2) )
                break;
            cI = cI2;
        }
        while( ++iter < max_iters && err > eps );

        // if new point is too far from initial, it means poor convergence.
        // leave initial point as the result
        if( fabs( cI.x - cT.x ) > win.width || fabs( cI.y - cT.y ) > win.height )
            cI = cT;

        corners[pt_i] = cI;
    }
}


CV_IMPL void
cvFindCornerSubPix( const void* srcarr, CvPoint2D32f* _corners,
                   int count, CvSize win, CvSize zeroZone,
                   CvTermCriteria criteria )
{
    if(!_corners || count <= 0)
        return;

    cv::Mat src = cv::cvarrToMat(srcarr), corners(count, 1, CV_32FC2, _corners);
    cv::cornerSubPix(src, corners, win, zeroZone, criteria);
}

/* End of file. */
