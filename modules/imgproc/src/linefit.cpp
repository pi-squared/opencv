/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

namespace cv
{

static const double eps = 1e-6;

static void fitLine2D_wods( const Point2f* points, int count, float *weights, float *line )
{
    CV_Assert(count > 0);
    double x = 0, y = 0, x2 = 0, y2 = 0, xy = 0, w = 0;
    double dx2, dy2, dxy;
    int i;
    float t;

    // Calculating the average of x and y...
    if( weights == 0 )
    {
#if CV_SIMD
        // SIMD optimized path for unweighted case
        const int vlanes = v_float32::nlanes;
        v_float32 v_x = vx_setzero<v_float32>();
        v_float32 v_y = vx_setzero<v_float32>();
        v_float32 v_x2 = vx_setzero<v_float32>();
        v_float32 v_y2 = vx_setzero<v_float32>();
        v_float32 v_xy = vx_setzero<v_float32>();
        
        // Process multiple points at once
        for( i = 0; i <= count - vlanes; i += vlanes )
        {
            v_float32 px, py;
            // Load x and y coordinates
            v_load_deinterleave(reinterpret_cast<const float*>(&points[i]), px, py);
            
            // Accumulate sums
            v_x = v_add(v_x, px);
            v_y = v_add(v_y, py);
            v_x2 = v_fma(px, px, v_x2);
            v_y2 = v_fma(py, py, v_y2);
            v_xy = v_fma(px, py, v_xy);
        }
        
        // Reduce SIMD vectors to scalars
        x = v_reduce_sum(v_x);
        y = v_reduce_sum(v_y);
        x2 = v_reduce_sum(v_x2);
        y2 = v_reduce_sum(v_y2);
        xy = v_reduce_sum(v_xy);
        
        // Process remaining points
        for( ; i < count; i++ )
        {
            x += points[i].x;
            y += points[i].y;
            x2 += points[i].x * points[i].x;
            y2 += points[i].y * points[i].y;
            xy += points[i].x * points[i].y;
        }
#else
        for( i = 0; i < count; i += 1 )
        {
            x += points[i].x;
            y += points[i].y;
            x2 += points[i].x * points[i].x;
            y2 += points[i].y * points[i].y;
            xy += points[i].x * points[i].y;
        }
#endif
        w = (float) count;
    }
    else
    {
#if CV_SIMD
        // SIMD optimized path for weighted case
        const int vlanes = v_float32::nlanes;
        v_float32 v_x = vx_setzero<v_float32>();
        v_float32 v_y = vx_setzero<v_float32>();
        v_float32 v_x2 = vx_setzero<v_float32>();
        v_float32 v_y2 = vx_setzero<v_float32>();
        v_float32 v_xy = vx_setzero<v_float32>();
        v_float32 v_w = vx_setzero<v_float32>();
        
        // Process multiple points at once
        for( i = 0; i <= count - vlanes; i += vlanes )
        {
            v_float32 px, py;
            // Load x and y coordinates
            v_load_deinterleave(reinterpret_cast<const float*>(&points[i]), px, py);
            v_float32 vw = vx_load(&weights[i]);
            
            // Weighted accumulation
            v_float32 wpx = v_mul(vw, px);
            v_float32 wpy = v_mul(vw, py);
            
            v_x = v_add(v_x, wpx);
            v_y = v_add(v_y, wpy);
            v_x2 = v_fma(wpx, px, v_x2);
            v_y2 = v_fma(wpy, py, v_y2);
            v_xy = v_fma(wpx, py, v_xy);
            v_w = v_add(v_w, vw);
        }
        
        // Reduce SIMD vectors to scalars
        x = v_reduce_sum(v_x);
        y = v_reduce_sum(v_y);
        x2 = v_reduce_sum(v_x2);
        y2 = v_reduce_sum(v_y2);
        xy = v_reduce_sum(v_xy);
        w = v_reduce_sum(v_w);
        
        // Process remaining points
        for( ; i < count; i++ )
        {
            x += weights[i] * points[i].x;
            y += weights[i] * points[i].y;
            x2 += weights[i] * points[i].x * points[i].x;
            y2 += weights[i] * points[i].y * points[i].y;
            xy += weights[i] * points[i].x * points[i].y;
            w += weights[i];
        }
#else
        for( i = 0; i < count; i += 1 )
        {
            x += weights[i] * points[i].x;
            y += weights[i] * points[i].y;
            x2 += weights[i] * points[i].x * points[i].x;
            y2 += weights[i] * points[i].y * points[i].y;
            xy += weights[i] * points[i].x * points[i].y;
            w += weights[i];
        }
#endif
    }

    x /= w;
    y /= w;
    x2 /= w;
    y2 /= w;
    xy /= w;

    dx2 = x2 - x * x;
    dy2 = y2 - y * y;
    dxy = xy - x * y;

    t = (float) atan2( 2 * dxy, dx2 - dy2 ) / 2;
    line[0] = (float) cos( t );
    line[1] = (float) sin( t );

    line[2] = (float) x;
    line[3] = (float) y;
}

static void fitLine3D_wods( const Point3f * points, int count, float *weights, float *line )
{
    CV_Assert(count > 0);
    int i;
    float w0 = 0;
    float x0 = 0, y0 = 0, z0 = 0;
    float x2 = 0, y2 = 0, z2 = 0, xy = 0, yz = 0, xz = 0;
    float dx2, dy2, dz2, dxy, dxz, dyz;
    float *v;
    float n;
    float det[9], evc[9], evl[3];

    memset( evl, 0, 3*sizeof(evl[0]));
    memset( evc, 0, 9*sizeof(evl[0]));

    if( weights )
    {
#if CV_SIMD
        // SIMD optimized path for weighted case
        const int vlanes = v_float32::nlanes;
        v_float32 v_x0 = vx_setzero<v_float32>();
        v_float32 v_y0 = vx_setzero<v_float32>();
        v_float32 v_z0 = vx_setzero<v_float32>();
        v_float32 v_x2 = vx_setzero<v_float32>();
        v_float32 v_y2 = vx_setzero<v_float32>();
        v_float32 v_z2 = vx_setzero<v_float32>();
        v_float32 v_xy = vx_setzero<v_float32>();
        v_float32 v_yz = vx_setzero<v_float32>();
        v_float32 v_xz = vx_setzero<v_float32>();
        v_float32 v_w0 = vx_setzero<v_float32>();
        
        // Process multiple points at once
        for( i = 0; i <= count - vlanes; i += vlanes )
        {
            v_float32 px, py, pz;
            // Load x, y, z coordinates
            v_load_deinterleave(reinterpret_cast<const float*>(&points[i]), px, py, pz);
            v_float32 vw = vx_load(&weights[i]);
            
            // Weighted values
            v_float32 wpx = v_mul(vw, px);
            v_float32 wpy = v_mul(vw, py);
            v_float32 wpz = v_mul(vw, pz);
            
            // Accumulate moments
            v_x0 = v_add(v_x0, wpx);
            v_y0 = v_add(v_y0, wpy);
            v_z0 = v_add(v_z0, wpz);
            
            v_x2 = v_fma(wpx, px, v_x2);
            v_y2 = v_fma(wpy, py, v_y2);
            v_z2 = v_fma(wpz, pz, v_z2);
            
            v_xy = v_fma(wpx, py, v_xy);
            v_yz = v_fma(wpy, pz, v_yz);
            v_xz = v_fma(wpx, pz, v_xz);
            
            v_w0 = v_add(v_w0, vw);
        }
        
        // Reduce SIMD vectors to scalars
        x0 = v_reduce_sum(v_x0);
        y0 = v_reduce_sum(v_y0);
        z0 = v_reduce_sum(v_z0);
        x2 = v_reduce_sum(v_x2);
        y2 = v_reduce_sum(v_y2);
        z2 = v_reduce_sum(v_z2);
        xy = v_reduce_sum(v_xy);
        yz = v_reduce_sum(v_yz);
        xz = v_reduce_sum(v_xz);
        w0 = v_reduce_sum(v_w0);
        
        // Process remaining points
        for( ; i < count; i++ )
        {
            float x = points[i].x;
            float y = points[i].y;
            float z = points[i].z;
            float w = weights[i];

            x2 += x * x * w;
            xy += x * y * w;
            xz += x * z * w;
            y2 += y * y * w;
            yz += y * z * w;
            z2 += z * z * w;
            x0 += x * w;
            y0 += y * w;
            z0 += z * w;
            w0 += w;
        }
#else
        for( i = 0; i < count; i++ )
        {
            float x = points[i].x;
            float y = points[i].y;
            float z = points[i].z;
            float w = weights[i];


            x2 += x * x * w;
            xy += x * y * w;
            xz += x * z * w;
            y2 += y * y * w;
            yz += y * z * w;
            z2 += z * z * w;
            x0 += x * w;
            y0 += y * w;
            z0 += z * w;
            w0 += w;
        }
#endif
    }
    else
    {
#if CV_SIMD
        // SIMD optimized path for unweighted case
        const int vlanes = v_float32::nlanes;
        v_float32 v_x0 = vx_setzero<v_float32>();
        v_float32 v_y0 = vx_setzero<v_float32>();
        v_float32 v_z0 = vx_setzero<v_float32>();
        v_float32 v_x2 = vx_setzero<v_float32>();
        v_float32 v_y2 = vx_setzero<v_float32>();
        v_float32 v_z2 = vx_setzero<v_float32>();
        v_float32 v_xy = vx_setzero<v_float32>();
        v_float32 v_yz = vx_setzero<v_float32>();
        v_float32 v_xz = vx_setzero<v_float32>();
        
        // Process multiple points at once
        for( i = 0; i <= count - vlanes; i += vlanes )
        {
            v_float32 px, py, pz;
            // Load x, y, z coordinates
            v_load_deinterleave(reinterpret_cast<const float*>(&points[i]), px, py, pz);
            
            // Accumulate moments
            v_x0 = v_add(v_x0, px);
            v_y0 = v_add(v_y0, py);
            v_z0 = v_add(v_z0, pz);
            
            v_x2 = v_fma(px, px, v_x2);
            v_y2 = v_fma(py, py, v_y2);
            v_z2 = v_fma(pz, pz, v_z2);
            
            v_xy = v_fma(px, py, v_xy);
            v_yz = v_fma(py, pz, v_yz);
            v_xz = v_fma(px, pz, v_xz);
        }
        
        // Reduce SIMD vectors to scalars
        x0 = v_reduce_sum(v_x0);
        y0 = v_reduce_sum(v_y0);
        z0 = v_reduce_sum(v_z0);
        x2 = v_reduce_sum(v_x2);
        y2 = v_reduce_sum(v_y2);
        z2 = v_reduce_sum(v_z2);
        xy = v_reduce_sum(v_xy);
        yz = v_reduce_sum(v_yz);
        xz = v_reduce_sum(v_xz);
        
        // Process remaining points
        for( ; i < count; i++ )
        {
            float x = points[i].x;
            float y = points[i].y;
            float z = points[i].z;

            x2 += x * x;
            xy += x * y;
            xz += x * z;
            y2 += y * y;
            yz += y * z;
            z2 += z * z;
            x0 += x;
            y0 += y;
            z0 += z;
        }
#else
        for( i = 0; i < count; i++ )
        {
            float x = points[i].x;
            float y = points[i].y;
            float z = points[i].z;

            x2 += x * x;
            xy += x * y;
            xz += x * z;
            y2 += y * y;
            yz += y * z;
            z2 += z * z;
            x0 += x;
            y0 += y;
            z0 += z;
        }
#endif
        w0 = (float) count;
    }

    x2 /= w0;
    xy /= w0;
    xz /= w0;
    y2 /= w0;
    yz /= w0;
    z2 /= w0;

    x0 /= w0;
    y0 /= w0;
    z0 /= w0;

    dx2 = x2 - x0 * x0;
    dxy = xy - x0 * y0;
    dxz = xz - x0 * z0;
    dy2 = y2 - y0 * y0;
    dyz = yz - y0 * z0;
    dz2 = z2 - z0 * z0;

    det[0] = dz2 + dy2;
    det[1] = -dxy;
    det[2] = -dxz;
    det[3] = det[1];
    det[4] = dx2 + dz2;
    det[5] = -dyz;
    det[6] = det[2];
    det[7] = det[5];
    det[8] = dy2 + dx2;

    // Searching for a eigenvector of det corresponding to the minimal eigenvalue
    Mat _det( 3, 3, CV_32F, det );
    Mat _evc( 3, 3, CV_32F, evc );
    Mat _evl( 3, 1, CV_32F, evl );
    eigen( _det, _evl, _evc );
    i = evl[0] < evl[1] ? (evl[0] < evl[2] ? 0 : 2) : (evl[1] < evl[2] ? 1 : 2);

    v = &evc[i * 3];
    n = (float) std::sqrt( (double)v[0] * v[0] + (double)v[1] * v[1] + (double)v[2] * v[2] );
    n = (float)MAX(n, eps);
    line[0] = v[0] / n;
    line[1] = v[1] / n;
    line[2] = v[2] / n;
    line[3] = x0;
    line[4] = y0;
    line[5] = z0;
}

static double calcDist2D( const Point2f* points, int count, float *_line, float *dist )
{
    int j;
    float px = _line[2], py = _line[3];
    float nx = _line[1], ny = -_line[0];
    double sum_dist = 0.;

#if CV_SIMD
    const int vlanes = v_float32::nlanes;
    v_float32 v_px = vx_setall<v_float32>(px);
    v_float32 v_py = vx_setall<v_float32>(py);
    v_float32 v_nx = vx_setall<v_float32>(nx);
    v_float32 v_ny = vx_setall<v_float32>(ny);
    v_float32 v_sum = vx_setzero<v_float32>();
    
    // Process multiple points at once
    for( j = 0; j <= count - vlanes; j += vlanes )
    {
        v_float32 vx, vy;
        // Load x and y coordinates
        v_load_deinterleave(reinterpret_cast<const float*>(&points[j]), vx, vy);
        
        // Calculate distances
        vx = v_sub(vx, v_px);
        vy = v_sub(vy, v_py);
        
        v_float32 d = v_fma(v_nx, vx, v_mul(v_ny, vy));
        d = v_abs(d);
        
        // Store distances
        vx_store(&dist[j], d);
        
        // Accumulate sum
        v_sum = v_add(v_sum, d);
    }
    
    sum_dist = v_reduce_sum(v_sum);
    
    // Process remaining points
    for( ; j < count; j++ )
    {
        float x, y;

        x = points[j].x - px;
        y = points[j].y - py;

        dist[j] = (float) fabs( nx * x + ny * y );
        sum_dist += dist[j];
    }
#else
    for( j = 0; j < count; j++ )
    {
        float x, y;

        x = points[j].x - px;
        y = points[j].y - py;

        dist[j] = (float) fabs( nx * x + ny * y );
        sum_dist += dist[j];
    }
#endif

    return sum_dist;
}

static double calcDist3D( const Point3f* points, int count, float *_line, float *dist )
{
    int j;
    float px = _line[3], py = _line[4], pz = _line[5];
    float vx = _line[0], vy = _line[1], vz = _line[2];
    double sum_dist = 0.;

#if CV_SIMD
    const int vlanes = v_float32::nlanes;
    v_float32 v_px = vx_setall<v_float32>(px);
    v_float32 v_py = vx_setall<v_float32>(py);
    v_float32 v_pz = vx_setall<v_float32>(pz);
    v_float32 v_vx = vx_setall<v_float32>(vx);
    v_float32 v_vy = vx_setall<v_float32>(vy);
    v_float32 v_vz = vx_setall<v_float32>(vz);
    v_float32 v_sum = vx_setzero<v_float32>();
    
    // Process multiple points at once
    for( j = 0; j <= count - vlanes; j += vlanes )
    {
        v_float32 x, y, z;
        // Load x, y, z coordinates
        v_load_deinterleave(reinterpret_cast<const float*>(&points[j]), x, y, z);
        
        // Calculate relative position
        x = v_sub(x, v_px);
        y = v_sub(y, v_py);
        z = v_sub(z, v_pz);
        
        // Cross product components
        v_float32 p1 = v_sub(v_mul(v_vy, z), v_mul(v_vz, y));
        v_float32 p2 = v_sub(v_mul(v_vz, x), v_mul(v_vx, z));
        v_float32 p3 = v_sub(v_mul(v_vx, y), v_mul(v_vy, x));
        
        // Distance = sqrt(p1^2 + p2^2 + p3^2)
        v_float32 d2 = v_fma(p1, p1, v_fma(p2, p2, v_mul(p3, p3)));
        v_float32 d = v_sqrt(d2);
        
        // Store distances
        vx_store(&dist[j], d);
        
        // Accumulate sum
        v_sum = v_add(v_sum, d);
    }
    
    sum_dist = v_reduce_sum(v_sum);
    
    // Process remaining points
    for( ; j < count; j++ )
    {
        float x, y, z;
        double p1, p2, p3;

        x = points[j].x - px;
        y = points[j].y - py;
        z = points[j].z - pz;

        p1 = vy * z - vz * y;
        p2 = vz * x - vx * z;
        p3 = vx * y - vy * x;

        dist[j] = (float) std::sqrt( p1*p1 + p2*p2 + p3*p3 );
        sum_dist += dist[j];
    }
#else
    for( j = 0; j < count; j++ )
    {
        float x, y, z;
        double p1, p2, p3;

        x = points[j].x - px;
        y = points[j].y - py;
        z = points[j].z - pz;

        p1 = vy * z - vz * y;
        p2 = vz * x - vx * z;
        p3 = vx * y - vy * x;

        dist[j] = (float) std::sqrt( p1*p1 + p2*p2 + p3*p3 );
        sum_dist += dist[j];
    }
#endif

    return sum_dist;
}

static void weightL1( float *d, int count, float *w )
{
    int i;

    for( i = 0; i < count; i++ )
    {
        double t = fabs( (double) d[i] );
        w[i] = (float)(1. / MAX(t, eps));
    }
}

static void weightL12( float *d, int count, float *w )
{
    int i;

    for( i = 0; i < count; i++ )
    {
        w[i] = 1.0f / (float) std::sqrt( 1 + (double) (d[i] * d[i] * 0.5) );
    }
}


static void weightHuber( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c <= 0 ? 1.345f : _c;

    for( i = 0; i < count; i++ )
    {
        if( d[i] < c )
            w[i] = 1.0f;
        else
            w[i] = c/d[i];
    }
}


static void weightFair( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c == 0 ? 1 / 1.3998f : 1 / _c;

    for( i = 0; i < count; i++ )
    {
        w[i] = 1 / (1 + d[i] * c);
    }
}

static void weightWelsch( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c == 0 ? 1 / 2.9846f : 1 / _c;

    for( i = 0; i < count; i++ )
    {
        w[i] = (float) std::exp( -d[i] * d[i] * c * c );
    }
}


/* Takes an array of 2D points, type of distance (including user-defined
 distance specified by callbacks, fills the array of four floats with line
 parameters A, B, C, D, where (A, B) is the normalized direction vector,
 (C, D) is the point that belongs to the line. */

static void fitLine2D( const Point2f * points, int count, int dist,
                      float _param, float reps, float aeps, float *line )
{
    double EPS = count*FLT_EPSILON;
    void (*calc_weights) (float *, int, float *) = 0;
    void (*calc_weights_param) (float *, int, float *, float) = 0;
    int i, j, k;
    float _line[4], _lineprev[4];
    float rdelta = reps != 0 ? reps : 1.0f;
    float adelta = aeps != 0 ? aeps : 0.01f;
    double min_err = DBL_MAX, err = 0;
    RNG rng((uint64)-1);

    memset( line, 0, 4*sizeof(line[0]) );

    switch (dist)
    {
    case cv::DIST_L2:
        return fitLine2D_wods( points, count, 0, line );

    case cv::DIST_L1:
        calc_weights = weightL1;
        break;

    case cv::DIST_L12:
        calc_weights = weightL12;
        break;

    case cv::DIST_FAIR:
        calc_weights_param = weightFair;
        break;

    case cv::DIST_WELSCH:
        calc_weights_param = weightWelsch;
        break;

    case cv::DIST_HUBER:
        calc_weights_param = weightHuber;
        break;

    /*case DIST_USER:
     calc_weights = (void ( * )(float *, int, float *)) _PFP.fp;
     break;*/
    default:
        CV_Error(cv::Error::StsBadArg, "Unknown distance type");
    }

    AutoBuffer<float> wr(count*2);
    float *w = wr.data(), *r = w + count;

    for( k = 0; k < 20; k++ )
    {
        int first = 1;
        for( i = 0; i < count; i++ )
            w[i] = 0.f;

        for( i = 0; i < MIN(count,10); )
        {
            j = rng.uniform(0, count);
            if( w[j] < FLT_EPSILON )
            {
                w[j] = 1.f;
                i++;
            }
        }

        fitLine2D_wods( points, count, w, _line );
        for( i = 0; i < 30; i++ )
        {
            double sum_w = 0;

            if( first )
            {
                first = 0;
            }
            else
            {
                double t = _line[0] * _lineprev[0] + _line[1] * _lineprev[1];
                t = MAX(t,-1.);
                t = MIN(t,1.);
                if( fabs(acos(t)) < adelta )
                {
                    float x, y, d;

                    x = (float) fabs( _line[2] - _lineprev[2] );
                    y = (float) fabs( _line[3] - _lineprev[3] );

                    d = x > y ? x : y;
                    if( d < rdelta )
                        break;
                }
            }
            /* calculate distances */
            err = calcDist2D( points, count, _line, r );

            if (err < min_err)
            {
                min_err = err;
                memcpy(line, _line, 4 * sizeof(line[0]));
                if (err < EPS)
                    break;
            }

            /* calculate weights */
            if( calc_weights )
                calc_weights( r, count, w );
            else
                calc_weights_param( r, count, w, _param );

            for( j = 0; j < count; j++ )
                sum_w += w[j];

            if( fabs(sum_w) > FLT_EPSILON )
            {
                sum_w = 1./sum_w;
                for( j = 0; j < count; j++ )
                    w[j] = (float)(w[j]*sum_w);
            }
            else
            {
                for( j = 0; j < count; j++ )
                    w[j] = 1.f;
            }

            /* save the line parameters */
            memcpy( _lineprev, _line, 4 * sizeof( float ));

            /* Run again... */
            fitLine2D_wods( points, count, w, _line );
        }

        if( err < min_err )
        {
            min_err = err;
            memcpy( line, _line, 4 * sizeof(line[0]));
            if( err < EPS )
                break;
        }
    }
}


/* Takes an array of 3D points, type of distance (including user-defined
 distance specified by callbacks, fills the array of four floats with line
 parameters A, B, C, D, E, F, where (A, B, C) is the normalized direction vector,
 (D, E, F) is the point that belongs to the line. */
static void fitLine3D( Point3f * points, int count, int dist,
                       float _param, float reps, float aeps, float *line )
{
    double EPS = count*FLT_EPSILON;
    void (*calc_weights) (float *, int, float *) = 0;
    void (*calc_weights_param) (float *, int, float *, float) = 0;
    int i, j, k;
    float _line[6]={0,0,0,0,0,0}, _lineprev[6]={0,0,0,0,0,0};
    float rdelta = reps != 0 ? reps : 1.0f;
    float adelta = aeps != 0 ? aeps : 0.01f;
    double min_err = DBL_MAX, err = 0;
    RNG rng((uint64)-1);

    switch (dist)
    {
    case cv::DIST_L2:
        return fitLine3D_wods( points, count, 0, line );

    case cv::DIST_L1:
        calc_weights = weightL1;
        break;

    case cv::DIST_L12:
        calc_weights = weightL12;
        break;

    case cv::DIST_FAIR:
        calc_weights_param = weightFair;
        break;

    case cv::DIST_WELSCH:
        calc_weights_param = weightWelsch;
        break;

    case cv::DIST_HUBER:
        calc_weights_param = weightHuber;
        break;

    default:
        CV_Error(cv::Error::StsBadArg, "Unknown distance");
    }

    AutoBuffer<float> buf(count*2);
    float *w = buf.data(), *r = w + count;

    for( k = 0; k < 20; k++ )
    {
        int first = 1;
        for( i = 0; i < count; i++ )
            w[i] = 0.f;

        for( i = 0; i < MIN(count,10); )
        {
            j = rng.uniform(0, count);
            if( w[j] < FLT_EPSILON )
            {
                w[j] = 1.f;
                i++;
            }
        }

        fitLine3D_wods( points, count, w, _line );
        for( i = 0; i < 30; i++ )
        {
            double sum_w = 0;

            if( first )
            {
                first = 0;
            }
            else
            {
                double t = _line[0] * _lineprev[0] + _line[1] * _lineprev[1] + _line[2] * _lineprev[2];
                t = MAX(t,-1.);
                t = MIN(t,1.);
                if( fabs(acos(t)) < adelta )
                {
                    float x, y, z, ax, ay, az, dx, dy, dz, d;

                    x = _line[3] - _lineprev[3];
                    y = _line[4] - _lineprev[4];
                    z = _line[5] - _lineprev[5];
                    ax = _line[0] - _lineprev[0];
                    ay = _line[1] - _lineprev[1];
                    az = _line[2] - _lineprev[2];
                    dx = (float) fabs( y * az - z * ay );
                    dy = (float) fabs( z * ax - x * az );
                    dz = (float) fabs( x * ay - y * ax );

                    d = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
                    if( d < rdelta )
                        break;
                }
            }
            /* calculate distances */
            err = calcDist3D( points, count, _line, r );
            if (err < min_err)
            {
                min_err = err;
                memcpy(line, _line, 6 * sizeof(line[0]));
                if (err < EPS)
                    break;
            }

            /* calculate weights */
            if( calc_weights )
                calc_weights( r, count, w );
            else
                calc_weights_param( r, count, w, _param );

            for( j = 0; j < count; j++ )
                sum_w += w[j];

            if( fabs(sum_w) > FLT_EPSILON )
            {
                sum_w = 1./sum_w;
                for( j = 0; j < count; j++ )
                    w[j] = (float)(w[j]*sum_w);
            }
            else
            {
                for( j = 0; j < count; j++ )
                    w[j] = 1.f;
            }

            /* save the line parameters */
            memcpy( _lineprev, _line, 6 * sizeof( float ));

            /* Run again... */
            fitLine3D_wods( points, count, w, _line );
        }

        if( err < min_err )
        {
            min_err = err;
            memcpy( line, _line, 6 * sizeof(line[0]));
            if( err < EPS )
                break;
        }
    }
}

}

void cv::fitLine( InputArray _points, OutputArray _line, int distType,
                 double param, double reps, double aeps )
{
    CV_INSTRUMENT_REGION();

    Mat points = _points.getMat();

    float linebuf[6]={0.f};
    int npoints2 = points.checkVector(2, -1, false);
    int npoints3 = points.checkVector(3, -1, false);

    CV_Assert( npoints2 >= 0 || npoints3 >= 0 );

    if( points.depth() != CV_32F || !points.isContinuous() )
    {
        Mat temp;
        points.convertTo(temp, CV_32F);
        points = temp;
    }

    if( npoints2 >= 0 )
        fitLine2D( points.ptr<Point2f>(), npoints2, distType,
                   (float)param, (float)reps, (float)aeps, linebuf);
    else
        fitLine3D( points.ptr<Point3f>(), npoints3, distType,
                   (float)param, (float)reps, (float)aeps, linebuf);

    Mat(npoints2 >= 0 ? 4 : 6, 1, CV_32F, linebuf).copyTo(_line);
}


CV_IMPL void
cvFitLine( const CvArr* array, int dist, double param,
           double reps, double aeps, float *line )
{
    CV_Assert(line != 0);

    cv::AutoBuffer<double> buf;
    cv::Mat points = cv::cvarrToMat(array, false, false, 0, &buf);
    cv::Mat linemat(points.checkVector(2) >= 0 ? 4 : 6, 1, CV_32F, line);

    cv::fitLine(points, linemat, dist, param, reps, aeps);
}

/* End of file. */
