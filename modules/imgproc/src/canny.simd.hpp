// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void cannyGradientMagnitude(const short* dx, const short* dy, int* mag, int width, bool L2gradient);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

void cannyGradientMagnitude(const short* dx, const short* dy, int* mag, int width, bool L2gradient)
{
    int j = 0;
    
    if (L2gradient)
    {
        // L2 gradient: mag = dx^2 + dy^2
#if CV_SIMD && CV_SIMD_WIDTH >= 32
        // AVX2/AVX-512 path - process more elements at once
        const int step = VTraits<v_int16>::vlanes() * 2; // Process 2x vector width
        for (; j <= width - step; j += step)
        {
            v_int16 v_dx0 = vx_load(dx + j);
            v_int16 v_dy0 = vx_load(dy + j);
            v_int16 v_dx1 = vx_load(dx + j + VTraits<v_int16>::vlanes());
            v_int16 v_dy1 = vx_load(dy + j + VTraits<v_int16>::vlanes());
            
            // First set
            v_int32 v_dxp_low, v_dxp_high;
            v_int32 v_dyp_low, v_dyp_high;
            v_expand(v_dx0, v_dxp_low, v_dxp_high);
            v_expand(v_dy0, v_dyp_low, v_dyp_high);
            
            v_store(mag + j, v_add(v_mul(v_dxp_low, v_dxp_low), v_mul(v_dyp_low, v_dyp_low)));
            v_store(mag + j + VTraits<v_int32>::vlanes(), v_add(v_mul(v_dxp_high, v_dxp_high), v_mul(v_dyp_high, v_dyp_high)));
            
            // Second set
            v_expand(v_dx1, v_dxp_low, v_dxp_high);
            v_expand(v_dy1, v_dyp_low, v_dyp_high);
            
            v_store(mag + j + VTraits<v_int16>::vlanes(), v_add(v_mul(v_dxp_low, v_dxp_low), v_mul(v_dyp_low, v_dyp_low)));
            v_store(mag + j + VTraits<v_int16>::vlanes() + VTraits<v_int32>::vlanes(), v_add(v_mul(v_dxp_high, v_dxp_high), v_mul(v_dyp_high, v_dyp_high)));
        }
#elif CV_SIMD
        for (; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
        {
            v_int16 v_dx = vx_load(dx + j);
            v_int16 v_dy = vx_load(dy + j);
            
            v_int32 v_dxp_low, v_dxp_high;
            v_int32 v_dyp_low, v_dyp_high;
            v_expand(v_dx, v_dxp_low, v_dxp_high);
            v_expand(v_dy, v_dyp_low, v_dyp_high);
            
            v_store(mag + j, v_add(v_mul(v_dxp_low, v_dxp_low), v_mul(v_dyp_low, v_dyp_low)));
            v_store(mag + j + VTraits<v_int32>::vlanes(), v_add(v_mul(v_dxp_high, v_dxp_high), v_mul(v_dyp_high, v_dyp_high)));
        }
#endif
        for (; j < width; ++j)
            mag[j] = int(dx[j])*dx[j] + int(dy[j])*dy[j];
    }
    else
    {
        // L1 gradient: mag = |dx| + |dy|
#if CV_SIMD && CV_SIMD_WIDTH >= 32
        // AVX2/AVX-512 path
        const int step = VTraits<v_int16>::vlanes() * 2;
        for (; j <= width - step; j += step)
        {
            v_int16 v_dx0 = vx_load(dx + j);
            v_int16 v_dy0 = vx_load(dy + j);
            v_int16 v_dx1 = vx_load(dx + j + VTraits<v_int16>::vlanes());
            v_int16 v_dy1 = vx_load(dy + j + VTraits<v_int16>::vlanes());
            
            // Absolute values
            v_dx0 = v_abs(v_dx0);
            v_dy0 = v_abs(v_dy0);
            v_dx1 = v_abs(v_dx1);
            v_dy1 = v_abs(v_dy1);
            
            // First set
            v_int32 v_dx_ml, v_dy_ml, v_dx_mh, v_dy_mh;
            v_expand(v_dx0, v_dx_ml, v_dx_mh);
            v_expand(v_dy0, v_dy_ml, v_dy_mh);
            
            v_store(mag + j, v_add(v_dx_ml, v_dy_ml));
            v_store(mag + j + VTraits<v_int32>::vlanes(), v_add(v_dx_mh, v_dy_mh));
            
            // Second set
            v_expand(v_dx1, v_dx_ml, v_dx_mh);
            v_expand(v_dy1, v_dy_ml, v_dy_mh);
            
            v_store(mag + j + VTraits<v_int16>::vlanes(), v_add(v_dx_ml, v_dy_ml));
            v_store(mag + j + VTraits<v_int16>::vlanes() + VTraits<v_int32>::vlanes(), v_add(v_dx_mh, v_dy_mh));
        }
#elif CV_SIMD
        for (; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
        {
            v_int16 v_dx = vx_load(dx + j);
            v_int16 v_dy = vx_load(dy + j);
            
            v_dx = v_abs(v_dx);
            v_dy = v_abs(v_dy);
            
            v_int32 v_dx_ml, v_dy_ml, v_dx_mh, v_dy_mh;
            v_expand(v_dx, v_dx_ml, v_dx_mh);
            v_expand(v_dy, v_dy_ml, v_dy_mh);
            
            v_store(mag + j, v_add(v_dx_ml, v_dy_ml));
            v_store(mag + j + VTraits<v_int32>::vlanes(), v_add(v_dx_mh, v_dy_mh));
        }
#endif
        for (; j < width; ++j)
            mag[j] = std::abs(int(dx[j])) + std::abs(int(dy[j]));
    }
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv