// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#if CV_AVX512_SKX

#include <immintrin.h>

namespace cv {

// AVX-512 optimized gradient magnitude calculation for Canny edge detection
void cannyGradientMagnitude_AVX512(const short* dx, const short* dy, int* mag, int width, bool L2gradient)
{
    const int step = 32; // Process 32 int16 values at once
    int j = 0;
    
    if (L2gradient)
    {
        // L2 gradient: mag = dx^2 + dy^2
        for (; j <= width - step; j += step)
        {
            // Load 32 int16 values
            __m512i v_dx = _mm512_loadu_si512((const __m512i*)(dx + j));
            __m512i v_dy = _mm512_loadu_si512((const __m512i*)(dy + j));
            
            // Convert lower 16 values to int32 and square
            __m512i v_dx_lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_dx, 0));
            __m512i v_dy_lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_dy, 0));
            __m512i v_mag_lo = _mm512_add_epi32(_mm512_mullo_epi32(v_dx_lo, v_dx_lo), 
                                                 _mm512_mullo_epi32(v_dy_lo, v_dy_lo));
            
            // Convert upper 16 values to int32 and square
            __m512i v_dx_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_dx, 1));
            __m512i v_dy_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_dy, 1));
            __m512i v_mag_hi = _mm512_add_epi32(_mm512_mullo_epi32(v_dx_hi, v_dx_hi), 
                                                 _mm512_mullo_epi32(v_dy_hi, v_dy_hi));
            
            // Store results
            _mm512_storeu_si512((__m512i*)(mag + j), v_mag_lo);
            _mm512_storeu_si512((__m512i*)(mag + j + 16), v_mag_hi);
        }
    }
    else
    {
        // L1 gradient: mag = |dx| + |dy|
        for (; j <= width - step; j += step)
        {
            // Load 32 int16 values
            __m512i v_dx = _mm512_loadu_si512((const __m512i*)(dx + j));
            __m512i v_dy = _mm512_loadu_si512((const __m512i*)(dy + j));
            
            // Absolute values
            v_dx = _mm512_abs_epi16(v_dx);
            v_dy = _mm512_abs_epi16(v_dy);
            
            // Convert lower 16 values to int32 and add
            __m512i v_dx_lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v_dx, 0));
            __m512i v_dy_lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v_dy, 0));
            __m512i v_mag_lo = _mm512_add_epi32(v_dx_lo, v_dy_lo);
            
            // Convert upper 16 values to int32 and add
            __m512i v_dx_hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v_dx, 1));
            __m512i v_dy_hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v_dy, 1));
            __m512i v_mag_hi = _mm512_add_epi32(v_dx_hi, v_dy_hi);
            
            // Store results
            _mm512_storeu_si512((__m512i*)(mag + j), v_mag_lo);
            _mm512_storeu_si512((__m512i*)(mag + j + 16), v_mag_hi);
        }
    }
    
    // Process remaining elements
    for (; j < width; ++j)
    {
        if (L2gradient)
            mag[j] = int(dx[j])*dx[j] + int(dy[j])*dy[j];
        else
            mag[j] = std::abs(int(dx[j])) + std::abs(int(dy[j]));
    }
}

} // namespace cv

#endif // CV_AVX512_SKX