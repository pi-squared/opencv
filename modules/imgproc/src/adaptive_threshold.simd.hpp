// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGPROC_ADAPTIVE_THRESHOLD_SIMD_HPP
#define OPENCV_IMGPROC_ADAPTIVE_THRESHOLD_SIMD_HPP

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace hal {

// SIMD optimized adaptive threshold implementation
static void adaptiveThresholdSIMD(const uchar* src_data, size_t src_step,
                                   const uchar* mean_data, size_t mean_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height,
                                   uchar maxval, int idelta, int type)
{
    CV_INSTRUMENT_REGION();
    
    // Create lookup table as per original implementation
    uchar tab[768];
    if (type == THRESH_BINARY)
    {
        for (int i = 0; i < 768; i++)
            tab[i] = (uchar)(i - 255 > -idelta ? maxval : 0);
    }
    else // THRESH_BINARY_INV
    {
        for (int i = 0; i < 768; i++)
            tab[i] = (uchar)(i - 255 <= -idelta ? maxval : 0);
    }
    
#if CV_SIMD
    const int VECSZ = v_uint8::nlanes;
    
    // Process continuous data if possible
    if (src_step == width && mean_step == width && dst_step == width)
    {
        width *= height;
        height = 1;
    }
    
    for (int y = 0; y < height; y++)
    {
        const uchar* srow = src_data + y * src_step;
        const uchar* mrow = mean_data + y * mean_step;
        uchar* drow = dst_data + y * dst_step;
        
        int x = 0;
        
        // Main SIMD loop with 4x unrolling
        for (; x <= width - 4*VECSZ; x += 4*VECSZ)
        {
            // Prefetch next cache line
            CV_PREFETCH_READ(srow + x + 4*VECSZ);
            CV_PREFETCH_READ(mrow + x + 4*VECSZ);
            
            // Process 4 vectors at once
            v_uint8 s0 = vx_load(srow + x);
            v_uint8 s1 = vx_load(srow + x + VECSZ);
            v_uint8 s2 = vx_load(srow + x + 2*VECSZ);
            v_uint8 s3 = vx_load(srow + x + 3*VECSZ);
            
            v_uint8 m0 = vx_load(mrow + x);
            v_uint8 m1 = vx_load(mrow + x + VECSZ);
            v_uint8 m2 = vx_load(mrow + x + 2*VECSZ);
            v_uint8 m3 = vx_load(mrow + x + 3*VECSZ);
            
            // Compute s - m using saturating subtraction and add offset
            // Note: We use lookup table approach which is more portable
            v_uint16 s0_0, s0_1, s1_0, s1_1, s2_0, s2_1, s3_0, s3_1;
            v_uint16 m0_0, m0_1, m1_0, m1_1, m2_0, m2_1, m3_0, m3_1;
            
            v_expand(s0, s0_0, s0_1);
            v_expand(s1, s1_0, s1_1);
            v_expand(s2, s2_0, s2_1);
            v_expand(s3, s3_0, s3_1);
            
            v_expand(m0, m0_0, m0_1);
            v_expand(m1, m1_0, m1_1);
            v_expand(m2, m2_0, m2_1);
            v_expand(m3, m3_0, m3_1);
            
            // Compute differences and apply lookup
            v_int16 diff0_0 = v_reinterpret_as_s16(s0_0) - v_reinterpret_as_s16(m0_0) + vx_setall_s16(255);
            v_int16 diff0_1 = v_reinterpret_as_s16(s0_1) - v_reinterpret_as_s16(m0_1) + vx_setall_s16(255);
            v_int16 diff1_0 = v_reinterpret_as_s16(s1_0) - v_reinterpret_as_s16(m1_0) + vx_setall_s16(255);
            v_int16 diff1_1 = v_reinterpret_as_s16(s1_1) - v_reinterpret_as_s16(m1_1) + vx_setall_s16(255);
            v_int16 diff2_0 = v_reinterpret_as_s16(s2_0) - v_reinterpret_as_s16(m2_0) + vx_setall_s16(255);
            v_int16 diff2_1 = v_reinterpret_as_s16(s2_1) - v_reinterpret_as_s16(m2_1) + vx_setall_s16(255);
            v_int16 diff3_0 = v_reinterpret_as_s16(s3_0) - v_reinterpret_as_s16(m3_0) + vx_setall_s16(255);
            v_int16 diff3_1 = v_reinterpret_as_s16(s3_1) - v_reinterpret_as_s16(m3_1) + vx_setall_s16(255);
            
            // Apply threshold directly based on condition
            v_uint8 d0, d1, d2, d3;
            if (type == THRESH_BINARY)
            {
                // For THRESH_BINARY: output maxval where s > m (diff > 255)
                v_int16 thresh = vx_setall_s16(255 - idelta);
                v_uint16 r0_0 = v_reinterpret_as_u16(diff0_0 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r0_1 = v_reinterpret_as_u16(diff0_1 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r1_0 = v_reinterpret_as_u16(diff1_0 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r1_1 = v_reinterpret_as_u16(diff1_1 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r2_0 = v_reinterpret_as_u16(diff2_0 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r2_1 = v_reinterpret_as_u16(diff2_1 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r3_0 = v_reinterpret_as_u16(diff3_0 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r3_1 = v_reinterpret_as_u16(diff3_1 > thresh) & vx_setall<ushort>(maxval);
                
                d0 = v_pack(r0_0, r0_1);
                d1 = v_pack(r1_0, r1_1);
                d2 = v_pack(r2_0, r2_1);
                d3 = v_pack(r3_0, r3_1);
            }
            else
            {
                // For THRESH_BINARY_INV: output maxval where s <= m (diff <= 255)
                v_int16 thresh = vx_setall_s16(256 - idelta);
                v_uint16 r0_0 = v_reinterpret_as_u16(diff0_0 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r0_1 = v_reinterpret_as_u16(diff0_1 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r1_0 = v_reinterpret_as_u16(diff1_0 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r1_1 = v_reinterpret_as_u16(diff1_1 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r2_0 = v_reinterpret_as_u16(diff2_0 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r2_1 = v_reinterpret_as_u16(diff2_1 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r3_0 = v_reinterpret_as_u16(diff3_0 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r3_1 = v_reinterpret_as_u16(diff3_1 < thresh) & vx_setall<ushort>(maxval);
                
                d0 = v_pack(r0_0, r0_1);
                d1 = v_pack(r1_0, r1_1);
                d2 = v_pack(r2_0, r2_1);
                d3 = v_pack(r3_0, r3_1);
            }
            
            v_store(drow + x, d0);
            v_store(drow + x + VECSZ, d1);
            v_store(drow + x + 2*VECSZ, d2);
            v_store(drow + x + 3*VECSZ, d3);
        }
        
        // Process remaining pixels with single vector
        for (; x <= width - VECSZ; x += VECSZ)
        {
            v_uint8 s = vx_load(srow + x);
            v_uint8 m = vx_load(mrow + x);
            
            v_uint16 s_0, s_1, m_0, m_1;
            v_expand(s, s_0, s_1);
            v_expand(m, m_0, m_1);
            
            v_int16 diff_0 = v_reinterpret_as_s16(s_0) - v_reinterpret_as_s16(m_0) + vx_setall_s16(255);
            v_int16 diff_1 = v_reinterpret_as_s16(s_1) - v_reinterpret_as_s16(m_1) + vx_setall_s16(255);
            
            v_uint8 d;
            if (type == THRESH_BINARY)
            {
                v_int16 thresh = vx_setall_s16(255 - idelta);
                v_uint16 r_0 = v_reinterpret_as_u16(diff_0 > thresh) & vx_setall<ushort>(maxval);
                v_uint16 r_1 = v_reinterpret_as_u16(diff_1 > thresh) & vx_setall<ushort>(maxval);
                d = v_pack(r_0, r_1);
            }
            else
            {
                v_int16 thresh = vx_setall_s16(256 - idelta);
                v_uint16 r_0 = v_reinterpret_as_u16(diff_0 < thresh) & vx_setall<ushort>(maxval);
                v_uint16 r_1 = v_reinterpret_as_u16(diff_1 < thresh) & vx_setall<ushort>(maxval);
                d = v_pack(r_0, r_1);
            }
            
            v_store(drow + x, d);
        }
        
        // Scalar processing for remaining pixels
        for (; x < width; x++)
            drow[x] = tab[srow[x] - mrow[x] + 255];
    }
#else
    // Fallback to scalar implementation
    if (src_step == width && mean_step == width && dst_step == width)
    {
        width *= height;
        height = 1;
    }
    
    for (int y = 0; y < height; y++)
    {
        const uchar* srow = src_data + y * src_step;
        const uchar* mrow = mean_data + y * mean_step;
        uchar* drow = dst_data + y * dst_step;
        
        for (int x = 0; x < width; x++)
            drow[x] = tab[srow[x] - mrow[x] + 255];
    }
#endif
}

// AVX-512 specific optimization
#if CV_AVX512_SKX
static void adaptiveThresholdAVX512(const uchar* src_data, size_t src_step,
                                     const uchar* mean_data, size_t mean_step,
                                     uchar* dst_data, size_t dst_step,
                                     int width, int height,
                                     uchar maxval, int idelta, int type)
{
    CV_INSTRUMENT_REGION();
    
    const int VECSZ = 64; // AVX-512 processes 64 bytes
    
    // Process continuous data if possible
    if (src_step == width && mean_step == width && dst_step == width)
    {
        width *= height;
        height = 1;
    }
    
    for (int y = 0; y < height; y++)
    {
        const uchar* srow = src_data + y * src_step;
        const uchar* mrow = mean_data + y * mean_step;
        uchar* drow = dst_data + y * dst_step;
        
        int x = 0;
        
        // Main AVX-512 loop
        for (; x <= width - VECSZ; x += VECSZ)
        {
            __m512i s = _mm512_loadu_si512((__m512i*)(srow + x));
            __m512i m = _mm512_loadu_si512((__m512i*)(mrow + x));
            
            // Compute difference with offset
            __m512i diff = _mm512_sub_epi8(s, m);
            __m512i offset_diff = _mm512_add_epi8(diff, _mm512_set1_epi8(-1)); // Add 255 (as signed -1)
            
            __m512i result;
            if (type == THRESH_BINARY)
            {
                // For THRESH_BINARY: check if diff + 255 > -idelta
                __m512i thresh = _mm512_set1_epi8(-idelta);
                __mmask64 mask = _mm512_cmpgt_epi8_mask(offset_diff, thresh);
                result = _mm512_mask_blend_epi8(mask, _mm512_setzero_si512(), _mm512_set1_epi8(maxval));
            }
            else
            {
                // For THRESH_BINARY_INV: check if diff + 255 <= -idelta
                __m512i thresh = _mm512_set1_epi8(-idelta + 1);
                __mmask64 mask = _mm512_cmplt_epi8_mask(offset_diff, thresh);
                result = _mm512_mask_blend_epi8(mask, _mm512_setzero_si512(), _mm512_set1_epi8(maxval));
            }
            
            _mm512_storeu_si512((__m512i*)(drow + x), result);
        }
        
        // Scalar tail using lookup table
        uchar tab[768];
        if (type == THRESH_BINARY)
        {
            for (int i = 0; i < 768; i++)
                tab[i] = (uchar)(i - 255 > -idelta ? maxval : 0);
        }
        else
        {
            for (int i = 0; i < 768; i++)
                tab[i] = (uchar)(i - 255 <= -idelta ? maxval : 0);
        }
        
        for (; x < width; x++)
            drow[x] = tab[srow[x] - mrow[x] + 255];
    }
}
#endif // CV_AVX512_SKX

} // namespace hal
} // namespace cv

#endif // OPENCV_IMGPROC_ADAPTIVE_THRESHOLD_SIMD_HPP