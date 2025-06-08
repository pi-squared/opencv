// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_IMGPROC_CLAHE_SIMD_HPP
#define CV_IMGPROC_CLAHE_SIMD_HPP

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace clahe_simd {

#if CV_SIMD

// SIMD optimized histogram calculation for 8-bit images
template<typename T>
inline void calcHistogram_SIMD(const T* src, int* hist, int width, int height, size_t step)
{
    const int HIST_SIZE = 256;
    const int SIMD_WIDTH = v_uint8::nlanes;
    
    // Clear histogram
    v_int32 v_zero = vx_setzero_s32();
    for (int i = 0; i < HIST_SIZE; i += v_int32::nlanes)
    {
        v_store(hist + i, v_zero);
    }
    
    // For 8-bit images, we can use a more efficient approach
    // Process multiple rows to improve cache efficiency
    for (int y = 0; y < height; y++)
    {
        const T* ptr = src + y * step;
        int x = 0;
        
        // Main SIMD loop
        for (; x <= width - SIMD_WIDTH; x += SIMD_WIDTH)
        {
            v_uint8 v_src = vx_load(ptr + x);
            
            // For histogram, we need to handle scatter operation
            // Since SIMD doesn't handle scatter well, we'll process in chunks
            // and use local histograms to reduce conflicts
            for (int i = 0; i < SIMD_WIDTH; i++)
            {
                hist[ptr[x + i]]++;
            }
        }
        
        // Process remaining pixels
        for (; x < width; x++)
        {
            hist[ptr[x]]++;
        }
    }
}

// SIMD optimized histogram clipping
inline int clipHistogram_SIMD(int* hist, int histSize, int clipLimit)
{
    int clipped = 0;
    
#if CV_SIMD_WIDTH >= 16  // AVX2 or higher
    v_int32 v_clipLimit = vx_setall(clipLimit);
    v_int32 v_clipped = vx_setzero_s32();
    
    int i = 0;
    for (; i <= histSize - v_int32::nlanes; i += v_int32::nlanes)
    {
        v_int32 v_hist = vx_load(hist + i);
        v_int32 v_mask = v_gt(v_hist, v_clipLimit);
        v_int32 v_excess = v_sub(v_hist, v_clipLimit);
        v_int32 v_masked = v_and(v_excess, v_mask);
        v_clipped = v_add(v_clipped, v_masked);
        v_hist = v_select(v_mask, v_clipLimit, v_hist);
        v_store(hist + i, v_hist);
    }
    
    // Sum up clipped values
    clipped = v_reduce_sum(v_clipped);
    
    // Process remaining elements
    for (; i < histSize; i++)
    {
        if (hist[i] > clipLimit)
        {
            clipped += hist[i] - clipLimit;
            hist[i] = clipLimit;
        }
    }
#else
    // Fallback to scalar implementation
    for (int i = 0; i < histSize; i++)
    {
        if (hist[i] > clipLimit)
        {
            clipped += hist[i] - clipLimit;
            hist[i] = clipLimit;
        }
    }
#endif
    
    return clipped;
}

// SIMD optimized interpolation for 8-bit images
inline void interpolate_SIMD(const uchar* srcRow, uchar* dstRow, int width,
                            const uchar* lutPlane1, const uchar* lutPlane2,
                            const int* ind1_p, const int* ind2_p,
                            const float* xa_p, const float* xa1_p,
                            float ya, float ya1)
{
    int x = 0;
    
#if CV_SIMD
    const int SIMD_WIDTH = v_float32::nlanes;
    
    v_float32 v_ya = vx_setall(ya);
    v_float32 v_ya1 = vx_setall(ya1);
    
    // Process multiple pixels at once
    for (; x <= width - SIMD_WIDTH; x += SIMD_WIDTH)
    {
        // Load source values
        v_uint32 v_src = vx_load_expand_q(srcRow + x);
        
        // Load indices
        v_int32 v_ind1 = vx_load(ind1_p + x);
        v_int32 v_ind2 = vx_load(ind2_p + x);
        
        // Add source values to indices
        v_ind1 = v_add(v_ind1, v_reinterpret_as_s32(v_src));
        v_ind2 = v_add(v_ind2, v_reinterpret_as_s32(v_src));
        
        // Load interpolation weights
        v_float32 v_xa = vx_load(xa_p + x);
        v_float32 v_xa1 = vx_load(xa1_p + x);
        
        // Gather LUT values (this is the challenging part for SIMD)
        // We need to extract indices and do scalar lookups
        alignas(16) int ind1_arr[SIMD_WIDTH], ind2_arr[SIMD_WIDTH];
        v_store_aligned(ind1_arr, v_ind1);
        v_store_aligned(ind2_arr, v_ind2);
        
        float lut_vals1[SIMD_WIDTH], lut_vals2[SIMD_WIDTH];
        float lut_vals3[SIMD_WIDTH], lut_vals4[SIMD_WIDTH];
        
        for (int i = 0; i < SIMD_WIDTH; i++)
        {
            lut_vals1[i] = (float)lutPlane1[ind1_arr[i]];
            lut_vals2[i] = (float)lutPlane1[ind2_arr[i]];
            lut_vals3[i] = (float)lutPlane2[ind1_arr[i]];
            lut_vals4[i] = (float)lutPlane2[ind2_arr[i]];
        }
        
        v_float32 v_lut1 = vx_load(lut_vals1);
        v_float32 v_lut2 = vx_load(lut_vals2);
        v_float32 v_lut3 = vx_load(lut_vals3);
        v_float32 v_lut4 = vx_load(lut_vals4);
        
        // Bilinear interpolation
        v_float32 v_temp1 = v_mul(v_lut2, v_xa);
        v_float32 v_temp2 = v_mul(v_lut4, v_xa);
        v_float32 v_interp1 = v_muladd(v_lut1, v_xa1, v_temp1);
        v_float32 v_interp2 = v_muladd(v_lut3, v_xa1, v_temp2);
        v_float32 v_temp3 = v_mul(v_interp2, v_ya);
        v_float32 v_result = v_muladd(v_interp1, v_ya1, v_temp3);
        
        // Convert back to uchar
        v_int32 v_res_int = v_round(v_result);
        // Pack int32 -> int16 -> uchar
        v_int16 v_res_16 = v_pack(v_res_int, v_res_int);
        v_uint8 v_res_8 = v_pack_u(v_res_16, v_res_16);
        // Store only the needed pixels
        for (int i = 0; i < SIMD_WIDTH && x + i < width; i++)
        {
            dstRow[x + i] = v_res_8.get0();
            v_res_8 = v_rotate_right<1>(v_res_8);
        }
    }
#endif
    
    // Scalar fallback for remaining pixels
    for (; x < width; x++)
    {
        int srcVal = srcRow[x];
        int ind1 = ind1_p[x] + srcVal;
        int ind2 = ind2_p[x] + srcVal;
        
        float res = (lutPlane1[ind1] * xa1_p[x] + lutPlane1[ind2] * xa_p[x]) * ya1 +
                    (lutPlane2[ind1] * xa1_p[x] + lutPlane2[ind2] * xa_p[x]) * ya;
        
        dstRow[x] = saturate_cast<uchar>(res);
    }
}

#ifdef CV_CPU_DISPATCH_MODE

// AVX-512 specific optimizations
#if CV_CPU_BASELINE_COMPILE_AVX512_SKX || CV_CPU_DISPATCH_COMPILE_AVX512_SKX

inline void interpolate_AVX512(const uchar* srcRow, uchar* dstRow, int width,
                              const uchar* lutPlane1, const uchar* lutPlane2,
                              const int* ind1_p, const int* ind2_p,
                              const float* xa_p, const float* xa1_p,
                              float ya, float ya1)
{
    int x = 0;
    
    __m512 v_ya = _mm512_set1_ps(ya);
    __m512 v_ya1 = _mm512_set1_ps(ya1);
    
    // Process 16 pixels at once
    for (; x <= width - 16; x += 16)
    {
        // Load source values
        __m128i v_src8 = _mm_loadu_si128((const __m128i*)(srcRow + x));
        __m512i v_src = _mm512_cvtepu8_epi32(v_src8);
        
        // Load indices
        __m512i v_ind1 = _mm512_loadu_si512((const __m512i*)(ind1_p + x));
        __m512i v_ind2 = _mm512_loadu_si512((const __m512i*)(ind2_p + x));
        
        // Add source values to indices
        v_ind1 = _mm512_add_epi32(v_ind1, v_src);
        v_ind2 = _mm512_add_epi32(v_ind2, v_src);
        
        // Load interpolation weights
        __m512 v_xa = _mm512_loadu_ps(xa_p + x);
        __m512 v_xa1 = _mm512_loadu_ps(xa1_p + x);
        
        // Gather LUT values using AVX-512 gather instructions
        __m512i v_lut1_i = _mm512_i32gather_epi32(v_ind1, (const int*)lutPlane1, 1);
        __m512i v_lut2_i = _mm512_i32gather_epi32(v_ind2, (const int*)lutPlane1, 1);
        __m512i v_lut3_i = _mm512_i32gather_epi32(v_ind1, (const int*)lutPlane2, 1);
        __m512i v_lut4_i = _mm512_i32gather_epi32(v_ind2, (const int*)lutPlane2, 1);
        
        // Convert to float
        __m512 v_lut1 = _mm512_cvtepi32_ps(v_lut1_i);
        __m512 v_lut2 = _mm512_cvtepi32_ps(v_lut2_i);
        __m512 v_lut3 = _mm512_cvtepi32_ps(v_lut3_i);
        __m512 v_lut4 = _mm512_cvtepi32_ps(v_lut4_i);
        
        // Bilinear interpolation using FMA
        __m512 v_interp1 = _mm512_fmadd_ps(v_lut1, v_xa1, _mm512_mul_ps(v_lut2, v_xa));
        __m512 v_interp2 = _mm512_fmadd_ps(v_lut3, v_xa1, _mm512_mul_ps(v_lut4, v_xa));
        __m512 v_result = _mm512_fmadd_ps(v_interp1, v_ya1, _mm512_mul_ps(v_interp2, v_ya));
        
        // Convert back to uchar with saturation
        __m512i v_res_int = _mm512_cvtps_epi32(v_result);
        __m256i v_res_16 = _mm512_cvtsepi32_epi16(v_res_int);
        __m128i v_res_8 = _mm256_cvtsepi16_epi8(v_res_16);
        
        _mm_storeu_si128((__m128i*)(dstRow + x), v_res_8);
    }
    
    // Process remaining pixels
    for (; x < width; x++)
    {
        int srcVal = srcRow[x];
        int ind1 = ind1_p[x] + srcVal;
        int ind2 = ind2_p[x] + srcVal;
        
        float res = (lutPlane1[ind1] * xa1_p[x] + lutPlane1[ind2] * xa_p[x]) * ya1 +
                    (lutPlane2[ind1] * xa1_p[x] + lutPlane2[ind2] * xa_p[x]) * ya;
        
        dstRow[x] = saturate_cast<uchar>(res);
    }
}

#endif // AVX-512

#endif // CV_CPU_DISPATCH_MODE

#endif // CV_SIMD

} // namespace clahe_simd
} // namespace cv

#endif // CV_IMGPROC_CLAHE_SIMD_HPP