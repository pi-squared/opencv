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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//  AVX-512 optimized version
//
// */

#include "precomp.hpp"
#include "imgwarp.hpp"

namespace cv
{
namespace opt_AVX512_SKX
{

int warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw)
{
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    int x1 = 0;
    
    // Process 32 values at once with AVX-512 (double the throughput of AVX2)
    __m512i fxy_mask = _mm512_set1_epi32(INTER_TAB_SIZE - 1);
    __m512i XX = _mm512_set1_epi32(X0), YY = _mm512_set1_epi32(Y0);
    
    // Cache prefetching for better memory access patterns
    #if defined(__x86_64__) || defined(_M_X64)
    _mm_prefetch((const char*)(adelta + 32), _MM_HINT_T0);
    _mm_prefetch((const char*)(bdelta + 32), _MM_HINT_T0);
    #endif
    
    for (; x1 <= bw - 32; x1 += 32)
    {
        __m512i tx0, tx1, ty0, ty1;
        
        // Load and process 32 values instead of 16
        tx0 = _mm512_add_epi32(_mm512_loadu_si512((const __m512i*)(adelta + x1)), XX);
        ty0 = _mm512_add_epi32(_mm512_loadu_si512((const __m512i*)(bdelta + x1)), YY);
        tx1 = _mm512_add_epi32(_mm512_loadu_si512((const __m512i*)(adelta + x1 + 16)), XX);
        ty1 = _mm512_add_epi32(_mm512_loadu_si512((const __m512i*)(bdelta + x1 + 16)), YY);
        
        // Prefetch next iteration's data
        #if defined(__x86_64__) || defined(_M_X64)
        _mm_prefetch((const char*)(adelta + x1 + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(bdelta + x1 + 64), _MM_HINT_T0);
        #endif

        tx0 = _mm512_srai_epi32(tx0, AB_BITS - INTER_BITS);
        ty0 = _mm512_srai_epi32(ty0, AB_BITS - INTER_BITS);
        tx1 = _mm512_srai_epi32(tx1, AB_BITS - INTER_BITS);
        ty1 = _mm512_srai_epi32(ty1, AB_BITS - INTER_BITS);

        __m256i fx_0 = _mm512_cvtepi32_epi16(_mm512_and_si512(tx0, fxy_mask));
        __m256i fx_1 = _mm512_cvtepi32_epi16(_mm512_and_si512(tx1, fxy_mask));
        __m256i fy_0 = _mm512_cvtepi32_epi16(_mm512_and_si512(ty0, fxy_mask));
        __m256i fy_1 = _mm512_cvtepi32_epi16(_mm512_and_si512(ty1, fxy_mask));
        
        __m256i tx0_16 = _mm512_cvtepi32_epi16(_mm512_srai_epi32(tx0, INTER_BITS));
        __m256i tx1_16 = _mm512_cvtepi32_epi16(_mm512_srai_epi32(tx1, INTER_BITS));
        __m256i ty0_16 = _mm512_cvtepi32_epi16(_mm512_srai_epi32(ty0, INTER_BITS));
        __m256i ty1_16 = _mm512_cvtepi32_epi16(_mm512_srai_epi32(ty1, INTER_BITS));
        
        // Combine fx and fy for alpha values
        fx_0 = _mm256_adds_epi16(fx_0, _mm256_slli_epi16(fy_0, INTER_BITS));
        fx_1 = _mm256_adds_epi16(fx_1, _mm256_slli_epi16(fy_1, INTER_BITS));
        
        // Store interleaved x,y coordinates
        __m256i xy0_lo = _mm256_unpacklo_epi16(tx0_16, ty0_16);
        __m256i xy0_hi = _mm256_unpackhi_epi16(tx0_16, ty0_16);
        __m256i xy1_lo = _mm256_unpacklo_epi16(tx1_16, ty1_16);
        __m256i xy1_hi = _mm256_unpackhi_epi16(tx1_16, ty1_16);
        
        _mm256_storeu_si256((__m256i*)(xy + x1 * 2), xy0_lo);
        _mm256_storeu_si256((__m256i*)(xy + x1 * 2 + 16), xy0_hi);
        _mm256_storeu_si256((__m256i*)(xy + x1 * 2 + 32), xy1_lo);
        _mm256_storeu_si256((__m256i*)(xy + x1 * 2 + 48), xy1_hi);
        
        // Store alpha values
        _mm256_storeu_si256((__m256i*)(alpha + x1), fx_0);
        _mm256_storeu_si256((__m256i*)(alpha + x1 + 16), fx_1);
    }
    
    // Fall back to AVX2 version for remaining elements
    if (x1 <= bw - 16)
    {
        __m256i fxy_mask_256 = _mm256_set1_epi32(INTER_TAB_SIZE - 1);
        __m256i XX_256 = _mm256_set1_epi32(X0), YY_256 = _mm256_set1_epi32(Y0);
        
        __m256i tx0, tx1, ty0, ty1;
        tx0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(adelta + x1)), XX_256);
        ty0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(bdelta + x1)), YY_256);
        tx1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(adelta + x1 + 8)), XX_256);
        ty1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(bdelta + x1 + 8)), YY_256);

        tx0 = _mm256_srai_epi32(tx0, AB_BITS - INTER_BITS);
        ty0 = _mm256_srai_epi32(ty0, AB_BITS - INTER_BITS);
        tx1 = _mm256_srai_epi32(tx1, AB_BITS - INTER_BITS);
        ty1 = _mm256_srai_epi32(ty1, AB_BITS - INTER_BITS);

        __m256i fx_ = _mm256_packs_epi32(_mm256_and_si256(tx0, fxy_mask_256),
            _mm256_and_si256(tx1, fxy_mask_256));
        __m256i fy_ = _mm256_packs_epi32(_mm256_and_si256(ty0, fxy_mask_256),
            _mm256_and_si256(ty1, fxy_mask_256));
        tx0 = _mm256_packs_epi32(_mm256_srai_epi32(tx0, INTER_BITS),
            _mm256_srai_epi32(tx1, INTER_BITS));
        ty0 = _mm256_packs_epi32(_mm256_srai_epi32(ty0, INTER_BITS),
            _mm256_srai_epi32(ty1, INTER_BITS));
        fx_ = _mm256_adds_epi16(fx_, _mm256_slli_epi16(fy_, INTER_BITS));
        fx_ = _mm256_permute4x64_epi64(fx_, (3 << 6) + (1 << 4) + (2 << 2) + 0);

        _mm256_storeu_si256((__m256i*)(xy + x1 * 2), _mm256_unpacklo_epi16(tx0, ty0));
        _mm256_storeu_si256((__m256i*)(xy + x1 * 2 + 16), _mm256_unpackhi_epi16(tx0, ty0));
        _mm256_storeu_si256((__m256i*)(alpha + x1), fx_);
        
        x1 += 16;
    }
    
    return x1;
}

}
}
/* End of file. */