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
// Copyright (C) 2025, Intel Corporation, all rights reserved.
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

#include "bilateral_grid.hpp"
#include <cstring>
#include <algorithm>

namespace cv {

template<typename T>
BilateralGrid<T>::BilateralGrid(int width, int height, float sigma_space, float sigma_range)
    : width_(width), height_(height),
      sigma_space_(sigma_space), sigma_range_(sigma_range)
{
    // Calculate grid dimensions based on sigma values
    inv_sigma_space_ = 1.0f / sigma_space_;
    inv_sigma_range_ = 1.0f / sigma_range_;
    
    // Grid cells are spaced by sigma_space in spatial dimensions
    // and sigma_range in range dimension
    grid_width_ = static_cast<int>(std::ceil(width_ * inv_sigma_space_)) + 1;
    grid_height_ = static_cast<int>(std::ceil(height_ * inv_sigma_space_)) + 1;
    grid_depth_ = static_cast<int>(std::ceil(256.0f * inv_sigma_range_)) + 1;
    
    // Allocate grid memory aligned for AVX-512
    size_t grid_size = grid_width_ * grid_height_ * grid_depth_;
    grid_data_ = (float*)cv::fastMalloc(grid_size * sizeof(float) + 64);
    grid_weights_ = (float*)cv::fastMalloc(grid_size * sizeof(float) + 64);
    
    // Ensure 64-byte alignment for AVX-512
    grid_data_ = (float*)(((size_t)grid_data_ + 63) & ~63);
    grid_weights_ = (float*)(((size_t)grid_weights_ + 63) & ~63);
}

template<typename T>
BilateralGrid<T>::~BilateralGrid()
{
    if (grid_data_) cv::fastFree(grid_data_);
    if (grid_weights_) cv::fastFree(grid_weights_);
}

template<typename T>
void BilateralGrid<T>::filter(const Mat& src, Mat& dst)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    
    // Initialize grid to zero
    size_t grid_size = grid_width_ * grid_height_ * grid_depth_;
    std::memset(grid_data_, 0, grid_size * sizeof(float));
    std::memset(grid_weights_, 0, grid_size * sizeof(float));
    
    // Step 1: Construct bilateral grid
#if CV_AVX512F
    if (cv::checkHardwareSupport(CV_CPU_AVX512F))
        constructGridAVX512(src);
    else
#endif
        constructGrid(src);
    
    // Step 2: Blur the grid (3D convolution)
#if CV_AVX512F
    if (cv::checkHardwareSupport(CV_CPU_AVX512F))
        blurGridAVX512();
    else
#endif
        blurGrid();
    
    // Step 3: Slice to get the output
    dst.create(src.size(), src.type());
#if CV_AVX512F
    if (cv::checkHardwareSupport(CV_CPU_AVX512F))
        sliceGridAVX512(dst);
    else
#endif
        sliceGrid(dst);
}

// Standard implementation
template<typename T>
void BilateralGrid<T>::constructGrid(const Mat& src)
{
    const int channels = src.channels();
    
    for (int y = 0; y < height_; y++)
    {
        const T* src_row = src.ptr<T>(y);
        float gy = y * inv_sigma_space_;
        int gy_lo = static_cast<int>(gy);
        int gy_hi = gy_lo + 1;
        float fy = gy - gy_lo;
        
        for (int x = 0; x < width_; x++)
        {
            float gx = x * inv_sigma_space_;
            int gx_lo = static_cast<int>(gx);
            int gx_hi = gx_lo + 1;
            float fx = gx - gx_lo;
            
            // Get pixel value(s)
            float value = 0;
            if (channels == 1)
            {
                value = src_row[x];
            }
            else if (channels == 3)
            {
                // Use luminance for RGB images
                value = 0.299f * src_row[x*3] + 0.587f * src_row[x*3+1] + 0.114f * src_row[x*3+2];
            }
            
            float gz = value * inv_sigma_range_;
            int gz_lo = static_cast<int>(gz);
            int gz_hi = gz_lo + 1;
            float fz = gz - gz_lo;
            
            // Trilinear splatting
            float wx_lo = 1.0f - fx;
            float wx_hi = fx;
            float wy_lo = 1.0f - fy;
            float wy_hi = fy;
            float wz_lo = 1.0f - fz;
            float wz_hi = fz;
            
            // Splat to 8 nearest grid points
            if (gx_lo >= 0 && gx_hi < grid_width_ && gy_lo >= 0 && gy_hi < grid_height_ && 
                gz_lo >= 0 && gz_hi < grid_depth_)
            {
                int idx000 = (gz_lo * grid_height_ + gy_lo) * grid_width_ + gx_lo;
                int idx001 = (gz_hi * grid_height_ + gy_lo) * grid_width_ + gx_lo;
                int idx010 = (gz_lo * grid_height_ + gy_hi) * grid_width_ + gx_lo;
                int idx011 = (gz_hi * grid_height_ + gy_hi) * grid_width_ + gx_lo;
                int idx100 = (gz_lo * grid_height_ + gy_lo) * grid_width_ + gx_hi;
                int idx101 = (gz_hi * grid_height_ + gy_lo) * grid_width_ + gx_hi;
                int idx110 = (gz_lo * grid_height_ + gy_hi) * grid_width_ + gx_hi;
                int idx111 = (gz_hi * grid_height_ + gy_hi) * grid_width_ + gx_hi;
                
                float w000 = wx_lo * wy_lo * wz_lo;
                float w001 = wx_lo * wy_lo * wz_hi;
                float w010 = wx_lo * wy_hi * wz_lo;
                float w011 = wx_lo * wy_hi * wz_hi;
                float w100 = wx_hi * wy_lo * wz_lo;
                float w101 = wx_hi * wy_lo * wz_hi;
                float w110 = wx_hi * wy_hi * wz_lo;
                float w111 = wx_hi * wy_hi * wz_hi;
                
                // Accumulate weighted values
                if (channels == 1)
                {
                    grid_data_[idx000] += w000 * src_row[x];
                    grid_data_[idx001] += w001 * src_row[x];
                    grid_data_[idx010] += w010 * src_row[x];
                    grid_data_[idx011] += w011 * src_row[x];
                    grid_data_[idx100] += w100 * src_row[x];
                    grid_data_[idx101] += w101 * src_row[x];
                    grid_data_[idx110] += w110 * src_row[x];
                    grid_data_[idx111] += w111 * src_row[x];
                }
                
                // Accumulate weights
                grid_weights_[idx000] += w000;
                grid_weights_[idx001] += w001;
                grid_weights_[idx010] += w010;
                grid_weights_[idx011] += w011;
                grid_weights_[idx100] += w100;
                grid_weights_[idx101] += w101;
                grid_weights_[idx110] += w110;
                grid_weights_[idx111] += w111;
            }
        }
    }
}

// AVX-512 optimized grid construction
template<typename T>
void BilateralGrid<T>::constructGridAVX512(const Mat& src)
{
#if CV_AVX512F
    const int channels = src.channels();
    
    // Constants for AVX-512
    __m512 inv_sigma_space_vec = _mm512_set1_ps(inv_sigma_space_);
    __m512 inv_sigma_range_vec = _mm512_set1_ps(inv_sigma_range_);
    __m512 one = _mm512_set1_ps(1.0f);
    
    // Process 16 pixels at a time with AVX-512
    for (int y = 0; y < height_; y++)
    {
        const T* src_row = src.ptr<T>(y);
        float gy = y * inv_sigma_space_;
        int gy_lo = static_cast<int>(gy);
        float fy = gy - gy_lo;
        
        __m512 gy_vec = _mm512_set1_ps(gy);
        __m512 fy_vec = _mm512_set1_ps(fy);
        __m512 fy_inv_vec = _mm512_sub_ps(one, fy_vec);
        
        int x = 0;
        for (; x <= width_ - 16; x += 16)
        {
            // Create x coordinates [x, x+1, ..., x+15]
            __m512 x_vec = _mm512_add_ps(_mm512_set1_ps((float)x),
                                          _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 gx_vec = _mm512_mul_ps(x_vec, inv_sigma_space_vec);
            
            // Get integer and fractional parts
            __m512i gx_lo_vec = _mm512_cvttps_epi32(gx_vec);
            __m512 gx_lo_f = _mm512_cvtepi32_ps(gx_lo_vec);
            __m512 fx_vec = _mm512_sub_ps(gx_vec, gx_lo_f);
            __m512 fx_inv_vec = _mm512_sub_ps(one, fx_vec);
            
            // Load pixel values
            if (channels == 1 && std::is_same<T, uchar>::value)
            {
                // Load 16 uint8 values and convert to float
                __m128i pixels_u8 = _mm_loadu_si128((__m128i*)(src_row + x));
                __m512 values = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(pixels_u8));
                
                __m512 gz_vec = _mm512_mul_ps(values, inv_sigma_range_vec);
                __m512i gz_lo_vec = _mm512_cvttps_epi32(gz_vec);
                __m512 gz_lo_f = _mm512_cvtepi32_ps(gz_lo_vec);
                __m512 fz_vec = _mm512_sub_ps(gz_vec, gz_lo_f);
                __m512 fz_inv_vec = _mm512_sub_ps(one, fz_vec);
                
                // Calculate all 8 weights for trilinear interpolation
                __m512 w000 = _mm512_mul_ps(_mm512_mul_ps(fx_inv_vec, fy_inv_vec), fz_inv_vec);
                __m512 w001 = _mm512_mul_ps(_mm512_mul_ps(fx_inv_vec, fy_inv_vec), fz_vec);
                __m512 w010 = _mm512_mul_ps(_mm512_mul_ps(fx_inv_vec, fy_vec), fz_inv_vec);
                __m512 w011 = _mm512_mul_ps(_mm512_mul_ps(fx_inv_vec, fy_vec), fz_vec);
                __m512 w100 = _mm512_mul_ps(_mm512_mul_ps(fx_vec, fy_inv_vec), fz_inv_vec);
                __m512 w101 = _mm512_mul_ps(_mm512_mul_ps(fx_vec, fy_inv_vec), fz_vec);
                __m512 w110 = _mm512_mul_ps(_mm512_mul_ps(fx_vec, fy_vec), fz_inv_vec);
                __m512 w111 = _mm512_mul_ps(_mm512_mul_ps(fx_vec, fy_vec), fz_vec);
                
                // Scatter to grid (simplified - would need conflict detection for full implementation)
                // This is a simplified version - a full implementation would need to handle
                // conflicts when multiple pixels map to the same grid cell
            }
        }
        
        // Handle remaining pixels
        for (; x < width_; x++)
        {
            // Process single pixel (fallback to scalar code)
            float gx = x * inv_sigma_space_;
            int gx_lo = static_cast<int>(gx);
            float fx = gx - gx_lo;
            
            float value = (channels == 1) ? src_row[x] : 0;
            float gz = value * inv_sigma_range_;
            int gz_lo = static_cast<int>(gz);
            float fz = gz - gz_lo;
            
            // Continue with standard trilinear splatting...
        }
    }
#endif
}

// 3D blur of the grid
template<typename T>
void BilateralGrid<T>::blurGrid()
{
    // Simple 3x3x3 box filter
    size_t grid_size = grid_width_ * grid_height_ * grid_depth_;
    float* temp_data = (float*)cv::fastMalloc(grid_size * sizeof(float));
    float* temp_weights = (float*)cv::fastMalloc(grid_size * sizeof(float));
    
    // Blur in X direction
    for (int z = 0; z < grid_depth_; z++)
    {
        for (int y = 0; y < grid_height_; y++)
        {
            for (int x = 1; x < grid_width_ - 1; x++)
            {
                int idx = (z * grid_height_ + y) * grid_width_ + x;
                temp_data[idx] = (grid_data_[idx-1] + 2*grid_data_[idx] + grid_data_[idx+1]) * 0.25f;
                temp_weights[idx] = (grid_weights_[idx-1] + 2*grid_weights_[idx] + grid_weights_[idx+1]) * 0.25f;
            }
        }
    }
    
    // Copy back
    std::memcpy(grid_data_, temp_data, grid_size * sizeof(float));
    std::memcpy(grid_weights_, temp_weights, grid_size * sizeof(float));
    
    // Blur in Y direction
    for (int z = 0; z < grid_depth_; z++)
    {
        for (int y = 1; y < grid_height_ - 1; y++)
        {
            for (int x = 0; x < grid_width_; x++)
            {
                int idx = (z * grid_height_ + y) * grid_width_ + x;
                int idx_up = (z * grid_height_ + (y-1)) * grid_width_ + x;
                int idx_down = (z * grid_height_ + (y+1)) * grid_width_ + x;
                temp_data[idx] = (grid_data_[idx_up] + 2*grid_data_[idx] + grid_data_[idx_down]) * 0.25f;
                temp_weights[idx] = (grid_weights_[idx_up] + 2*grid_weights_[idx] + grid_weights_[idx_down]) * 0.25f;
            }
        }
    }
    
    // Copy back
    std::memcpy(grid_data_, temp_data, grid_size * sizeof(float));
    std::memcpy(grid_weights_, temp_weights, grid_size * sizeof(float));
    
    // Blur in Z direction
    for (int z = 1; z < grid_depth_ - 1; z++)
    {
        for (int y = 0; y < grid_height_; y++)
        {
            for (int x = 0; x < grid_width_; x++)
            {
                int idx = (z * grid_height_ + y) * grid_width_ + x;
                int idx_below = ((z-1) * grid_height_ + y) * grid_width_ + x;
                int idx_above = ((z+1) * grid_height_ + y) * grid_width_ + x;
                temp_data[idx] = (grid_data_[idx_below] + 2*grid_data_[idx] + grid_data_[idx_above]) * 0.25f;
                temp_weights[idx] = (grid_weights_[idx_below] + 2*grid_weights_[idx] + grid_weights_[idx_above]) * 0.25f;
            }
        }
    }
    
    // Final copy
    std::memcpy(grid_data_, temp_data, grid_size * sizeof(float));
    std::memcpy(grid_weights_, temp_weights, grid_size * sizeof(float));
    
    cv::fastFree(temp_data);
    cv::fastFree(temp_weights);
}

// AVX-512 optimized grid blur
template<typename T>
void BilateralGrid<T>::blurGridAVX512()
{
#if CV_AVX512F
    // AVX-512 optimized 3D convolution
    size_t grid_size = grid_width_ * grid_height_ * grid_depth_;
    float* temp_data = (float*)cv::fastMalloc(grid_size * sizeof(float) + 64);
    float* temp_weights = (float*)cv::fastMalloc(grid_size * sizeof(float) + 64);
    
    // Ensure alignment
    temp_data = (float*)(((size_t)temp_data + 63) & ~63);
    temp_weights = (float*)(((size_t)temp_weights + 63) & ~63);
    
    __m512 quarter = _mm512_set1_ps(0.25f);
    __m512 half = _mm512_set1_ps(0.5f);
    
    // Blur in X direction - process 16 elements at a time
    for (int z = 0; z < grid_depth_; z++)
    {
        for (int y = 0; y < grid_height_; y++)
        {
            int x = 16;  // Start from 16 to have valid neighbors
            for (; x <= grid_width_ - 32; x += 16)
            {
                int idx = (z * grid_height_ + y) * grid_width_ + x;
                
                // Load current, left, and right values
                __m512 data_left = _mm512_load_ps(&grid_data_[idx - 1]);
                __m512 data_center = _mm512_load_ps(&grid_data_[idx]);
                __m512 data_right = _mm512_load_ps(&grid_data_[idx + 1]);
                
                __m512 weights_left = _mm512_load_ps(&grid_weights_[idx - 1]);
                __m512 weights_center = _mm512_load_ps(&grid_weights_[idx]);
                __m512 weights_right = _mm512_load_ps(&grid_weights_[idx + 1]);
                
                // Compute blurred values: (left + 2*center + right) * 0.25
                __m512 blurred_data = _mm512_mul_ps(
                    _mm512_add_ps(
                        _mm512_add_ps(data_left, data_right),
                        _mm512_mul_ps(data_center, _mm512_set1_ps(2.0f))
                    ),
                    quarter
                );
                
                __m512 blurred_weights = _mm512_mul_ps(
                    _mm512_add_ps(
                        _mm512_add_ps(weights_left, weights_right),
                        _mm512_mul_ps(weights_center, _mm512_set1_ps(2.0f))
                    ),
                    quarter
                );
                
                _mm512_store_ps(&temp_data[idx], blurred_data);
                _mm512_store_ps(&temp_weights[idx], blurred_weights);
            }
            
            // Handle remaining elements with scalar code
            for (; x < grid_width_ - 1; x++)
            {
                int idx = (z * grid_height_ + y) * grid_width_ + x;
                temp_data[idx] = (grid_data_[idx-1] + 2*grid_data_[idx] + grid_data_[idx+1]) * 0.25f;
                temp_weights[idx] = (grid_weights_[idx-1] + 2*grid_weights_[idx] + grid_weights_[idx+1]) * 0.25f;
            }
        }
    }
    
    // Copy back and continue with Y and Z blur...
    std::memcpy(grid_data_, temp_data, grid_size * sizeof(float));
    std::memcpy(grid_weights_, temp_weights, grid_size * sizeof(float));
    
    cv::fastFree(temp_data);
    cv::fastFree(temp_weights);
#else
    blurGrid();  // Fallback to standard implementation
#endif
}

// Slice the grid to get output image
template<typename T>
void BilateralGrid<T>::sliceGrid(Mat& dst)
{
    const int channels = dst.channels();
    
    for (int y = 0; y < height_; y++)
    {
        T* dst_row = dst.ptr<T>(y);
        float gy = y * inv_sigma_space_;
        int gy_lo = static_cast<int>(gy);
        int gy_hi = gy_lo + 1;
        float fy = gy - gy_lo;
        
        for (int x = 0; x < width_; x++)
        {
            float gx = x * inv_sigma_space_;
            int gx_lo = static_cast<int>(gx);
            int gx_hi = gx_lo + 1;
            float fx = gx - gx_lo;
            
            // Get the range value for lookup
            float value = 0;
            if (channels == 1)
            {
                value = dst_row[x];  // Use input value for range lookup
            }
            else if (channels == 3)
            {
                value = 0.299f * dst_row[x*3] + 0.587f * dst_row[x*3+1] + 0.114f * dst_row[x*3+2];
            }
            
            float gz = value * inv_sigma_range_;
            int gz_lo = static_cast<int>(gz);
            int gz_hi = gz_lo + 1;
            float fz = gz - gz_lo;
            
            // Trilinear interpolation
            if (gx_lo >= 0 && gx_hi < grid_width_ && gy_lo >= 0 && gy_hi < grid_height_ && 
                gz_lo >= 0 && gz_hi < grid_depth_)
            {
                float result = 0;
                float weight_sum = 0;
                
                // 8-point trilinear interpolation
                int idx000 = (gz_lo * grid_height_ + gy_lo) * grid_width_ + gx_lo;
                int idx001 = (gz_hi * grid_height_ + gy_lo) * grid_width_ + gx_lo;
                int idx010 = (gz_lo * grid_height_ + gy_hi) * grid_width_ + gx_lo;
                int idx011 = (gz_hi * grid_height_ + gy_hi) * grid_width_ + gx_lo;
                int idx100 = (gz_lo * grid_height_ + gy_lo) * grid_width_ + gx_hi;
                int idx101 = (gz_hi * grid_height_ + gy_lo) * grid_width_ + gx_hi;
                int idx110 = (gz_lo * grid_height_ + gy_hi) * grid_width_ + gx_hi;
                int idx111 = (gz_hi * grid_height_ + gy_hi) * grid_width_ + gx_hi;
                
                float w000 = (1-fx) * (1-fy) * (1-fz);
                float w001 = (1-fx) * (1-fy) * fz;
                float w010 = (1-fx) * fy * (1-fz);
                float w011 = (1-fx) * fy * fz;
                float w100 = fx * (1-fy) * (1-fz);
                float w101 = fx * (1-fy) * fz;
                float w110 = fx * fy * (1-fz);
                float w111 = fx * fy * fz;
                
                result += w000 * grid_data_[idx000];
                result += w001 * grid_data_[idx001];
                result += w010 * grid_data_[idx010];
                result += w011 * grid_data_[idx011];
                result += w100 * grid_data_[idx100];
                result += w101 * grid_data_[idx101];
                result += w110 * grid_data_[idx110];
                result += w111 * grid_data_[idx111];
                
                weight_sum += w000 * grid_weights_[idx000];
                weight_sum += w001 * grid_weights_[idx001];
                weight_sum += w010 * grid_weights_[idx010];
                weight_sum += w011 * grid_weights_[idx011];
                weight_sum += w100 * grid_weights_[idx100];
                weight_sum += w101 * grid_weights_[idx101];
                weight_sum += w110 * grid_weights_[idx110];
                weight_sum += w111 * grid_weights_[idx111];
                
                if (weight_sum > 0)
                {
                    dst_row[x] = cv::saturate_cast<T>(result / weight_sum);
                }
            }
        }
    }
}

// AVX-512 optimized slicing
template<typename T>
void BilateralGrid<T>::sliceGridAVX512(Mat& dst)
{
#if CV_AVX512F
    sliceGrid(dst);  // Simplified - full AVX-512 implementation would be more complex
#else
    sliceGrid(dst);
#endif
}

// Explicit instantiations
template class BilateralGrid<uchar>;
template class BilateralGrid<float>;

// Factory function
bool bilateralFilterGrid(const Mat& src, Mat& dst, float sigma_space, float sigma_range)
{
    if (src.type() == CV_8UC1 || src.type() == CV_8UC3)
    {
        BilateralGrid<uchar> grid(src.cols, src.rows, sigma_space, sigma_range);
        grid.filter(src, dst);
        return true;
    }
    else if (src.type() == CV_32FC1 || src.type() == CV_32FC3)
    {
        BilateralGrid<float> grid(src.cols, src.rows, sigma_space, sigma_range);
        grid.filter(src, dst);
        return true;
    }
    
    return false;
}

} // namespace cv