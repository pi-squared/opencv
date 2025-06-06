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

#ifndef OPENCV_IMGPROC_BILATERAL_GRID_HPP
#define OPENCV_IMGPROC_BILATERAL_GRID_HPP

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {

// Bilateral Grid acceleration for bilateral filtering
// Based on "Real-time edge-aware image processing with the bilateral grid" by Chen et al.
template<typename T>
class BilateralGrid
{
public:
    BilateralGrid(int width, int height, float sigma_space, float sigma_range);
    ~BilateralGrid();
    
    void filter(const Mat& src, Mat& dst);
    
private:
    int width_, height_;
    float sigma_space_, sigma_range_;
    float inv_sigma_space_, inv_sigma_range_;
    int grid_width_, grid_height_, grid_depth_;
    float* grid_data_;
    float* grid_weights_;
    
    void constructGrid(const Mat& src);
    void blurGrid();
    void sliceGrid(Mat& dst);
    
    // AVX-512 optimized functions
    void constructGridAVX512(const Mat& src);
    void blurGridAVX512();
    void sliceGridAVX512(Mat& dst);
};

// Factory function to create bilateral grid filter
bool bilateralFilterGrid(const Mat& src, Mat& dst, float sigma_space, float sigma_range);

} // namespace cv

#endif // OPENCV_IMGPROC_BILATERAL_GRID_HPP