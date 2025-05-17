#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Core_Normalize, hamming_support)
{
    // Create matrices of different types
    Mat src_u8(5, 5, CV_8UC1, Scalar(5));
    Mat src_f32(5, 5, CV_32FC1, Scalar(1.0f));
    Mat dst;
    
    // Test 1: Normalize with NORM_HAMMING on CV_8U should work
    EXPECT_NO_THROW(normalize(src_u8, dst, 1.0, 0.0, NORM_HAMMING));
    
    // Test 2: Normalize with NORM_HAMMING2 on CV_8U should work
    EXPECT_NO_THROW(normalize(src_u8, dst, 1.0, 0.0, NORM_HAMMING2));
    
    // Test 3: Normalize with NORM_HAMMING on CV_32F should throw exception
    EXPECT_ANY_THROW(normalize(src_f32, dst, 1.0, 0.0, NORM_HAMMING));
    
    // Test 4: Normalize with NORM_HAMMING2 on CV_32F should throw exception
    EXPECT_ANY_THROW(normalize(src_f32, dst, 1.0, 0.0, NORM_HAMMING2));
}

}} // namespace