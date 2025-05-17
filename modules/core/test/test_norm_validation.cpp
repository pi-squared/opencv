#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Core_Norm, invalid_normType_validation)
{
    // Test case for detecting invalid normType values
    // The purpose of this test is to verify that invalid normType values 
    // are properly detected and appropriate exceptions are thrown
    // instead of causing memory access issues or undefined behavior.
    
    Mat src1 = Mat::ones(10, 10, CV_8UC1);
    Mat src2 = Mat::ones(10, 10, CV_8UC1) * 2;
    
    // Valid normType values for reference
    EXPECT_NO_THROW(norm(src1, NORM_INF));
    EXPECT_NO_THROW(norm(src1, NORM_L1));
    EXPECT_NO_THROW(norm(src1, NORM_L2));
    EXPECT_NO_THROW(norm(src1, NORM_L2SQR));
    EXPECT_NO_THROW(norm(src1, NORM_HAMMING));
    EXPECT_NO_THROW(norm(src1, NORM_HAMMING2));
    
    EXPECT_NO_THROW(norm(src1, src2, NORM_INF));
    EXPECT_NO_THROW(norm(src1, src2, NORM_L1));
    EXPECT_NO_THROW(norm(src1, src2, NORM_L2));
    EXPECT_NO_THROW(norm(src1, src2, NORM_L2SQR));
    EXPECT_NO_THROW(norm(src1, src2, NORM_HAMMING));
    EXPECT_NO_THROW(norm(src1, src2, NORM_HAMMING2));
    
    // Invalid normType values should throw exceptions
    // Test with normType values that would cause array index out of bounds
    
    // Single source norm
    EXPECT_THROW(norm(src1, 8), cv::Exception);    // 8 >> 1 = 4, out of bounds for normTab
    EXPECT_THROW(norm(src1, -2), cv::Exception);   // -2 >> 1 = -1, out of bounds for normTab
    EXPECT_THROW(norm(src1, 100), cv::Exception);  // 100 >> 1 = 50, out of bounds for normTab
    
    // Dual source norm
    EXPECT_THROW(norm(src1, src2, 8), cv::Exception);
    EXPECT_THROW(norm(src1, src2, -2), cv::Exception);
    EXPECT_THROW(norm(src1, src2, 100), cv::Exception);
}

}} // namespace