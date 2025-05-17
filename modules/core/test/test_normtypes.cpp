#include "test_precomp.hpp"

using namespace cv;
using namespace std;

TEST(Core_Norm, InvalidNormType)
{
    Mat mat1 = Mat::ones(10, 10, CV_8UC1);
    Mat mat2 = Mat::ones(10, 10, CV_8UC1) * 2;
    
    // Test with invalid normType values
    EXPECT_ANY_THROW(norm(mat1, 3)); // 3 is not a valid normType
    EXPECT_ANY_THROW(norm(mat1, -1)); // -1 is not a valid normType
    EXPECT_ANY_THROW(norm(mat1, 100)); // 100 is not a valid normType
    
    // Test with invalid normType values for norm(Mat, Mat, ...)
    EXPECT_ANY_THROW(norm(mat1, mat2, 3)); // 3 is not a valid normType
    EXPECT_ANY_THROW(norm(mat1, mat2, -1)); // -1 is not a valid normType
    EXPECT_ANY_THROW(norm(mat1, mat2, 100)); // 100 is not a valid normType
    
    // Valid normType values should work fine
    EXPECT_NO_THROW(norm(mat1, NORM_L1));
    EXPECT_NO_THROW(norm(mat1, NORM_L2));
    EXPECT_NO_THROW(norm(mat1, NORM_INF));
    EXPECT_NO_THROW(norm(mat1, NORM_L2SQR));
    
    EXPECT_NO_THROW(norm(mat1, mat2, NORM_L1));
    EXPECT_NO_THROW(norm(mat1, mat2, NORM_L2));
    EXPECT_NO_THROW(norm(mat1, mat2, NORM_INF));
    EXPECT_NO_THROW(norm(mat1, mat2, NORM_L2SQR));
}