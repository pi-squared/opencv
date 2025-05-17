#include "test_precomp.hpp"

namespace opencv_test { namespace {

class Core_NormValidation : public testing::Test
{
protected:
    void SetUp() override
    {
        mat1 = cv::Mat::ones(10, 10, CV_8UC1);
        mat2 = cv::Mat::ones(10, 10, CV_8UC1) * 2;
    }

    cv::Mat mat1;
    cv::Mat mat2;
};

TEST_F(Core_NormValidation, ValidNormTypes)
{
    // Valid norm types should not throw exceptions
    EXPECT_NO_THROW(cv::norm(mat1, cv::NORM_L1));
    EXPECT_NO_THROW(cv::norm(mat1, cv::NORM_L2));
    EXPECT_NO_THROW(cv::norm(mat1, cv::NORM_INF));
    EXPECT_NO_THROW(cv::norm(mat1, cv::NORM_L2SQR));
    
    EXPECT_NO_THROW(cv::norm(mat1, mat2, cv::NORM_L1));
    EXPECT_NO_THROW(cv::norm(mat1, mat2, cv::NORM_L2));
    EXPECT_NO_THROW(cv::norm(mat1, mat2, cv::NORM_INF));
    EXPECT_NO_THROW(cv::norm(mat1, mat2, cv::NORM_L2SQR));

    // Valid norm types with flags should also work
    EXPECT_NO_THROW(cv::norm(mat1, mat2, cv::NORM_RELATIVE | cv::NORM_L2));
    EXPECT_NO_THROW(cv::norm(mat1, mat2, cv::NORM_RELATIVE | cv::NORM_L1));
    EXPECT_NO_THROW(cv::norm(mat1, mat2, cv::NORM_RELATIVE | cv::NORM_INF));
}

TEST_F(Core_NormValidation, InvalidNormTypes)
{
    // Invalid norm types should throw exceptions
    EXPECT_ANY_THROW(cv::norm(mat1, 3));  // 3 is not a valid normType
    EXPECT_ANY_THROW(cv::norm(mat1, -1)); // -1 is not a valid normType
    EXPECT_ANY_THROW(cv::norm(mat1, 100)); // 100 is not a valid normType
    
    EXPECT_ANY_THROW(cv::norm(mat1, mat2, 3));
    EXPECT_ANY_THROW(cv::norm(mat1, mat2, -1));
    EXPECT_ANY_THROW(cv::norm(mat1, mat2, 100));

    // Invalid flags should also fail
    EXPECT_ANY_THROW(cv::norm(mat1, mat2, cv::NORM_RELATIVE | 3));
    EXPECT_ANY_THROW(cv::norm(mat1, mat2, cv::NORM_RELATIVE | -1));
    EXPECT_ANY_THROW(cv::norm(mat1, mat2, cv::NORM_RELATIVE | 100));
}

}} // namespace