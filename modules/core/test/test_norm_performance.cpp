#include "test_precomp.hpp"
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

namespace opencv_test {
namespace {

using namespace cv;
using namespace std;

// Performance test for the norm function
// Tests specifically the cv::norm(src1, src2) case that had performance regressions
class NormPerformanceTest : public testing::Test {
protected:
    void SetUp() override {
        // Generate test matrices
        const int width = 1024;
        const int height = 768;
        img1.create(height, width, CV_32FC3);
        img2.create(height, width, CV_32FC3);
        
        rng.fill(img1, RNG::UNIFORM, 0, 1);
        rng.fill(img2, RNG::UNIFORM, 0, 1);
    }

    double timeNorm(const Mat& src1, const Mat& src2, int normType, int numIterations) {
        auto start = chrono::high_resolution_clock::now();
        double result = 0;
        for (int i = 0; i < numIterations; i++) {
            result = norm(src1, src2, normType);
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        // Return average time per iteration in milliseconds
        return duration.count() / numIterations;
    }

    double timeNormWithSubtraction(const Mat& src1, const Mat& src2, int normType, int numIterations) {
        auto start = chrono::high_resolution_clock::now();
        double result = 0;
        for (int i = 0; i < numIterations; i++) {
            result = norm(src1 - src2, normType);
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        // Return average time per iteration in milliseconds
        return duration.count() / numIterations;
    }

    Mat img1, img2;
    RNG rng;
};

// Test standard norm vs norm with subtraction for L2 norm (most common case)
TEST_F(NormPerformanceTest, L2NormPerformanceComparison) {
    const int numIterations = 20;
    
    double directTime = timeNorm(img1, img2, NORM_L2, numIterations);
    double subtractionTime = timeNormWithSubtraction(img1, img2, NORM_L2, numIterations);
    
    cout << "L2 norm performance comparison:" << endl;
    cout << "  norm(img1, img2) time: " << directTime << " ms" << endl;
    cout << "  norm(img1 - img2) time: " << subtractionTime << " ms" << endl;
    cout << "  Performance ratio: " << subtractionTime / directTime << endl;
    
    // The direct calculation should be similar or faster than the subtraction version
    // We use a relaxed threshold to account for system variability
    EXPECT_LE(directTime, subtractionTime * 1.5);
}

// Test standard norm vs norm with subtraction for L1 norm
TEST_F(NormPerformanceTest, L1NormPerformanceComparison) {
    const int numIterations = 20;
    
    double directTime = timeNorm(img1, img2, NORM_L1, numIterations);
    double subtractionTime = timeNormWithSubtraction(img1, img2, NORM_L1, numIterations);
    
    cout << "L1 norm performance comparison:" << endl;
    cout << "  norm(img1, img2) time: " << directTime << " ms" << endl;
    cout << "  norm(img1 - img2) time: " << subtractionTime << " ms" << endl;
    cout << "  Performance ratio: " << subtractionTime / directTime << endl;
    
    // The direct calculation should be similar or faster than the subtraction version
    // We use a relaxed threshold to account for system variability
    EXPECT_LE(directTime, subtractionTime * 1.5);
}

// Test standard norm vs norm with subtraction for INF norm
TEST_F(NormPerformanceTest, InfNormPerformanceComparison) {
    const int numIterations = 20;
    
    double directTime = timeNorm(img1, img2, NORM_INF, numIterations);
    double subtractionTime = timeNormWithSubtraction(img1, img2, NORM_INF, numIterations);
    
    cout << "INF norm performance comparison:" << endl;
    cout << "  norm(img1, img2) time: " << directTime << " ms" << endl;
    cout << "  norm(img1 - img2) time: " << subtractionTime << " ms" << endl;
    cout << "  Performance ratio: " << subtractionTime / directTime << endl;
    
    // The direct calculation should be similar or faster than the subtraction version
    // We use a relaxed threshold to account for system variability
    EXPECT_LE(directTime, subtractionTime * 1.5);
}

// Also verify that the results match for both methods
TEST_F(NormPerformanceTest, NormResultsCorrectnessComparison) {
    const double eps = 1e-5;
    
    // L2 norm results should match
    double resultDirect = norm(img1, img2, NORM_L2);
    double resultSubtraction = norm(img1 - img2, NORM_L2);
    EXPECT_NEAR(resultDirect, resultSubtraction, eps);
    
    // L1 norm results should match
    resultDirect = norm(img1, img2, NORM_L1);
    resultSubtraction = norm(img1 - img2, NORM_L1);
    EXPECT_NEAR(resultDirect, resultSubtraction, eps);
    
    // INF norm results should match
    resultDirect = norm(img1, img2, NORM_INF);
    resultSubtraction = norm(img1 - img2, NORM_INF);
    EXPECT_NEAR(resultDirect, resultSubtraction, eps);
}

}} // namespace