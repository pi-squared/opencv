#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <immintrin.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main() {
    // Check if AVX-512 is supported
    bool avx512_supported = checkHardwareSupport(CV_CPU_AVX512_SKX);
    cout << "AVX-512 SKX supported: " << (avx512_supported ? "Yes" : "No") << endl;
    
    // Create a test image
    int width = 1920;
    int height = 1080;
    Mat src(height, width, CV_8UC3);
    randu(src, Scalar(0, 0, 0), Scalar(255, 255, 255));
    
    // Define affine transformation
    Point2f srcTri[3];
    srcTri[0] = Point2f(0.f, 0.f);
    srcTri[1] = Point2f(src.cols - 1.f, 0.f);
    srcTri[2] = Point2f(0.f, src.rows - 1.f);
    
    Point2f dstTri[3];
    dstTri[0] = Point2f(0.f, src.rows * 0.33f);
    dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
    dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);
    
    Mat warp_mat = getAffineTransform(srcTri, dstTri);
    Mat dst;
    
    // Warm up
    warpAffine(src, dst, warp_mat, src.size());
    
    // Benchmark
    const int iterations = 100;
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        warpAffine(src, dst, warp_mat, src.size(), INTER_LINEAR);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    cout << "Average time per warpAffine: " << duration.count() / iterations << " microseconds" << endl;
    cout << "Total time for " << iterations << " iterations: " << duration.count() / 1000.0 << " milliseconds" << endl;
    
    // Calculate throughput
    double megapixels = (width * height) / 1000000.0;
    double seconds = duration.count() / 1000000.0;
    double throughput = (megapixels * iterations) / seconds;
    cout << "Throughput: " << throughput << " Megapixels/second" << endl;
    
    // Save a sample output
    imwrite("warpaffine_test_output.jpg", dst);
    cout << "Test output saved to warpaffine_test_output.jpg" << endl;
    
    return 0;
}