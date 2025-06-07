#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;
using namespace std::chrono;

void test_remap_performance(const string& type_name, int cv_type, Size size, int iterations = 100) {
    cout << "\nTesting " << type_name << " image (" << size.width << "x" << size.height << "):" << endl;
    
    // Create test image
    Mat src(size, cv_type);
    randu(src, 0, (cv_type == CV_8UC1 || cv_type == CV_8UC3) ? 255 : 
                  (cv_type == CV_16UC1 || cv_type == CV_16UC3) ? 65535 : 1.0);
    
    // Create maps for a simple rotation + scale transform
    Mat map_x(size, CV_32FC1), map_y(size, CV_32FC1);
    Point2f center(size.width / 2.0f, size.height / 2.0f);
    double angle = 15.0; // 15 degree rotation
    double scale = 1.1;  // 10% zoom
    
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            float dx = x - center.x;
            float dy = y - center.y;
            float cos_a = cos(angle * CV_PI / 180.0) / scale;
            float sin_a = sin(angle * CV_PI / 180.0) / scale;
            
            map_x.at<float>(y, x) = center.x + dx * cos_a - dy * sin_a;
            map_y.at<float>(y, x) = center.y + dx * sin_a + dy * cos_a;
        }
    }
    
    Mat dst;
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT);
    }
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT);
    }
    auto end = high_resolution_clock::now();
    
    double duration = duration_cast<microseconds>(end - start).count() / 1000.0;
    double avg_time = duration / iterations;
    
    cout << "  Average time per remap: " << avg_time << " ms" << endl;
    cout << "  Throughput: " << (size.width * size.height * iterations / duration / 1000.0) << " Mpixels/sec" << endl;
    
    // Verify correctness by checking a few pixels
    cout << "  Sample output values: ";
    if (cv_type == CV_32FC1 || cv_type == CV_32FC3) {
        for (int i = 0; i < min(3, (int)dst.total()); i++) {
            cout << dst.at<float>(i) << " ";
        }
    } else if (cv_type == CV_16UC1 || cv_type == CV_16UC3) {
        for (int i = 0; i < min(3, (int)dst.total()); i++) {
            cout << (int)dst.at<ushort>(i) << " ";
        }
    } else {
        for (int i = 0; i < min(3, (int)dst.total()); i++) {
            cout << (int)dst.at<uchar>(i) << " ";
        }
    }
    cout << endl;
}

int main() {
    cout << "OpenCV Remap SIMD Optimization Test" << endl;
    cout << "===================================" << endl;
    
    // Get OpenCV build information
    cout << "\nOpenCV version: " << CV_VERSION << endl;
    cout << "SIMD optimizations available:" << endl;
    
#ifdef CV_SIMD
    cout << "  CV_SIMD: Yes" << endl;
    cout << "  CV_SIMD_WIDTH: " << CV_SIMD_WIDTH << endl;
#else
    cout << "  CV_SIMD: No" << endl;
#endif
    
    // Test different image types and sizes
    vector<Size> sizes = {Size(640, 480), Size(1280, 720), Size(1920, 1080)};
    
    for (const auto& size : sizes) {
        test_remap_performance("8-bit grayscale", CV_8UC1, size);
        test_remap_performance("8-bit RGB", CV_8UC3, size);
        test_remap_performance("16-bit grayscale", CV_16UC1, size);
        test_remap_performance("16-bit RGB", CV_16UC3, size);
        test_remap_performance("32-bit float grayscale", CV_32FC1, size);
        test_remap_performance("32-bit float RGB", CV_32FC3, size);
    }
    
    return 0;
}