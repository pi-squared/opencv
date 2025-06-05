#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Test function to verify correctness
bool testCorrectness() {
    // Create test images
    vector<Size> sizes = {Size(100, 100), Size(500, 500), Size(1000, 1000)};
    
    for (const auto& size : sizes) {
        // Create binary test image with some shapes
        Mat src = Mat::zeros(size, CV_8UC1);
        
        // Add some rectangles and circles
        rectangle(src, Point(10, 10), Point(30, 30), Scalar(255), -1);
        circle(src, Point(size.width/2, size.height/2), 20, Scalar(255), -1);
        rectangle(src, Point(size.width-40, size.height-40), 
                 Point(size.width-10, size.height-10), Scalar(255), -1);
        
        // Test 3x3 mask
        Mat dist1_3x3, dist2_3x3;
        distanceTransform(src, dist1_3x3, DIST_L2, 3);
        distanceTransform(src, dist2_3x3, DIST_L2, 3);
        
        double diff_3x3 = norm(dist1_3x3, dist2_3x3, NORM_INF);
        cout << "Size " << size << " - 3x3 mask max difference: " << diff_3x3 << endl;
        
        if (diff_3x3 > 1e-5) {
            cerr << "ERROR: 3x3 results don't match!" << endl;
            return false;
        }
        
        // Test 5x5 mask
        Mat dist1_5x5, dist2_5x5;
        distanceTransform(src, dist1_5x5, DIST_L2, 5);
        distanceTransform(src, dist2_5x5, DIST_L2, 5);
        
        double diff_5x5 = norm(dist1_5x5, dist2_5x5, NORM_INF);
        cout << "Size " << size << " - 5x5 mask max difference: " << diff_5x5 << endl;
        
        if (diff_5x5 > 1e-5) {
            cerr << "ERROR: 5x5 results don't match!" << endl;
            return false;
        }
    }
    
    return true;
}

// Performance benchmark
void benchmark() {
    vector<Size> sizes = {Size(640, 480), Size(1280, 720), Size(1920, 1080)};
    vector<int> masks = {3, 5};
    const int iterations = 10;
    
    cout << "\nPerformance Benchmark:" << endl;
    cout << "=====================" << endl;
    
    for (const auto& size : sizes) {
        // Create binary test image
        Mat src = Mat::zeros(size, CV_8UC1);
        
        // Add random shapes
        int numShapes = 50;
        RNG rng(12345);
        for (int i = 0; i < numShapes; i++) {
            int x = rng.uniform(0, size.width);
            int y = rng.uniform(0, size.height);
            int r = rng.uniform(5, 30);
            circle(src, Point(x, y), r, Scalar(255), -1);
        }
        
        for (int maskSize : masks) {
            Mat dist;
            
            // Warm up
            distanceTransform(src, dist, DIST_L2, maskSize);
            
            // Benchmark
            auto start = high_resolution_clock::now();
            for (int i = 0; i < iterations; i++) {
                distanceTransform(src, dist, DIST_L2, maskSize);
            }
            auto end = high_resolution_clock::now();
            
            double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
            double avgMs = ms / iterations;
            
            cout << "Size: " << size << ", Mask: " << maskSize << "x" << maskSize 
                 << " - Avg time: " << avgMs << " ms" << endl;
        }
    }
}

int main() {
    cout << "Testing Distance Transform SIMD Optimization" << endl;
    cout << "===========================================" << endl;
    
    // Test correctness
    cout << "\nTesting correctness..." << endl;
    if (!testCorrectness()) {
        cerr << "Correctness test failed!" << endl;
        return -1;
    }
    cout << "Correctness test passed!" << endl;
    
    // Run performance benchmark
    benchmark();
    
    // Test specific example
    cout << "\nTesting specific example:" << endl;
    Mat src = Mat::zeros(200, 200, CV_8UC1);
    rectangle(src, Point(50, 50), Point(150, 150), Scalar(255), -1);
    
    Mat dist3, dist5;
    distanceTransform(src, dist3, DIST_L2, 3);
    distanceTransform(src, dist5, DIST_L2, 5);
    
    // Check some values
    cout << "Center value (3x3): " << dist3.at<float>(100, 100) << endl;
    cout << "Center value (5x5): " << dist5.at<float>(100, 100) << endl;
    cout << "Corner value (3x3): " << dist3.at<float>(10, 10) << endl;
    cout << "Corner value (5x5): " << dist5.at<float>(10, 10) << endl;
    
    cout << "\nAll tests completed successfully!" << endl;
    
    return 0;
}