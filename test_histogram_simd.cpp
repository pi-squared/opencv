#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <random>

using namespace cv;
using namespace std;
using namespace chrono;

void testHistogramPerformance() {
    // Create test images of various sizes
    vector<Size> sizes = {Size(640, 480), Size(1920, 1080), Size(3840, 2160)};
    
    for (const auto& size : sizes) {
        cout << "\nTesting histogram for image size: " << size.width << "x" << size.height << endl;
        
        // Create random test image
        Mat img(size, CV_8UC1);
        randu(img, Scalar(0), Scalar(256));
        
        // Test parameters
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        
        // Warm up
        Mat hist;
        for (int i = 0; i < 5; i++) {
            calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange);
        }
        
        // Performance test
        const int iterations = 100;
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange);
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        cout << "Average time per histogram: " << duration.count() / iterations << " microseconds" << endl;
        
        // Verify correctness (sum should equal total pixels)
        double sum = 0;
        for (int i = 0; i < histSize; i++) {
            sum += hist.at<float>(i);
        }
        cout << "Histogram sum: " << sum << " (expected: " << img.total() << ")" << endl;
        
        // Test with mask
        Mat mask(size, CV_8UC1);
        randu(mask, Scalar(0), Scalar(2)); // 50% chance of 0 or 1
        mask = mask * 255; // Convert to proper mask
        
        start = high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            calcHist(&img, 1, 0, mask, hist, 1, &histSize, &histRange);
        }
        end = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end - start);
        
        cout << "Average time per histogram (with mask): " << duration.count() / iterations << " microseconds" << endl;
    }
    
    // Test multi-channel histogram
    cout << "\nTesting multi-channel histogram:" << endl;
    Mat img3c(Size(1920, 1080), CV_8UC3);
    randu(img3c, Scalar(0, 0, 0), Scalar(256, 256, 256));
    
    // Split channels and compute histogram for first channel
    vector<Mat> channels;
    split(img3c, channels);
    
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        calcHist(&channels[0], 1, 0, Mat(), hist, 1, &histSize, &histRange);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    cout << "3-channel image, first channel histogram: " << duration.count() / 100 << " microseconds" << endl;
}

int main(int argc, char** argv) {
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "SIMD support: " << cv::checkHardwareSupport(CV_CPU_SSE2) << " (SSE2), "
         << cv::checkHardwareSupport(CV_CPU_AVX) << " (AVX), "
         << cv::checkHardwareSupport(CV_CPU_AVX2) << " (AVX2)" << endl;
    
    testHistogramPerformance();
    
    return 0;
}