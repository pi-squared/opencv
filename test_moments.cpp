#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main()
{
    // Create test images of different sizes
    std::vector<cv::Size> sizes = {
        cv::Size(640, 480),
        cv::Size(1280, 720),
        cv::Size(1920, 1080)
    };
    
    for (const auto& size : sizes)
    {
        std::cout << "\nTesting image size: " << size.width << "x" << size.height << std::endl;
        
        // Create a test image with gradient pattern
        cv::Mat img(size, CV_8UC1);
        for (int y = 0; y < size.height; y++)
        {
            uchar* ptr = img.ptr<uchar>(y);
            for (int x = 0; x < size.width; x++)
            {
                ptr[x] = static_cast<uchar>((x + y) % 256);
            }
        }
        
        // Warm up
        cv::Moments m;
        for (int i = 0; i < 10; i++)
        {
            m = cv::moments(img, false);
        }
        
        // Benchmark
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++)
        {
            m = cv::moments(img, false);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_time = duration.count() / static_cast<double>(iterations);
        std::cout << "Average time per iteration: " << avg_time << " us" << std::endl;
        std::cout << "Throughput: " << (size.width * size.height) / (avg_time * 1000.0) << " Mpixels/s" << std::endl;
        
        // Print some moment values to verify correctness
        std::cout << "m00: " << m.m00 << std::endl;
        std::cout << "m10: " << m.m10 << std::endl;
        std::cout << "m01: " << m.m01 << std::endl;
        std::cout << "m20: " << m.m20 << std::endl;
        std::cout << "Centroid: (" << m.m10/m.m00 << ", " << m.m01/m.m00 << ")" << std::endl;
    }
    
    // Test with 16-bit image
    std::cout << "\nTesting 16-bit image (1280x720):" << std::endl;
    cv::Mat img16(720, 1280, CV_16UC1);
    for (int y = 0; y < img16.rows; y++)
    {
        ushort* ptr = img16.ptr<ushort>(y);
        for (int x = 0; x < img16.cols; x++)
        {
            ptr[x] = static_cast<ushort>((x + y) * 100);
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Moments m16 = cv::moments(img16, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Time: " << duration.count() << " us" << std::endl;
    std::cout << "m00: " << m16.m00 << std::endl;
    std::cout << "Centroid: (" << m16.m10/m16.m00 << ", " << m16.m01/m16.m00 << ")" << std::endl;
    
    return 0;
}