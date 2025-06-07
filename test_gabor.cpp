#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;
using namespace std::chrono;

void testGaborCorrectness() {
    cout << "Testing Gabor filter correctness..." << endl;
    
    // Test parameters
    Size ksize(21, 21);
    double sigma = 4.0;
    double theta = CV_PI / 4;  // 45 degrees
    double lambda = 10.0;
    double gamma = 0.5;
    double psi = CV_PI / 2;
    
    // Generate CV_32F kernel
    Mat kernel32f = getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, CV_32F);
    
    // Generate CV_64F kernel
    Mat kernel64f = getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, CV_64F);
    
    // Convert 64f to 32f for comparison
    Mat kernel64f_as_32f;
    kernel64f.convertTo(kernel64f_as_32f, CV_32F);
    
    // Compare the kernels
    double maxDiff = 0;
    for (int y = 0; y < ksize.height; y++) {
        for (int x = 0; x < ksize.width; x++) {
            float val32 = kernel32f.at<float>(y, x);
            float val64 = kernel64f_as_32f.at<float>(y, x);
            double diff = abs(val32 - val64);
            maxDiff = max(maxDiff, diff);
        }
    }
    
    cout << "Max difference between CV_32F and CV_64F kernels: " << maxDiff << endl;
    cout << "Kernel center value (CV_32F): " << kernel32f.at<float>(ksize.height/2, ksize.width/2) << endl;
    cout << "Kernel center value (CV_64F): " << kernel64f.at<double>(ksize.height/2, ksize.width/2) << endl;
    
    // Verify the kernel has expected properties
    double sum = cv::sum(kernel32f)[0];
    cout << "Kernel sum: " << sum << endl;
    
    if (maxDiff < 1e-6) {
        cout << "✓ Correctness test PASSED" << endl;
    } else {
        cout << "✗ Correctness test FAILED" << endl;
    }
    cout << endl;
}

void benchmarkGabor(int iterations = 1000) {
    cout << "Benchmarking Gabor filter performance..." << endl;
    
    // Test different kernel sizes
    vector<Size> sizes = {Size(11, 11), Size(21, 21), Size(31, 31), Size(41, 41)};
    
    // Common parameters
    double sigma = 4.0;
    double theta = CV_PI / 4;
    double lambda = 10.0;
    double gamma = 0.5;
    double psi = CV_PI / 2;
    
    for (const auto& ksize : sizes) {
        cout << "\nKernel size: " << ksize.width << "x" << ksize.height << endl;
        
        // Benchmark CV_32F
        {
            auto start = high_resolution_clock::now();
            for (int i = 0; i < iterations; i++) {
                Mat kernel = getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, CV_32F);
            }
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            double avg_time = duration.count() / (double)iterations;
            cout << "  CV_32F: " << fixed << setprecision(2) << avg_time << " μs per kernel" << endl;
        }
        
        // Benchmark CV_64F
        {
            auto start = high_resolution_clock::now();
            for (int i = 0; i < iterations; i++) {
                Mat kernel = getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, CV_64F);
            }
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            double avg_time = duration.count() / (double)iterations;
            cout << "  CV_64F: " << fixed << setprecision(2) << avg_time << " μs per kernel" << endl;
        }
    }
}

void visualizeGaborFilter() {
    cout << "\nGenerating Gabor filter visualization..." << endl;
    
    Size ksize(61, 61);
    double sigma = 8.0;
    double lambda = 20.0;
    double gamma = 0.5;
    double psi = 0;
    
    // Generate filters at different orientations
    vector<double> thetas = {0, CV_PI/6, CV_PI/4, CV_PI/3, CV_PI/2};
    
    for (size_t i = 0; i < thetas.size(); i++) {
        Mat kernel = getGaborKernel(ksize, sigma, thetas[i], lambda, gamma, psi, CV_32F);
        
        // Normalize for visualization
        Mat kernel_norm;
        normalize(kernel, kernel_norm, 0, 1, NORM_MINMAX);
        
        // Scale up for better visualization
        Mat kernel_vis;
        resize(kernel_norm, kernel_vis, Size(300, 300), 0, 0, INTER_NEAREST);
        
        // Display some statistics
        double minVal, maxVal;
        minMaxLoc(kernel, &minVal, &maxVal);
        cout << "Theta: " << thetas[i] * 180 / CV_PI << "° - Min: " << minVal << ", Max: " << maxVal << endl;
    }
}

int main(int argc, char** argv) {
    cout << "OpenCV Gabor Filter SIMD Optimization Test" << endl;
    cout << "==========================================" << endl;
    
    // Print build info
    cout << "\nOpenCV Build Information:" << endl;
    cout << "Version: " << CV_VERSION << endl;
    
    // Run tests
    testGaborCorrectness();
    benchmarkGabor();
    visualizeGaborFilter();
    
    cout << "\nAll tests completed!" << endl;
    return 0;
}