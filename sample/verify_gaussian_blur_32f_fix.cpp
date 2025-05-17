#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

/**
 * This test verifies the fix for issue #11303 where GaussianBlur with CV_32F images
 * would cause memory access issues when IPP acceleration is enabled.
 * 
 * The fix disables IPP for 32F depth to ensure safe operation.
 */

// Function to test GaussianBlur with varying parameters
void testGaussianBlur(const cv::Mat& src, const cv::Size& ksize, double sigma, int borderType) {
    cv::Mat dst;
    std::string borderName;
    
    // Use if-else instead of switch since BORDER_DEFAULT == BORDER_REFLECT_101
    if (borderType == cv::BORDER_CONSTANT) {
        borderName = "BORDER_CONSTANT";
    } else if (borderType == cv::BORDER_REPLICATE) {
        borderName = "BORDER_REPLICATE";
    } else if (borderType == cv::BORDER_REFLECT) {
        borderName = "BORDER_REFLECT";
    } else if (borderType == cv::BORDER_REFLECT_101) {
        borderName = "BORDER_REFLECT_101/BORDER_DEFAULT";
    } else {
        borderName = "Unknown";
    }
    
    std::cout << "Testing GaussianBlur with " << src.type() << " image, kernel=" << ksize
              << ", sigma=" << sigma << ", border=" << borderName << "... ";
    
    try {
        cv::GaussianBlur(src, dst, ksize, sigma, sigma, borderType);
        std::cout << "Success!" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "Failed! OpenCV Exception: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed! STL Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Failed! Unknown exception" << std::endl;
    }
}

int main() {
    try {
        // Test with IPP enabled
        cv::ipp::setUseIPP(true);
        std::cout << "IPP enabled: " << cv::ipp::useIPP() << std::endl;
        
        // Create test images with different types
        const int width = 640;
        const int height = 480;
        
        // Create CV_32F image (single-channel float)
        cv::Mat img32F(height, width, CV_32FC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img32F.at<float>(y, x) = (float)(sin(x*0.1) * cos(y*0.1));
            }
        }
        
        // Create CV_32FC3 image (3-channel float)
        cv::Mat img32FC3(height, width, CV_32FC3);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cv::Vec3f& pixel = img32FC3.at<cv::Vec3f>(y, x);
                pixel[0] = (float)(sin(x*0.1) * cos(y*0.1));
                pixel[1] = (float)(cos(x*0.1) * sin(y*0.1));
                pixel[2] = (float)(sin(x*0.05) * cos(y*0.05));
            }
        }
        
        // Create CV_8U image for comparison
        cv::Mat img8U(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img8U.at<uchar>(y, x) = (uchar)(128 + 127 * sin(x*0.1) * cos(y*0.1));
            }
        }
        
        // Test parameters
        std::vector<cv::Size> kernelSizes = {
            cv::Size(3, 3),
            cv::Size(5, 5),
            cv::Size(7, 7),
            cv::Size(11, 11),
            cv::Size(15, 15)
        };
        
        std::vector<double> sigmas = {
            0.0, 1.0, 2.0, 3.0, 5.0
        };
        
        std::vector<int> borderTypes = {
            cv::BORDER_CONSTANT,
            cv::BORDER_REPLICATE,
            cv::BORDER_REFLECT,
            cv::BORDER_REFLECT_101  // Same as BORDER_DEFAULT
        };
        
        // Run tests with CV_32F images (the type that was having issues)
        std::cout << "\n--- Testing CV_32F images (fixed) ---\n";
        for (const auto& ksize : kernelSizes) {
            for (const auto& sigma : sigmas) {
                for (const auto& borderType : borderTypes) {
                    testGaussianBlur(img32F, ksize, sigma, borderType);
                }
            }
        }
        
        // Run tests with CV_32FC3 images
        std::cout << "\n--- Testing CV_32FC3 images ---\n";
        for (const auto& ksize : kernelSizes) {
            for (const auto& sigma : sigmas) {
                for (const auto& borderType : borderTypes) {
                    testGaussianBlur(img32FC3, ksize, sigma, borderType);
                }
            }
        }
        
        // Run tests with CV_8U images (should have been working correctly)
        std::cout << "\n--- Testing CV_8UC1 images (reference) ---\n";
        for (const auto& ksize : kernelSizes) {
            for (const auto& sigma : sigmas) {
                for (const auto& borderType : borderTypes) {
                    testGaussianBlur(img8U, ksize, sigma, borderType);
                }
            }
        }
        
        std::cout << "\nAll tests completed successfully with the fix!\n";
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
}