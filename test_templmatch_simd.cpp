#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

int main() {
    // Create test image and template
    Mat img = imread("sample.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        // Create synthetic test data if image not found
        img = Mat::zeros(640, 480, CV_8UC1);
        rectangle(img, Point(100, 100), Point(300, 300), Scalar(255), -1);
        circle(img, Point(200, 200), 50, Scalar(0), -1);
    }
    
    // Create template from a region of the image
    Mat templ = img(Rect(150, 150, 100, 100)).clone();
    
    // Test all template matching methods
    int methods[] = {TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, 
                     TM_CCOEFF, TM_CCOEFF_NORMED};
    string method_names[] = {"TM_SQDIFF", "TM_SQDIFF_NORMED", "TM_CCORR", 
                            "TM_CCORR_NORMED", "TM_CCOEFF", "TM_CCOEFF_NORMED"};
    
    cout << "Testing template matching with SIMD optimization..." << endl;
    cout << "Image size: " << img.size() << endl;
    cout << "Template size: " << templ.size() << endl << endl;
    
    for (int i = 0; i < 6; i++) {
        Mat result;
        
        // Warm up
        matchTemplate(img, templ, result, methods[i]);
        
        // Time the operation
        auto start = high_resolution_clock::now();
        int iterations = 10;
        
        for (int j = 0; j < iterations; j++) {
            matchTemplate(img, templ, result, methods[i]);
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        
        cout << method_names[i] << ": " 
             << duration.count() / iterations << " microseconds (avg of " 
             << iterations << " iterations)" << endl;
        
        // Find the best match location
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        
        Point matchLoc;
        if (methods[i] == TM_SQDIFF || methods[i] == TM_SQDIFF_NORMED) {
            matchLoc = minLoc;
        } else {
            matchLoc = maxLoc;
        }
        
        cout << "  Best match at: " << matchLoc << endl;
        cout << "  Result size: " << result.size() << endl << endl;
    }
    
    // Test with multi-channel image
    cout << "Testing with 3-channel image..." << endl;
    Mat img_color;
    cvtColor(img, img_color, COLOR_GRAY2BGR);
    Mat templ_color;
    cvtColor(templ, templ_color, COLOR_GRAY2BGR);
    
    for (int i = 0; i < 6; i++) {
        Mat result;
        
        auto start = high_resolution_clock::now();
        matchTemplate(img_color, templ_color, result, methods[i]);
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<microseconds>(end - start);
        cout << method_names[i] << " (3-channel): " 
             << duration.count() << " microseconds" << endl;
    }
    
    return 0;
}