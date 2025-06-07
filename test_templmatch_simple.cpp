#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

int main() {
    // Create test data
    Mat img = Mat::ones(640, 480, CV_8UC1) * 128;
    rectangle(img, Point(100, 100), Point(300, 300), Scalar(255), -1);
    circle(img, Point(200, 200), 50, Scalar(0), -1);
    
    Mat templ = img(Rect(150, 150, 100, 100)).clone();
    
    // Test normalized correlation coefficient method
    Mat result;
    
    cout << "Testing template matching optimization..." << endl;
    cout << "Image size: " << img.size() << endl;
    cout << "Template size: " << templ.size() << endl;
    
    // Warm up
    matchTemplate(img, templ, result, TM_CCOEFF_NORMED);
    
    // Time the operation
    auto start = high_resolution_clock::now();
    int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        matchTemplate(img, templ, result, TM_CCOEFF_NORMED);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    cout << "TM_CCOEFF_NORMED: " << duration.count() / iterations 
         << " microseconds (avg of " << iterations << " iterations)" << endl;
    
    // Find the best match
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    
    cout << "Best match at: " << maxLoc << " with score: " << maxVal << endl;
    cout << "Expected location: (150, 150)" << endl;
    
    // Test 3-channel image
    Mat img_color;
    cvtColor(img, img_color, COLOR_GRAY2BGR);
    Mat templ_color;
    cvtColor(templ, templ_color, COLOR_GRAY2BGR);
    
    start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matchTemplate(img_color, templ_color, result, TM_CCOEFF_NORMED);
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    
    cout << "\nTM_CCOEFF_NORMED (3-channel): " << duration.count() / iterations 
         << " microseconds (avg of " << iterations << " iterations)" << endl;
    
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    cout << "Best match at: " << maxLoc << " with score: " << maxVal << endl;
    
    return 0;
}