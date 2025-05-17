#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Test program to verify the fix for INTER_NEAREST_EXACT interpolation method
// This test reproduces the issue described in #27191 (precision issue with large scaling factors)
// Before the fix, there was a visible difference between INTER_NEAREST and INTER_NEAREST_EXACT
// After the fix, both methods should produce similar results for integer scaling

int main(int argc, char** argv)
{
    // Create a simple 10x10 test image similar to the one in the issue
    Mat img(10, 10, CV_8UC3);
    
    // Fill different regions with different colors
    img(Rect(0, 0, 5, 5)).setTo(Scalar(255, 0, 0));    // Blue
    img(Rect(5, 0, 5, 5)).setTo(Scalar(0, 255, 0));    // Green
    img(Rect(0, 5, 5, 5)).setTo(Scalar(0, 0, 255));    // Red
    img(Rect(5, 5, 5, 5)).setTo(Scalar(255, 255, 0));  // Yellow

    // Save original image
    imwrite("original.png", img);
    
    // Use large scaling factor (like the 192x in the issue)
    int scale = 100;
    Size newSize(img.cols * scale, img.rows * scale);
    
    // Test both interpolation methods
    Mat resultNN, resultNNExact;
    resize(img, resultNN, newSize, 0, 0, INTER_NEAREST);
    resize(img, resultNNExact, newSize, 0, 0, INTER_NEAREST_EXACT);
    
    // Save results
    imwrite("result_nearest.png", resultNN);
    imwrite("result_nearest_exact.png", resultNNExact);
    
    // Check if the results are similar
    Mat diff;
    absdiff(resultNN, resultNNExact, diff);
    double maxDiff = 0;
    minMaxLoc(diff, nullptr, &maxDiff);
    
    cout << "Maximum difference between INTER_NEAREST and INTER_NEAREST_EXACT: " << maxDiff << endl;
    cout << "With 32-bit fixed-point arithmetic, this should be 0 or close to 0 for integer scaling" << endl;
    
    // Save difference visualization (scaled to see it better)
    diff *= 50;  // Scale the difference to make it more visible
    imwrite("difference.png", diff);
    
    return 0;
}