#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Create a test image with corners
    Mat image = Mat::zeros(640, 640, CV_8UC1);
    
    // Draw a checkerboard pattern
    int block_size = 40;
    for (int i = 0; i < image.rows / block_size; i++)
    {
        for (int j = 0; j < image.cols / block_size; j++)
        {
            if ((i + j) % 2 == 0)
            {
                rectangle(image, 
                    Point(j * block_size, i * block_size),
                    Point((j + 1) * block_size - 1, (i + 1) * block_size - 1),
                    Scalar(255), -1);
            }
        }
    }
    
    // Convert to float for processing
    Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0/255.0);
    
    // Find corners using Harris detector
    Mat corners;
    cornerHarris(img_float, corners, 2, 3, 0.04);
    
    // Find local maxima
    Mat corners_norm;
    normalize(corners, corners_norm, 0, 255, NORM_MINMAX, CV_32FC1);
    
    // Threshold and find corner points
    vector<Point2f> corner_points;
    float threshold = 100.0f;
    
    for (int i = 1; i < corners_norm.rows - 1; i++)
    {
        for (int j = 1; j < corners_norm.cols - 1; j++)
        {
            float val = corners_norm.at<float>(i, j);
            if (val > threshold)
            {
                // Check if local maximum
                bool is_max = true;
                for (int di = -1; di <= 1 && is_max; di++)
                {
                    for (int dj = -1; dj <= 1 && is_max; dj++)
                    {
                        if (di == 0 && dj == 0) continue;
                        if (corners_norm.at<float>(i + di, j + dj) >= val)
                            is_max = false;
                    }
                }
                
                if (is_max)
                {
                    corner_points.push_back(Point2f(j, i));
                }
            }
        }
    }
    
    cout << "Found " << corner_points.size() << " corners" << endl;
    
    // Make copies for comparison
    vector<Point2f> refined_corners = corner_points;
    
    // Set refinement parameters
    Size win_size(5, 5);
    Size zero_zone(-1, -1);
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.001);
    
    // Time the corner subpixel refinement
    const int num_iterations = 100;
    auto start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < num_iterations; iter++)
    {
        refined_corners = corner_points; // Reset corners
        cornerSubPix(img_float, refined_corners, win_size, zero_zone, criteria);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    cout << "Average time per iteration: " << duration.count() / num_iterations << " microseconds" << endl;
    cout << "Time per corner: " << duration.count() / (num_iterations * corner_points.size()) << " microseconds" << endl;
    
    // Check refinement results
    double total_displacement = 0.0;
    double max_displacement = 0.0;
    
    for (size_t i = 0; i < corner_points.size(); i++)
    {
        double dx = refined_corners[i].x - corner_points[i].x;
        double dy = refined_corners[i].y - corner_points[i].y;
        double displacement = sqrt(dx*dx + dy*dy);
        
        total_displacement += displacement;
        max_displacement = max(max_displacement, displacement);
    }
    
    cout << "Average displacement: " << total_displacement / corner_points.size() << " pixels" << endl;
    cout << "Maximum displacement: " << max_displacement << " pixels" << endl;
    
    // Visualize results
    Mat vis;
    cvtColor(image, vis, COLOR_GRAY2BGR);
    
    // Draw original corners in red
    for (const auto& pt : corner_points)
    {
        circle(vis, Point(pt.x, pt.y), 3, Scalar(0, 0, 255), -1);
    }
    
    // Draw refined corners in green
    for (const auto& pt : refined_corners)
    {
        circle(vis, Point(pt.x, pt.y), 2, Scalar(0, 255, 0), -1);
    }
    
    imwrite("cornersubpix_test.png", vis);
    cout << "Results saved to cornersubpix_test.png" << endl;
    
    return 0;
}