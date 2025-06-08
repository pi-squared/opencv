#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    // Create a synthetic image with a checkerboard pattern
    Mat image = Mat::zeros(640, 640, CV_32F);
    
    // Generate checkerboard pattern
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
                    Scalar(1.0), -1);
            }
        }
    }
    
    // Add some noise to make it more realistic
    Mat noise = Mat(image.size(), CV_32F);
    randn(noise, 0, 0.01);
    image += noise;
    
    // Generate corner points at checkerboard intersections
    vector<Point2f> corners;
    for (int i = 1; i < image.rows / block_size; i++)
    {
        for (int j = 1; j < image.cols / block_size; j++)
        {
            // Add slight random offset to simulate detection inaccuracy
            float x = j * block_size + (rand() % 11 - 5) * 0.1f;
            float y = i * block_size + (rand() % 11 - 5) * 0.1f;
            corners.push_back(Point2f(x, y));
        }
    }
    
    cout << "Number of corners: " << corners.size() << endl;
    
    // Parameters for cornerSubPix
    Size win_size(5, 5);
    Size zero_zone(-1, -1);
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.001);
    
    // Warm up
    vector<Point2f> refined_corners = corners;
    cornerSubPix(image, refined_corners, win_size, zero_zone, criteria);
    
    // Benchmark
    const int num_iterations = 1000;
    auto start = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++)
    {
        refined_corners = corners; // Reset corners
        cornerSubPix(image, refined_corners, win_size, zero_zone, criteria);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    cout << "Total time for " << num_iterations << " iterations: " 
         << duration.count() / 1000.0 << " ms" << endl;
    cout << "Average time per iteration: " 
         << duration.count() / (double)num_iterations << " microseconds" << endl;
    cout << "Average time per corner: " 
         << duration.count() / (double)(num_iterations * corners.size()) << " microseconds" << endl;
    
    // Verify results (check that refinement is reasonable)
    double total_displacement = 0.0;
    for (size_t i = 0; i < corners.size(); i++)
    {
        double dx = refined_corners[i].x - corners[i].x;
        double dy = refined_corners[i].y - corners[i].y;
        total_displacement += sqrt(dx*dx + dy*dy);
    }
    
    cout << "Average displacement: " << total_displacement / corners.size() << " pixels" << endl;
    
    return 0;
}