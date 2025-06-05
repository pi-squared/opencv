// Standalone test for distance transform SIMD optimization
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <climits>

using namespace std;
using namespace std::chrono;

// Simulate the distance transform algorithm
const int DIST_SHIFT = 16;
#define CV_FLT_TO_FIX(x,n) ((int)round((x)*(1<<(n))))

struct Size {
    int width, height;
    Size(int w, int h) : width(w), height(h) {}
};

// Original scalar version
void distanceTransform_3x3_scalar(const unsigned char* src, unsigned int* temp, float* dist,
                                  Size size, int srcstep, int step, int dststep,
                                  const float* metrics) {
    const int BORDER = 1;
    const unsigned int HV_DIST = CV_FLT_TO_FIX(metrics[0], DIST_SHIFT);
    const unsigned int DIAG_DIST = CV_FLT_TO_FIX(metrics[1], DIST_SHIFT);
    const unsigned int DIST_MAX = UINT_MAX - DIAG_DIST;
    const float scale = 1.f/(1 << DIST_SHIFT);
    
    // Initialize borders
    for(int i = 0; i < BORDER; i++) {
        for(int j = 0; j < size.width + 2*BORDER; j++) {
            temp[i * step + j] = DIST_MAX;
            temp[(size.height + BORDER + i) * step + j] = DIST_MAX;
        }
    }
    
    // Forward pass
    unsigned int* tmp = temp + BORDER * step + BORDER;
    const unsigned char* s = src;
    
    for(int i = 0; i < size.height; i++) {
        for(int j = 0; j < BORDER; j++)
            tmp[-j-1] = tmp[size.width + j] = DIST_MAX;
            
        for(int j = 0; j < size.width; j++) {
            if(!s[j])
                tmp[j] = 0;
            else {
                unsigned int t0 = tmp[j-step-1] + DIAG_DIST;
                unsigned int t = tmp[j-step] + HV_DIST;
                if(t0 > t) t0 = t;
                t = tmp[j-step+1] + DIAG_DIST;
                if(t0 > t) t0 = t;
                t = tmp[j-1] + HV_DIST;
                if(t0 > t) t0 = t;
                tmp[j] = (t0 > DIST_MAX) ? DIST_MAX : t0;
            }
        }
        tmp += step;
        s += srcstep;
    }
    
    // Backward pass
    float* d = dist + (size.height - 1) * dststep;
    tmp = temp + (size.height + BORDER - 1) * step + BORDER;
    
    for(int i = size.height - 1; i >= 0; i--) {
        for(int j = size.width - 1; j >= 0; j--) {
            unsigned int t0 = tmp[j];
            if(t0 > HV_DIST) {
                unsigned int t = tmp[j+step+1] + DIAG_DIST;
                if(t0 > t) t0 = t;
                t = tmp[j+step] + HV_DIST;
                if(t0 > t) t0 = t;
                t = tmp[j+step-1] + DIAG_DIST;
                if(t0 > t) t0 = t;
                t = tmp[j+1] + HV_DIST;
                if(t0 > t) t0 = t;
                tmp[j] = t0;
            }
            d[j] = (float)(t0 * scale);
        }
        d -= dststep;
        tmp -= step;
    }
}

// Test correctness
bool testCorrectness() {
    cout << "Testing correctness of distance transform..." << endl;
    
    vector<Size> sizes = {Size(50, 50), Size(100, 100), Size(200, 200)};
    float metrics[2] = {1.0f, 1.4142f}; // L2 approximation
    
    for(const auto& size : sizes) {
        int totalSize = size.width * size.height;
        vector<unsigned char> src(totalSize, 0);
        
        // Create test pattern - rectangle in center
        int x1 = size.width / 4, y1 = size.height / 4;
        int x2 = 3 * size.width / 4, y2 = 3 * size.height / 4;
        
        for(int y = y1; y < y2; y++) {
            for(int x = x1; x < x2; x++) {
                src[y * size.width + x] = 255;
            }
        }
        
        // Allocate buffers
        int step = size.width + 2;
        vector<unsigned int> temp((size.height + 2) * step);
        vector<float> dist(totalSize);
        
        // Run distance transform
        distanceTransform_3x3_scalar(src.data(), temp.data(), dist.data(),
                                    size, size.width, step, size.width, metrics);
        
        // Check some expected values
        float centerVal = dist[size.height/2 * size.width + size.width/2];
        float cornerVal = dist[0];
        
        cout << "Size " << size.width << "x" << size.height 
             << " - Center: " << centerVal << ", Corner: " << cornerVal << endl;
        
        // Basic sanity checks
        if(centerVal < 0 || cornerVal < 0) {
            cerr << "ERROR: Negative distance values!" << endl;
            return false;
        }
        
        // Center should have larger distance than edges for our rectangle
        if(centerVal < 10) {
            cerr << "ERROR: Center distance too small!" << endl;
            return false;
        }
    }
    
    return true;
}

// Performance benchmark
void benchmark() {
    cout << "\nPerformance Benchmark:" << endl;
    cout << "=====================" << endl;
    
    vector<Size> sizes = {Size(640, 480), Size(1280, 720), Size(1920, 1080)};
    float metrics[2] = {1.0f, 1.4142f};
    const int iterations = 10;
    
    for(const auto& size : sizes) {
        int totalSize = size.width * size.height;
        vector<unsigned char> src(totalSize);
        
        // Create random binary image
        for(int i = 0; i < totalSize; i++) {
            src[i] = (rand() % 100 < 80) ? 255 : 0; // 80% foreground
        }
        
        // Allocate buffers
        int step = size.width + 2;
        vector<unsigned int> temp((size.height + 2) * step);
        vector<float> dist(totalSize);
        
        // Warm up
        distanceTransform_3x3_scalar(src.data(), temp.data(), dist.data(),
                                    size, size.width, step, size.width, metrics);
        
        // Benchmark
        auto start = high_resolution_clock::now();
        for(int i = 0; i < iterations; i++) {
            distanceTransform_3x3_scalar(src.data(), temp.data(), dist.data(),
                                        size, size.width, step, size.width, metrics);
        }
        auto end = high_resolution_clock::now();
        
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double avgMs = ms / iterations;
        double pixelsPerSec = (size.width * size.height) / (avgMs / 1000.0) / 1e6;
        
        cout << "Size: " << size.width << "x" << size.height 
             << " - Avg time: " << avgMs << " ms"
             << " (" << pixelsPerSec << " Mpixels/sec)" << endl;
    }
}

int main() {
    cout << "Distance Transform SIMD Optimization Test" << endl;
    cout << "========================================" << endl;
    
    if(!testCorrectness()) {
        cerr << "Correctness test failed!" << endl;
        return -1;
    }
    cout << "Correctness test passed!" << endl;
    
    benchmark();
    
    cout << "\nNote: This is a simplified test. Full SIMD benefits would be seen in actual OpenCV build." << endl;
    
    return 0;
}