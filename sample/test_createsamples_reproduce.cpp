#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// Function to read vec file header and extract sample data
void readVecFile(const char* vecfile, std::vector<unsigned char>& data) {
    FILE* file = fopen(vecfile, "rb");
    if (!file) {
        std::cerr << "Unable to open vec file: " << vecfile << std::endl;
        return;
    }
    
    int count, vecsize;
    short tmp;
    
    // Read header
    fread(&count, sizeof(count), 1, file);
    fread(&vecsize, sizeof(vecsize), 1, file);
    fread(&tmp, sizeof(tmp), 1, file);
    fread(&tmp, sizeof(tmp), 1, file);
    
    std::cout << "Vec file contains " << count << " samples of size " << vecsize << std::endl;
    
    // Read data (only a sample)
    int maxSamples = std::min(count, 5);
    data.resize(maxSamples * vecsize);
    
    for (int i = 0; i < maxSamples; i++) {
        unsigned char marker;
        fread(&marker, sizeof(marker), 1, file);
        
        std::vector<short> sampleData(vecsize);
        fread(sampleData.data(), sizeof(short), vecsize, file);
        
        // Copy data
        for (int j = 0; j < vecsize; j++) {
            data[i * vecsize + j] = static_cast<unsigned char>(sampleData[j]);
        }
    }
    
    fclose(file);
}

// Calculate a simple checksum for the data
long calculateChecksum(const std::vector<unsigned char>& data) {
    long sum = 0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i] * (i + 1);
    }
    return sum;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <seed1> <seed2>" << std::endl;
        return 1;
    }
    
    int seed1 = atoi(argv[1]);
    int seed2 = atoi(argv[2]);
    
    // Generate sample data with first seed
    std::cout << "Generating samples with seed " << seed1 << std::endl;
    cv::setRNGSeed(seed1);
    system("mkdir -p /tmp/test_opencv_seed1");
    system("opencv_createsamples -img /home/site/wwwroot/ocv/opencv/data/haarcascades/haarcascade_frontalface_alt.xml -bg /dev/null -vec /tmp/test_opencv_seed1/samples.vec -num 10 -w 24 -h 24 -rngseed 12345");
    
    // Generate sample data with second seed
    std::cout << "Generating samples with seed " << seed2 << std::endl;
    cv::setRNGSeed(seed2);
    system("mkdir -p /tmp/test_opencv_seed2");
    system("opencv_createsamples -img /home/site/wwwroot/ocv/opencv/data/haarcascades/haarcascade_frontalface_alt.xml -bg /dev/null -vec /tmp/test_opencv_seed2/samples.vec -num 10 -w 24 -h 24 -rngseed 12345");
    
    // Read and compare the generated data
    std::vector<unsigned char> data1, data2;
    readVecFile("/tmp/test_opencv_seed1/samples.vec", data1);
    readVecFile("/tmp/test_opencv_seed2/samples.vec", data2);
    
    long checksum1 = calculateChecksum(data1);
    long checksum2 = calculateChecksum(data2);
    
    std::cout << "Checksum with seed " << seed1 << ": " << checksum1 << std::endl;
    std::cout << "Checksum with seed " << seed2 << ": " << checksum2 << std::endl;
    
    if (checksum1 == checksum2 && !data1.empty() && !data2.empty()) {
        std::cout << "SUCCESS: Samples are reproducible with the same seed!" << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Samples are not reproducible with the same seed!" << std::endl;
        return 1;
    }
}