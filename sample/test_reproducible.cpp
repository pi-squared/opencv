#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Test if the RNG is reproducible with the same seed
    int seed = 12345;
    if (argc > 1) {
        seed = atoi(argv[1]);
    }
    cv::setRNGSeed(seed);
    
    std::vector<double> values;
    for (int i = 0; i < 5; i++) {
        values.push_back(cv::theRNG().uniform(0.0, 1.0));
    }
    
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Random values:" << std::endl;
    for (double value : values) {
        std::cout << value << std::endl;
    }
    
    return 0;
}