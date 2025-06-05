#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build_test
cd build_test

# Configure with optimizations enabled
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_TESTS=OFF \
         -DBUILD_PERF_TESTS=OFF \
         -DBUILD_EXAMPLES=OFF \
         -DBUILD_opencv_apps=OFF \
         -DBUILD_DOCS=OFF \
         -DBUILD_opencv_python2=OFF \
         -DBUILD_opencv_python3=OFF \
         -DWITH_IPP=OFF \
         -DCPU_BASELINE=AVX2 \
         -DCPU_DISPATCH=AVX2,AVX512_SKX

# Build only imgproc module
make -j$(nproc) opencv_imgproc

# Compile test program
g++ -O3 -march=native ../test_distance_transform.cpp \
    -I../include \
    -I../modules/core/include \
    -I../modules/imgproc/include \
    -I../modules/imgcodecs/include \
    -I../modules/highgui/include \
    -I. \
    -L./lib \
    -lopencv_imgproc -lopencv_core \
    -o test_distance_transform

echo "Build complete. Run with: ./build_test/test_distance_transform"