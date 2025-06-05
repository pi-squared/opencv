#!/bin/bash

# Build script for OpenCV with SIMD optimizations

echo "Building OpenCV with histogram SIMD optimizations..."

cd build

# Configure with optimizations enabled
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native" \
    -DCPU_BASELINE=AVX2 \
    -DCPU_DISPATCH=AVX2,AVX512_SKX \
    -DBUILD_TESTS=ON \
    -DBUILD_PERF_TESTS=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_DOCS=OFF \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_OPENMP=ON \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF

# Build imgproc module only for faster compilation
echo "Building imgproc module..."
make -j$(nproc) opencv_imgproc

# Build the test
echo "Building test program..."
cd ..
g++ -o test_histogram_simd test_histogram_simd.cpp \
    -I./build \
    -I./modules/core/include \
    -I./modules/imgproc/include \
    -I./modules/imgcodecs/include \
    -L./build/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -O3 -march=native -std=c++11

echo "Build complete!"