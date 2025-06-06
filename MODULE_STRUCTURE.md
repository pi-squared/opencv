# OpenCV Module Structure Overview

## Main Modules

1. **core** - Core functionality
   - Basic data structures (Mat, Point, Rect, etc.)
   - Arithmetic operations (+, -, *, /, etc.)
   - Matrix operations
   - Memory management
   - SIMD optimizations

2. **imgproc** - Image Processing  
   - Filtering (blur, gaussian blur, median filter, etc.)
   - Geometric transformations (resize, warp, rotate)
   - Color space conversions
   - Morphological operations
   - Feature detection (corners, edges)
   - Histograms

3. **calib3d** - Camera Calibration and 3D Reconstruction
   - Camera calibration
   - Pose estimation
   - Stereo correspondence
   - 3D reconstruction

4. **features2d** - 2D Features Framework
   - Feature detectors (SIFT, SURF, ORB, etc.)
   - Feature descriptors
   - Feature matching

5. **dnn** - Deep Neural Networks
   - Model loading and inference
   - Layer implementations
   - Backend optimizations

6. **objdetect** - Object Detection
   - Cascade classifiers
   - HOG descriptors
   - QR code detection

7. **video** - Video Analysis
   - Motion estimation
   - Object tracking
   - Background subtraction

8. **videoio** - Video I/O
   - Video capture
   - Video writing
   - Camera interfaces

## Code Organization Patterns

### Source Files Structure
- `*.cpp` - Main implementation files
- `*.dispatch.cpp` - CPU dispatch for optimized implementations
- `*.simd.hpp` - SIMD optimized implementations
- `*.avx2.cpp`, `*.sse4_1.cpp` - Architecture-specific optimizations
- `opencl/*.cl` - OpenCL kernels for GPU acceleration

### Performance Testing
- Each module has a `perf/` directory with performance benchmarks
- Tests use the `PERF_TEST` macro framework
- Benchmarks measure function execution time across different parameters

## Key Optimization Opportunities

### 1. Commonly Used Functions (High Impact)
- **resize()** - Image resizing (very frequently used)
- **cvtColor()** - Color space conversions
- **filter2D()** - 2D convolution operations
- **blur(), GaussianBlur()** - Image smoothing
- **threshold()** - Binary thresholding
- **add(), subtract(), multiply()** - Arithmetic operations

### 2. Optimization Strategies
- **SIMD Vectorization** - Using intrinsics for parallel processing
- **Multi-threading** - Parallel execution with OpenMP/TBB
- **GPU Acceleration** - OpenCL/CUDA implementations
- **Cache Optimization** - Better memory access patterns
- **Algorithm Selection** - Choosing optimal algorithms based on input

### 3. Architecture-Specific Optimizations
- AVX2/AVX512 for newer Intel processors
- NEON for ARM processors
- LASX for LoongArch processors
- RVV for RISC-V processors

### 4. Build System Integration
- CMake-based build system
- Automatic CPU feature detection
- Runtime CPU dispatching
- Optional module system