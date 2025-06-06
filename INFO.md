# OpenCV Development Log

## Memory Allocation Optimization (2025-01-06)

### Branch: optimize-memory-allocation

**What was done:**
- Optimized memory allocation in `modules/core/src/alloc.cpp`
- Added support for C++17's `std::aligned_alloc` when available
- Replaced `std::map` with `std::unordered_map` for allocation tracking

**Key changes:**
1. Added C++17 feature detection and conditional use of `std::aligned_alloc`
2. Modified `fastMalloc` to use `std::aligned_alloc` as the preferred method when available
3. Updated `fastFree` to handle memory allocated with `std::aligned_alloc`
4. Changed allocation tracking from `std::map` to `std::unordered_map` for better performance

**Benefits:**
- Better performance for aligned memory allocations on modern C++17 compilers
- Improved hash table performance for allocation statistics tracking
- Maintains backward compatibility with existing allocation methods

**Testing:**
- Ran existing allocator tests: PASSED
- Verified alignment calculations work correctly
- Confirmed no regression in core Mat tests

**What works:**
- C++17 aligned_alloc integration when compiler supports it
- Fallback to existing methods (posix_memalign, memalign, etc.) on older systems
- All existing tests pass

**What doesn't work / Limitations:**
- C++17 aligned_alloc is not used on MSVC (excluded due to compatibility)
- Memory pool implementation was considered but not included due to complexity of supporting proper deallocation

**Future improvements to consider:**
- Implement a small object memory pool with proper deallocation support
- Add performance benchmarks to quantify the improvement
- Consider using jemalloc or tcmalloc integration for even better performance
- Explore lock-free data structures for allocation tracking

## GEMM (Matrix Multiplication) Optimization with AVX-512/AVX2 (2025-01-06)

### Branch: optimize-gemm-avx512

**What was done:**
- Created optimized GEMM implementation using AVX-512 and AVX2 SIMD instructions
- Added cache-friendly blocking strategy for better memory access patterns
- Implemented micro-kernels for efficient SIMD utilization
- Added FMA (Fused Multiply-Add) instructions for better performance

**Key changes:**
1. Created new file `modules/core/src/matmul_avx512.cpp` with optimized implementation
2. Modified `matmul.simd.hpp` to use optimized version for suitable matrix sizes
3. Implemented 8x16 micro-kernel for AVX-512 and 4x8 micro-kernel for AVX2
4. Added three-level cache blocking (M, N, K dimensions)

**Key optimizations:**
- **SIMD Vectorization**: Uses AVX-512 (16 floats) or AVX2 (8 floats) vector operations
- **Cache Blocking**: Blocks of 64x256x256 to fit in L1/L2 cache
- **FMA Instructions**: Uses hardware FMA for combined multiply-add operations
- **Micro-kernels**: Optimized inner loops for 8x16 (AVX-512) or 4x8 (AVX2) tiles

**Performance results:**
- Test configuration: 256x256 matrix multiplication
- Baseline performance: 1.78 GFLOPS
- AVX2 optimized performance: 39.3 GFLOPS
- **Speedup: 22x faster than baseline**
- Results verified to be numerically correct (max error < 1e-5)

**What works:**
- AVX-512 and AVX2 optimized paths for float32 GEMM
- Non-transposed matrix multiplication (flags = 0)
- Matrices with dimensions >= 32 for good SIMD efficiency
- Proper handling of edge cases with scalar fallback

**What doesn't work / Limitations:**
- Transposed matrix cases fall back to baseline (optimization opportunity)
- Only implemented for float32, not double or complex types yet
- Requires AVX2 or AVX-512 capable CPU
- Simple parallelization strategy (could use OpenMP or TBB)

**Future improvements to consider:**
- Implement optimized kernels for transposed cases (GEMM_1_T, GEMM_2_T)
- Add multi-threading with OpenMP or TBB for large matrices
- Implement AVX-512 VNNI instructions for int8 GEMM
- Add support for double precision (gemm64f) with AVX-512
- Implement Strassen's algorithm for very large matrices
- Add NUMA-aware memory allocation for multi-socket systems
- Create specialized kernels for small fixed-size matrices