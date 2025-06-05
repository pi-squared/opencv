# cornerSubPix Optimization Research

## Algorithm Overview

The `cornerSubPix` function refines corner locations to sub-pixel accuracy. The algorithm:

1. Creates a Gaussian-weighted mask for the search window
2. For each corner point, iteratively:
   - Extracts a window around the current corner estimate
   - Computes gradients in x and y directions
   - Accumulates gradient products weighted by the mask
   - Solves a 2x2 linear system to find the corner position update
   - Updates the corner position until convergence

The key computational bottleneck is the gradient computation loop (lines 112-133 in cornersubpix.cpp).

## Current Implementation Analysis

The inner loop computes:
```cpp
for( i = 0, k = 0; i < win_h; i++, subpix += win_w + 2 )
{
    double py = i - win.height;
    for( j = 0; j < win_w; j++, k++ )
    {
        double m = mask[k];
        double tgx = subpix[j+1] - subpix[j-1];
        double tgy = subpix[j+win_w+2] - subpix[j-win_w-2];
        double gxx = tgx * tgx * m;
        double gxy = tgx * tgy * m;
        double gyy = tgy * tgy * m;
        double px = j - win.width;
        
        a += gxx;
        b += gxy;
        c += gyy;
        bb1 += gxx * px + gxy * py;
        bb2 += gxy * px + gyy * py;
    }
}
```

## SIMD Optimization Opportunities

### 1. Memory Access Pattern
- The gradient computation accesses memory with stride patterns
- X gradient: `subpix[j+1] - subpix[j-1]` (stride 1)
- Y gradient: `subpix[j+win_w+2] - subpix[j-win_w-2]` (stride win_w+2)
- This creates potential cache misses for Y gradient

### 2. Vectorization Strategy

Using OpenCV's Universal Intrinsics, we can process multiple pixels simultaneously:

```cpp
// Process 8 float values at once (AVX) or 4 (SSE)
const int nlanes = v_float32::nlanes;

for( j = 0; j < win_w - nlanes + 1; j += nlanes )
{
    // Load mask values
    v_float32 v_m = vx_load(&mask[k + j]);
    
    // Compute gradients
    v_float32 v_gx = v_sub(vx_load(&subpix[j+1]), vx_load(&subpix[j-1]));
    v_float32 v_gy = v_sub(vx_load(&subpix[j+win_w+2]), vx_load(&subpix[j-win_w-2]));
    
    // Compute products
    v_float32 v_gxx = v_mul(v_mul(v_gx, v_gx), v_m);
    v_float32 v_gxy = v_mul(v_mul(v_gx, v_gy), v_m);
    v_float32 v_gyy = v_mul(v_mul(v_gy, v_gy), v_m);
    
    // px values: j - win.width, j+1 - win.width, ..., j+nlanes-1 - win.width
    v_float32 v_px = v_add(v_setall_f32((float)(j - win.width)), 
                           v_float32(0, 1, 2, 3, 4, 5, 6, 7)); // for AVX
    
    // Accumulate
    v_a = v_add(v_a, v_gxx);
    v_b = v_add(v_b, v_gxy);
    v_c = v_add(v_c, v_gyy);
    v_bb1 = v_add(v_bb1, v_add(v_mul(v_gxx, v_px), v_mul(v_gxy, v_py)));
    v_bb2 = v_add(v_bb2, v_add(v_mul(v_gxy, v_px), v_mul(v_gyy, v_py)));
}
```

### 3. Cache Optimization

To improve cache efficiency:
1. **Transpose optimization**: For small windows, consider transposing the data to make Y gradient access sequential
2. **Prefetching**: Use prefetch intrinsics for the next row's data
3. **Blocking**: Process multiple corners together if they share overlapping windows

### 4. Implementation Best Practices

Based on OpenCV patterns:

1. **Use dispatch mechanism**: Create separate files for different SIMD levels
   - `cornersubpix.cpp` (baseline)
   - `cornersubpix.simd.hpp` (common SIMD code)
   - `cornersubpix.avx2.cpp` (AVX2 optimized)
   - `cornersubpix.sse4.cpp` (SSE4 optimized)

2. **Follow OpenCV's Universal Intrinsics pattern**:
   ```cpp
   #if CV_SIMD
   // SIMD implementation
   #endif
   // Scalar fallback
   ```

3. **Handle boundary cases**: Process remaining pixels with scalar code

### 5. Similar Optimized Code References

1. **spatialGradient** (spatialgradient.cpp): Shows gradient computation with SIMD
2. **calcMinEigenVal** (corner.avx.cpp): Shows similar matrix computation pattern
3. **Sobel filter** (deriv.cpp): Shows optimized derivative computation

## Performance Considerations

1. **Window size impact**: Larger windows benefit more from SIMD
2. **Data type**: Consider using float32 throughout instead of double for better SIMD utilization
3. **Memory alignment**: Ensure proper alignment for SIMD loads/stores
4. **Compiler auto-vectorization**: Modern compilers may already vectorize parts of this code

## Recommended Implementation Steps

1. Profile the current implementation to confirm the bottleneck
2. Implement Universal Intrinsics version first (portable across platforms)
3. Add platform-specific optimizations (AVX2, AVX-512) if needed
4. Benchmark against the scalar version with various window sizes
5. Consider GPU implementation for processing many corners simultaneously