# OpenCV imgproc Functions Without SIMD Optimization

Based on my analysis of the OpenCV imgproc module, here are the files and functions that could benefit from SIMD optimization:

## High Priority Candidates (Heavy Computation, No SIMD)

### 1. **distransform.cpp** - Distance Transform
- Contains nested loops with heavy per-pixel computation
- Functions: `distanceTransform_3x3()`, `distanceTransform_5x5()`
- Multiple array accesses and comparisons per pixel
- No SIMD optimization currently present

### 2. **clahe.cpp** - Contrast Limited Adaptive Histogram Equalization
- Functions: `CLAHE_CalcLut_Body::operator()`, `CLAHE_Interpolation_Body::operator()`
- Heavy histogram computation and interpolation
- Nested loops with floating-point operations
- No dedicated SIMD implementation (only OpenCL)

### 3. **grabcut.cpp** - GrabCut Segmentation
- GMM (Gaussian Mixture Model) computations
- Functions: `GMM::operator()`, matrix operations
- Heavy floating-point arithmetic in inner loops
- No SIMD optimization

### 4. **demosaicing.cpp** - Bayer Pattern Demosaicing
- Has some SIMD for specific cases but many unoptimized paths
- Functions for different Bayer patterns lack comprehensive SIMD
- Heavy color interpolation computations

### 5. **templmatch.cpp** - Template Matching
- Functions for normalized cross-correlation
- Heavy computation in sliding window operations
- Limited SIMD optimization

## Medium Priority Candidates

### 6. **histogram.cpp** - Histogram Calculation
- Functions: `calcHistLookupTables_8u()`, histogram accumulation
- Array indexing and accumulation operations
- Could benefit from SIMD gather/scatter operations

### 7. **connectedcomponents.cpp** - Connected Components Labeling
- Pixel connectivity analysis
- Multiple passes over image data
- No apparent SIMD optimization

### 8. **contours.cpp** - Contour Detection
- Chain code generation and processing
- Boundary following algorithms
- No SIMD optimization

### 9. **convhull.cpp** - Convex Hull
- Geometric computations
- Distance and angle calculations
- No SIMD optimization

### 10. **floodfill.cpp** - Flood Fill Algorithm
- Recursive/iterative pixel processing
- Boundary checking and filling
- No SIMD optimization

## Lower Priority (Less Computation-Intensive)

- **approx.cpp** - Polygon approximation
- **cornersubpix.cpp** - Sub-pixel corner refinement
- **drawing.cpp** - Drawing primitives
- **featureselect.cpp** - Feature selection
- **linefit.cpp** - Line fitting
- **matchcontours.cpp** - Contour matching

## Already Optimized (Have SIMD)

These files already have SIMD optimizations:
- accum.cpp (dispatch + SIMD)
- bilateral_filter.cpp (dispatch + SIMD)
- box_filter.cpp (dispatch + SIMD)
- canny.cpp (inline SIMD)
- color_*.cpp (dispatch + SIMD for HSV, RGB, YUV)
- filter.cpp (dispatch + SIMD)
- median_blur.cpp (dispatch + SIMD)
- morph.cpp (dispatch + SIMD)
- moments.cpp (inline SIMD)
- resize.cpp (AVX2, SSE4.1 optimizations)
- smooth.cpp (dispatch + SIMD)
- stackblur.cpp (inline SIMD)
- sumpixels.cpp (dispatch + SIMD)

## Recommendations

1. **Start with distransform.cpp** - It has clear nested loops with heavy computation and would benefit significantly from SIMD
2. **Focus on clahe.cpp next** - CLAHE is computationally intensive and widely used
3. **Consider grabcut.cpp** for GMM computations which involve many floating-point operations
4. **Template matching** operations in templmatch.cpp would benefit from SIMD for correlation computations