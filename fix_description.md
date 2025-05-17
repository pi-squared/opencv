# Fix for normalize() function to support NORM_HAMMING and NORM_HAMMING2

## Issue Description
The `normalize()` function in OpenCV does not accept NORM_HAMMING or NORM_HAMMING2 normalization types even for CV_8U matrices where these norm types would be valid. The function only explicitly supports CV_L1, CV_L2, and CV_C (NORM_INF) normalization types, while the underlying `norm()` function supports additional types including NORM_HAMMING and NORM_HAMMING2 for appropriate matrices.

## Fix Implemented
Modified the `normalize()` function in modules/core/src/norm.dispatch.cpp to check if the normalization type is either NORM_HAMMING or NORM_HAMMING2 and allow these types:

```cpp
else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C ||
         norm_type == NORM_HAMMING || norm_type == NORM_HAMMING2 )
{
    scale = norm( _src, norm_type, _mask );
    scale = scale > DBL_EPSILON ? a/scale : 0.;
    shift = 0;
}
```

The `norm()` function already has validation that prevents using NORM_HAMMING and NORM_HAMMING2 with inappropriate matrix types.

## Tests Added
Added a test in modules/core/test/test_math.cpp (Core_Normalize.HammingNormTypes) that verifies:
1. NORM_HAMMING and NORM_HAMMING2 work with CV_8U matrices
2. NORM_HAMMING and NORM_HAMMING2 still throw appropriate exceptions with other matrix types such as CV_32F

## Related Changes
This fix is similar to the recent fix for the `norm()` function (commit 9201ca1af1) which addressed array access in the `getNormDiffFunc` function to ensure proper behavior when using invalid normalization types.