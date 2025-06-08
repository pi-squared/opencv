// This file contains optimized sections for CLAHE
// To be integrated into the main clahe.cpp file

#include "opencv2/core/hal/intrin.hpp"

// Optimized histogram calculation for CLAHE using multiple histograms
template <typename T>
void optimizedHistogramCalc(const T* ptr, int width, int height, int sstep, int* tileHist, int histSize, int shift)
{
#if CV_SIMD
    // Use 4 separate histograms to reduce conflicts
    cv::AutoBuffer<int> _hist2(histSize), _hist3(histSize), _hist4(histSize);
    int* hist2 = _hist2.data();
    int* hist3 = _hist3.data();  
    int* hist4 = _hist4.data();
    
    std::fill(hist2, hist2 + histSize, 0);
    std::fill(hist3, hist3 + histSize, 0);
    std::fill(hist4, hist4 + histSize, 0);
    
    for (int y = 0; y < height; ++y, ptr += sstep)
    {
        int x = 0;
        // Process 16 pixels at once
        for (; x <= width - 16; x += 16)
        {
            // Update 4 histograms to reduce memory conflicts
            tileHist[ptr[x] >> shift]++;
            hist2[ptr[x+1] >> shift]++;
            hist3[ptr[x+2] >> shift]++;
            hist4[ptr[x+3] >> shift]++;
            
            tileHist[ptr[x+4] >> shift]++;
            hist2[ptr[x+5] >> shift]++;
            hist3[ptr[x+6] >> shift]++;
            hist4[ptr[x+7] >> shift]++;
            
            tileHist[ptr[x+8] >> shift]++;
            hist2[ptr[x+9] >> shift]++;
            hist3[ptr[x+10] >> shift]++;
            hist4[ptr[x+11] >> shift]++;
            
            tileHist[ptr[x+12] >> shift]++;
            hist2[ptr[x+13] >> shift]++;
            hist3[ptr[x+14] >> shift]++;
            hist4[ptr[x+15] >> shift]++;
        }
        
        // Process remaining pixels
        for (; x < width; ++x)
            tileHist[ptr[x] >> shift]++;
    }
    
    // Merge histograms using SIMD
    const int vecSize = cv::v_int32::nlanes;
    int i = 0;
    for (; i <= histSize - vecSize; i += vecSize)
    {
        cv::v_int32 h1 = cv::vx_load(&tileHist[i]);
        cv::v_int32 h2 = cv::vx_load(&hist2[i]);
        cv::v_int32 h3 = cv::vx_load(&hist3[i]);
        cv::v_int32 h4 = cv::vx_load(&hist4[i]);
        cv::v_store(&tileHist[i], h1 + h2 + h3 + h4);
    }
    for (; i < histSize; ++i)
        tileHist[i] += hist2[i] + hist3[i] + hist4[i];
#else
    // Fallback to original implementation
    for (int y = 0; y < height; ++y, ptr += sstep)
    {
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            int t0 = ptr[x], t1 = ptr[x+1];
            tileHist[t0 >> shift]++; tileHist[t1 >> shift]++;
            t0 = ptr[x+2]; t1 = ptr[x+3];
            tileHist[t0 >> shift]++; tileHist[t1 >> shift]++;
        }
        for (; x < width; ++x)
            tileHist[ptr[x] >> shift]++;
    }
#endif
}

// Optimized clipping for CLAHE histogram
void optimizedClipHistogram(int* tileHist, int histSize, int clipLimit, float lutScale)
{
#if CV_SIMD
    int clipped = 0;
    const int vecSize = cv::v_int32::nlanes;
    cv::v_int32 vClipLimit = cv::vx_setall(clipLimit);
    cv::v_int32 vClipped = cv::vx_setzero<cv::v_int32>();
    
    int i = 0;
    for (; i <= histSize - vecSize; i += vecSize)
    {
        cv::v_int32 vHist = cv::vx_load(&tileHist[i]);
        cv::v_int32 vExcess = cv::v_max(vHist - vClipLimit, cv::vx_setzero<cv::v_int32>());
        vClipped = vClipped + vExcess;
        cv::v_store(&tileHist[i], cv::v_min(vHist, vClipLimit));
    }
    
    // Reduce vClipped to scalar
    clipped = cv::v_reduce_sum(vClipped);
    
    // Process remaining elements
    for (; i < histSize; ++i)
    {
        if (tileHist[i] > clipLimit)
        {
            clipped += tileHist[i] - clipLimit;
            tileHist[i] = clipLimit;
        }
    }
    
    // Redistribute clipped pixels
    int redistBatch = clipped / histSize;
    int residual = clipped - redistBatch * histSize;
    
    cv::v_int32 vRedistBatch = cv::vx_setall(redistBatch);
    i = 0;
    for (; i <= histSize - vecSize; i += vecSize)
    {
        cv::v_int32 vHist = cv::vx_load(&tileHist[i]);
        cv::v_store(&tileHist[i], vHist + vRedistBatch);
    }
    for (; i < histSize; ++i)
        tileHist[i] += redistBatch;
    
    if (residual != 0)
    {
        int residualStep = cv::max(histSize / residual, 1);
        for (int j = 0; j < histSize && residual > 0; j += residualStep, residual--)
            tileHist[j]++;
    }
#else
    // Original implementation
    int clipped = 0;
    for (int i = 0; i < histSize; ++i)
    {
        if (tileHist[i] > clipLimit)
        {
            clipped += tileHist[i] - clipLimit;
            tileHist[i] = clipLimit;
        }
    }
    
    int redistBatch = clipped / histSize;
    int residual = clipped - redistBatch * histSize;
    
    for (int i = 0; i < histSize; ++i)
        tileHist[i] += redistBatch;
    
    if (residual != 0)
    {
        int residualStep = cv::max(histSize / residual, 1);
        for (int i = 0; i < histSize && residual > 0; i += residualStep, residual--)
            tileHist[i]++;
    }
#endif
}

// Optimized bilinear interpolation for CLAHE
template <typename T>
void optimizedBilinearInterpolation(const T* srcRow, T* dstRow, int width,
                                   const T* lutPlane1, const T* lutPlane2,
                                   const int* ind1_p, const int* ind2_p,
                                   const float* xa_p, const float* xa1_p,
                                   float ya, float ya1, int shift)
{
#if CV_SIMD
    const int vecSize = cv::v_float32::nlanes;
    cv::v_float32 vYa = cv::vx_setall(ya);
    cv::v_float32 vYa1 = cv::vx_setall(ya1);
    
    int x = 0;
    for (; x <= width - vecSize; x += vecSize)
    {
        cv::v_float32 vXa = cv::vx_load(xa_p + x);
        cv::v_float32 vXa1 = cv::vx_load(xa1_p + x);
        
        // Manual gather for LUT values
        float lut1[CV_SIMD_WIDTH], lut2[CV_SIMD_WIDTH], lut3[CV_SIMD_WIDTH], lut4[CV_SIMD_WIDTH];
        for (int i = 0; i < vecSize; ++i)
        {
            int srcVal = srcRow[x + i] >> shift;
            int idx1 = ind1_p[x + i] + srcVal;
            int idx2 = ind2_p[x + i] + srcVal;
            
            lut1[i] = lutPlane1[idx1];
            lut2[i] = lutPlane1[idx2];
            lut3[i] = lutPlane2[idx1];
            lut4[i] = lutPlane2[idx2];
        }
        
        cv::v_float32 vLut1 = cv::vx_load(lut1);
        cv::v_float32 vLut2 = cv::vx_load(lut2);
        cv::v_float32 vLut3 = cv::vx_load(lut3);
        cv::v_float32 vLut4 = cv::vx_load(lut4);
        
        // Bilinear interpolation
        cv::v_float32 vInterp1 = vLut1 * vXa1 + vLut2 * vXa;
        cv::v_float32 vInterp2 = vLut3 * vXa1 + vLut4 * vXa;
        cv::v_float32 vRes = vInterp1 * vYa1 + vInterp2 * vYa;
        
        // Convert and store
        if (sizeof(T) == 1) // 8-bit
        {
            cv::v_int32 vResInt = cv::v_round(vRes);
            cv::v_uint16 vRes16 = cv::v_pack_u(vResInt, vResInt);
            cv::v_uint8 vRes8 = cv::v_pack(vRes16, vRes16);
            
            // Store to temporary array and copy
            uchar temp[CV_SIMD_WIDTH];
            cv::v_store(temp, vRes8);
            for (int i = 0; i < vecSize; ++i)
                dstRow[x + i] = temp[i] << shift;
        }
        else // 16-bit
        {
            cv::v_int32 vResInt = cv::v_round(vRes);
            int temp[CV_SIMD_WIDTH];
            cv::v_store(temp, vResInt);
            for (int i = 0; i < vecSize; ++i)
                dstRow[x + i] = cv::saturate_cast<T>(temp[i]) << shift;
        }
    }
    
    // Process remaining pixels
    for (; x < width; ++x)
    {
        int srcVal = srcRow[x] >> shift;
        int ind1 = ind1_p[x] + srcVal;
        int ind2 = ind2_p[x] + srcVal;
        
        float res = (lutPlane1[ind1] * xa1_p[x] + lutPlane1[ind2] * xa_p[x]) * ya1 +
                   (lutPlane2[ind1] * xa1_p[x] + lutPlane2[ind2] * xa_p[x]) * ya;
        
        dstRow[x] = cv::saturate_cast<T>(res) << shift;
    }
#else
    // Original implementation
    for (int x = 0; x < width; ++x)
    {
        int srcVal = srcRow[x] >> shift;
        int ind1 = ind1_p[x] + srcVal;
        int ind2 = ind2_p[x] + srcVal;
        
        float res = (lutPlane1[ind1] * xa1_p[x] + lutPlane1[ind2] * xa_p[x]) * ya1 +
                   (lutPlane2[ind1] * xa1_p[x] + lutPlane2[ind2] * xa_p[x]) * ya;
        
        dstRow[x] = cv::saturate_cast<T>(res) << shift;
    }
#endif
}