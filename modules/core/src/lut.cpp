// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "convert.hpp"
#include "opencv2/core/hal/intrin.hpp"

/****************************************************************************************\
*                                    LUT Transform                                       *
\****************************************************************************************/

namespace cv
{

#if (CV_SIMD || CV_SIMD_SCALABLE)
// SIMD-optimized LUT for 8-bit to 8-bit transformation
static void LUT8u_8u_simd( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
    if( lutcn == 1 )
    {
        int i = 0;
        const int vlanes = VTraits<v_uint8>::vlanes();
        
        // Process multiple vectors at once for better cache utilization
        const int unroll = 4;
        for( ; i <= len*cn - vlanes*unroll; i += vlanes*unroll )
        {
            // Load multiple vectors
            v_uint8 vsrc0 = vx_load(src + i);
            v_uint8 vsrc1 = vx_load(src + i + vlanes);
            v_uint8 vsrc2 = vx_load(src + i + vlanes*2);
            v_uint8 vsrc3 = vx_load(src + i + vlanes*3);
            
            // Extract indices and perform lookups
            uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) idx0[vlanes], idx1[vlanes], idx2[vlanes], idx3[vlanes];
            v_store_aligned(idx0, vsrc0);
            v_store_aligned(idx1, vsrc1);
            v_store_aligned(idx2, vsrc2);
            v_store_aligned(idx3, vsrc3);
            
            // Perform lookups with better memory access pattern
            for( int j = 0; j < vlanes; j++ )
            {
                idx0[j] = lut[idx0[j]];
                idx1[j] = lut[idx1[j]];
                idx2[j] = lut[idx2[j]];
                idx3[j] = lut[idx3[j]];
            }
            
            // Store results
            v_store(dst + i, vx_load_aligned(idx0));
            v_store(dst + i + vlanes, vx_load_aligned(idx1));
            v_store(dst + i + vlanes*2, vx_load_aligned(idx2));
            v_store(dst + i + vlanes*3, vx_load_aligned(idx3));
        }
        
        // Process remaining full vectors
        for( ; i <= len*cn - vlanes; i += vlanes )
        {
            v_uint8 vsrc = vx_load(src + i);
            uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) temp[vlanes];
            v_store_aligned(temp, vsrc);
            
            for( int j = 0; j < vlanes; j++ )
                temp[j] = lut[temp[j]];
            
            v_store(dst + i, vx_load_aligned(temp));
        }
        vx_cleanup();
        
        // Process remaining pixels
        for( ; i < len*cn; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        // Multi-channel LUT
        for( int i = 0; i < len*cn; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn+k];
    }
}

// SIMD-optimized LUT for 8-bit to 16-bit transformation
static void LUT8u_16u_simd( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
    if( lutcn == 1 )
    {
        int i = 0;
        const int step = VTraits<v_uint16>::vlanes();
        
        // Process two vectors at once for better performance
        for( ; i <= len*cn - step*2; i += step*2 )
        {
            // Load and expand 16 bytes to two 16-bit vectors
            v_uint8 vsrc8 = vx_load(src + i);
            v_uint16 vsrc0, vsrc1;
            v_expand(vsrc8, vsrc0, vsrc1);
            
            ushort CV_DECL_ALIGNED(CV_SIMD_WIDTH) idx0[step], idx1[step];
            v_store_aligned(idx0, vsrc0);
            v_store_aligned(idx1, vsrc1);
            
            // Perform lookups
            for( int j = 0; j < step; j++ )
            {
                idx0[j] = lut[idx0[j]];
                idx1[j] = lut[idx1[j]];
            }
            
            v_store(dst + i, vx_load_aligned(idx0));
            v_store(dst + i + step, vx_load_aligned(idx1));
        }
        
        // Process remaining full vectors
        for( ; i <= len*cn - step; i += step )
        {
            v_uint16 vsrc = v_expand_low(vx_load_low(src + i));
            ushort CV_DECL_ALIGNED(CV_SIMD_WIDTH) temp[step];
            v_store_aligned(temp, vsrc);
            
            for( int j = 0; j < step; j++ )
                temp[j] = lut[temp[j]];
            
            v_store(dst + i, vx_load_aligned(temp));
        }
        vx_cleanup();
        
        // Process remaining pixels
        for( ; i < len*cn; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        // Multi-channel LUT
        for( int i = 0; i < len*cn; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn+k];
    }
}
#endif

template<typename T> static void
LUT8u_( const uchar* src, const T* lut, T* dst, int len, int cn, int lutcn )
{
    if( lutcn == 1 )
    {
        for( int i = 0; i < len*cn; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        for( int i = 0; i < len*cn; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn+k];
    }
}

static void LUT8u_8u( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    LUT8u_8u_simd( src, lut, dst, len, cn, lutcn );
#else
    LUT8u_( src, lut, dst, len, cn, lutcn );
#endif
}

static void LUT8u_8s( const uchar* src, const schar* lut, schar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16u( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    LUT8u_16u_simd( src, lut, dst, len, cn, lutcn );
#else
    LUT8u_( src, lut, dst, len, cn, lutcn );
#endif
}

static void LUT8u_16s( const uchar* src, const short* lut, short* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32s( const uchar* src, const int* lut, int* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16f( const uchar* src, const hfloat* lut, hfloat* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32f( const uchar* src, const float* lut, float* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_64f( const uchar* src, const double* lut, double* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

typedef void (*LUTFunc)( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );

static LUTFunc lutTab[CV_DEPTH_MAX] =
{
    (LUTFunc)LUT8u_8u, (LUTFunc)LUT8u_8s, (LUTFunc)LUT8u_16u, (LUTFunc)LUT8u_16s,
    (LUTFunc)LUT8u_32s, (LUTFunc)LUT8u_32f, (LUTFunc)LUT8u_64f, (LUTFunc)LUT8u_16f
};

#ifdef HAVE_OPENCL

static bool ocl_LUT(InputArray _src, InputArray _lut, OutputArray _dst)
{
    int lcn = _lut.channels(), dcn = _src.channels(), ddepth = _lut.depth();

    UMat src = _src.getUMat(), lut = _lut.getUMat();
    _dst.create(src.size(), CV_MAKETYPE(ddepth, dcn));
    UMat dst = _dst.getUMat();
    int kercn = lcn == 1 ? std::min(4, ocl::predictOptimalVectorWidth(_src, _dst)) : dcn;

    ocl::Kernel k("LUT", ocl::core::lut_oclsrc,
                  format("-D dcn=%d -D lcn=%d -D srcT=%s -D dstT=%s", kercn, lcn,
                         ocl::typeToStr(src.depth()), ocl::memopTypeToStr(ddepth)));
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::ReadOnlyNoSize(lut),
        ocl::KernelArg::WriteOnly(dst, dcn, kercn));

    size_t globalSize[2] = { (size_t)dst.cols * dcn / kercn, ((size_t)dst.rows + 3) / 4 };
    return k.run(2, globalSize, NULL, false);
}

#endif

class LUTParallelBody : public ParallelLoopBody
{
public:
    bool* ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    LUTFunc func;

    LUTParallelBody(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst)
    {
        func = lutTab[lut.depth()];
        *ok = (func != NULL);
    }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        CV_Assert(*ok);

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        int cn = src.channels();
        int lutcn = lut_.channels();

        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2] = {};
        NAryMatIterator it(arrays, ptrs);
        int len = (int)it.size;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], lut_.ptr(), ptrs[1], len, cn, lutcn);
    }
private:
    LUTParallelBody(const LUTParallelBody&);
    LUTParallelBody& operator=(const LUTParallelBody&);
};

} // cv::

void cv::LUT( InputArray _src, InputArray _lut, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int cn = _src.channels(), depth = _src.depth();
    int lutcn = _lut.channels();

    CV_Assert( (lutcn == cn || lutcn == 1) &&
        _lut.total() == 256 && _lut.isContinuous() &&
        (depth == CV_8U || depth == CV_8S) );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_LUT(_src, _lut, _dst))

    Mat src = _src.getMat(), lut = _lut.getMat();
    _dst.create(src.dims, src.size, CV_MAKETYPE(_lut.depth(), cn));
    Mat dst = _dst.getMat();

    CALL_HAL(LUT, cv_hal_lut, src.data, src.step, src.type(), lut.data,
             lut.elemSize1(), lutcn, dst.data, dst.step, src.cols, src.rows);

    if (_src.dims() <= 2)
    {
        bool ok = false;
        LUTParallelBody body(src, lut, dst, &ok);
        if (ok)
        {
            Range all(0, dst.rows);
            if (dst.total() >= (size_t)(1<<18))
                parallel_for_(all, body, (double)std::max((size_t)1, dst.total()>>16));
            else
                body(all);
            if (ok)
                return;
        }
    }

    LUTFunc func = lutTab[lut.depth()];
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], lut.ptr(), ptrs[1], len, cn, lutcn);
}
