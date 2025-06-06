// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "stat.hpp"

#include "count_non_zero.simd.hpp"
#include "count_non_zero.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

static CountNonZeroFunc getCountNonZeroTab(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getCountNonZeroTab, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

#ifdef HAVE_OPENCL
static bool ocl_countNonZero( InputArray _src, int & res )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), kercn = ocl::predictOptimalVectorWidth(_src);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (depth == CV_64F && !doubleSupport)
        return false;

    int dbsize = ocl::Device::getDefault().maxComputeUnits();
    size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc,
                  format("-D srcT=%s -D srcT1=%s -D cn=1 -D OP_COUNT_NON_ZERO"
                         " -D WGS=%d -D kercn=%d -D WGS2_ALIGNED=%d%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(depth), (int)wgs, kercn,
                         wgs2_aligned, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : ""));
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), db(1, dbsize, CV_32SC1);
    k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
           dbsize, ocl::KernelArg::PtrWriteOnly(db));

    size_t globalsize = dbsize * wgs;
    if (k.run(1, &globalsize, &wgs, true))
        return res = saturate_cast<int>(cv::sum(db.getMat(ACCESS_READ))[0]), true;
    return false;
}
#endif

#if defined HAVE_IPP
static bool ipp_countNonZero( Mat &src, int &res )
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 < 201801
    // Poor performance of SSE42
    if(cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

    Ipp32s  count = 0;
    int     depth = src.depth();

    if(src.dims <= 2)
    {
        IppStatus status;
        IppiSize  size = {src.cols*src.channels(), src.rows};

        if(depth == CV_8U)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, (const Ipp8u *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else if(depth == CV_32F)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, (const Ipp32f *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else
            return false;

        if(status < 0)
            return false;

        res = size.width*size.height - count;
    }
    else
    {
        IppStatus       status;
        const Mat      *arrays[] = {&src, NULL};
        Mat            planes[1];
        NAryMatIterator it(arrays, planes, 1);
        IppiSize        size  = {(int)it.size*src.channels(), 1};
        res = 0;
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            if(depth == CV_8U)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, it.planes->ptr<Ipp8u>(), (int)it.planes->step, size, &count, 0, 0);
            else if(depth == CV_32F)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, it.planes->ptr<Ipp32f>(), (int)it.planes->step, size, &count, 0, 0);
            else
                return false;

            if(status < 0 || (int)it.planes->total()*src.channels() < count)
                return false;

            res += (int)it.planes->total()*src.channels() - count;
        }
    }

    return true;
}
#endif

int countNonZero(InputArray _src)
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), cn = CV_MAT_CN(type);
    CV_Assert( cn == 1 );

#if defined HAVE_OPENCL || defined HAVE_IPP
    int res = -1;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_countNonZero(_src, res),
                res)
#endif

    Mat src = _src.getMat();
    CV_IPP_RUN_FAST(ipp_countNonZero(src, res), res);

    CountNonZeroFunc func = getCountNonZeroTab(src.depth());
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1] = {};
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, nz = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        nz += func( ptrs[0], total );

    return nz;
}

// Forward declarations for optimized implementations
template<typename T>
static void findNonZeroImpl(const Mat& src, std::vector<Point>& idxvec);

#if CV_SIMD || CV_SIMD_SCALABLE
template<typename T>
static inline void findNonZeroRow_SIMD(const T* src, int cols, int row, std::vector<Point>& idxvec);

// Specialization for 8-bit types with SIMD
template<>
inline void findNonZeroRow_SIMD<uchar>(const uchar* src, int cols, int row, std::vector<Point>& idxvec)
{
    int i = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
    const int vlanes = VTraits<v_uint8>::vlanes();
    int cols0 = cols & -vlanes;
    v_uint8 v_zero = vx_setzero_u8();
    
    // Process SIMD chunks
    for (; i < cols0; i += vlanes)
    {
        v_uint8 v_src = vx_load(src + i);
        v_uint8 v_mask = v_ne(v_src, v_zero);
        
        // Check if any elements are non-zero
        if (v_check_any(v_mask))
        {
            // Fall back to scalar for this chunk if we have non-zero elements
            for (int j = 0; j < vlanes; j++)
            {
                if (src[i + j] != 0)
                    idxvec.push_back(Point(i + j, row));
            }
        }
    }
    v_cleanup();
#endif
    // Scalar fallback for remaining elements
    for (; i < cols; i++)
        if (src[i] != 0)
            idxvec.push_back(Point(i, row));
}

// Specialization for 16-bit types
template<>
inline void findNonZeroRow_SIMD<ushort>(const ushort* src, int cols, int row, std::vector<Point>& idxvec)
{
    int i = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
    const int vlanes = VTraits<v_uint16>::vlanes();
    int cols0 = cols & -vlanes;
    v_uint16 v_zero = vx_setzero_u16();
    
    // Process SIMD chunks
    for (; i < cols0; i += vlanes)
    {
        v_uint16 v_src = vx_load(src + i);
        v_uint16 v_mask = v_ne(v_src, v_zero);
        
        // Check if any elements are non-zero
        if (v_check_any(v_mask))
        {
            // Fall back to scalar for this chunk if we have non-zero elements
            for (int j = 0; j < vlanes; j++)
            {
                if (src[i + j] != 0)
                    idxvec.push_back(Point(i + j, row));
            }
        }
    }
    v_cleanup();
#endif
    // Scalar fallback for remaining elements
    for (; i < cols; i++)
        if (src[i] != 0)
            idxvec.push_back(Point(i, row));
}

// Specialization for 32-bit integer types
template<>
inline void findNonZeroRow_SIMD<int>(const int* src, int cols, int row, std::vector<Point>& idxvec)
{
    int i = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
    const int vlanes = VTraits<v_int32>::vlanes();
    int cols0 = cols & -vlanes;
    v_int32 v_zero = vx_setzero_s32();
    
    // Process SIMD chunks
    for (; i < cols0; i += vlanes)
    {
        v_int32 v_src = vx_load(src + i);
        v_int32 v_mask = v_ne(v_src, v_zero);
        
        // Check if any elements are non-zero
        if (v_check_any(v_mask))
        {
            // Fall back to scalar for this chunk if we have non-zero elements
            for (int j = 0; j < vlanes; j++)
            {
                if (src[i + j] != 0)
                    idxvec.push_back(Point(i + j, row));
            }
        }
    }
    v_cleanup();
#endif
    // Scalar fallback for remaining elements
    for (; i < cols; i++)
        if (src[i] != 0)
            idxvec.push_back(Point(i, row));
}

// Specialization for 32-bit float types
template<>
inline void findNonZeroRow_SIMD<float>(const float* src, int cols, int row, std::vector<Point>& idxvec)
{
    int i = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
    const int vlanes = VTraits<v_float32>::vlanes();
    int cols0 = cols & -vlanes;
    v_float32 v_zero = vx_setzero_f32();
    
    // Process SIMD chunks
    for (; i < cols0; i += vlanes)
    {
        v_float32 v_src = vx_load(src + i);
        v_float32 v_mask = v_ne(v_src, v_zero);
        
        // Check if any elements are non-zero
        if (v_check_any(v_mask))
        {
            // Fall back to scalar for this chunk if we have non-zero elements
            for (int j = 0; j < vlanes; j++)
            {
                if (src[i + j] != 0)
                    idxvec.push_back(Point(i + j, row));
            }
        }
    }
    v_cleanup();
#endif
    // Scalar fallback for remaining elements
    for (; i < cols; i++)
        if (src[i] != 0)
            idxvec.push_back(Point(i, row));
}

// Generic fallback for other types
template<typename T>
static inline void findNonZeroRow_SIMD(const T* src, int cols, int row, std::vector<Point>& idxvec)
{
    for (int i = 0; i < cols; i++)
        if (src[i] != 0)
            idxvec.push_back(Point(i, row));
}
#endif

// Implementation for each type
template<typename T>
static void findNonZeroImpl(const Mat& src, std::vector<Point>& idxvec)
{
    const int rows = src.rows;
    const int cols = src.cols;
    
    // Pre-allocate space for better performance
    idxvec.reserve(std::min(rows * cols / 8, 100000));
    
    for (int i = 0; i < rows; i++)
    {
        const T* ptr = src.ptr<T>(i);
#if CV_SIMD || CV_SIMD_SCALABLE
        findNonZeroRow_SIMD<T>(ptr, cols, i, idxvec);
#else
        for (int j = 0; j < cols; j++)
            if (ptr[j] != 0)
                idxvec.push_back(Point(j, i));
#endif
    }
}

void findNonZero(InputArray _src, OutputArray _idx)
{
    Mat src = _src.getMat();
    CV_Assert( src.channels() == 1 && src.dims == 2 );

    std::vector<Point> idxvec;
    
    // Type dispatch table
    typedef void (*FindNonZeroFunc)(const Mat&, std::vector<Point>&);
    static FindNonZeroFunc funcs[CV_DEPTH_MAX] = 
    {
        findNonZeroImpl<uchar>,   // CV_8U
        findNonZeroImpl<schar>,   // CV_8S  
        findNonZeroImpl<ushort>,  // CV_16U
        findNonZeroImpl<short>,   // CV_16S
        findNonZeroImpl<int>,     // CV_32S
        findNonZeroImpl<float>,   // CV_32F
        findNonZeroImpl<double>,  // CV_64F
        0
    };
    
    FindNonZeroFunc func = funcs[src.depth()];
    CV_Assert( func != 0 );
    
    func(src, idxvec);
    
    if( idxvec.empty() || (_idx.kind() == _InputArray::MAT && !_idx.getMatRef().isContinuous()) )
        _idx.release();

    if( !idxvec.empty() )
        Mat(idxvec).copyTo(_idx);
}

} // namespace
