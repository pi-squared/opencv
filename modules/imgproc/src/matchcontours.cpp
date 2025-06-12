/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"


double cv::matchShapes(InputArray contour1, InputArray contour2, int method, double)
{
    CV_INSTRUMENT_REGION();

    double ma[7], mb[7];
    int i, sma, smb;
    double eps = 1.e-5;
    double mmm;
    double result = 0;
    bool anyA = false, anyB = false;

    HuMoments( moments(contour1), ma );
    HuMoments( moments(contour2), mb );

#if CV_SIMD
    // SIMD optimization for processing multiple Hu moments at once
    // Since we have 7 moments, we'll process them in parallel where possible
    if (method == 1 || method == 2)
    {
        v_float64 v_eps = v_setall<double>(eps);
        v_float64 v_zero = v_setzero<v_float64>();
        v_float64 v_one = v_setall<double>(1.0);
        v_float64 v_neg_one = v_setall<double>(-1.0);
        
        // Process moments with SIMD
        if (VTraits<v_float64>::vlanes() >= 2)
        {
            for (i = 0; i <= 4; i += 2)
            {
                if (i + 1 >= 7) break;
                
                // Load 2 pairs of moments
                v_float64 va = v_load(ma + i);
                v_float64 vb = v_load(mb + i);
                
                // Calculate absolute values
                v_float64 v_ama = v_abs(va);
                v_float64 v_amb = v_abs(vb);
                
                // Check if any values are > 0
                v_float64 mask_a = v_gt(v_ama, v_zero);
                v_float64 mask_b = v_gt(v_amb, v_zero);
                
                if (v_check_any(mask_a))
                    anyA = true;
                if (v_check_any(mask_b))
                    anyB = true;
                
                // Calculate signs
                v_float64 v_sma = v_select(v_gt(va, v_zero), v_one, 
                                         v_select(v_lt(va, v_zero), v_neg_one, v_zero));
                v_float64 v_smb = v_select(v_gt(vb, v_zero), v_one,
                                         v_select(v_lt(vb, v_zero), v_neg_one, v_zero));
                
                // Check if both ama and amb > eps
                v_float64 mask = v_and(v_gt(v_ama, v_eps), v_gt(v_amb, v_eps));
                
                if (v_check_any(mask))
                {
                    // Need to process elements individually due to log10
                    double ta[2], tb[2];
                    v_store(ta, v_ama);
                    v_store(tb, v_amb);
                    
                    double tsma[2], tsmb[2];
                    v_store(tsma, v_sma);
                    v_store(tsmb, v_smb);
                    
                    for (int j = 0; j < 2 && i + j < 7; j++)
                    {
                        if (ta[j] > eps && tb[j] > eps)
                        {
                            if (method == 1)
                            {
                                double ama = 1. / (tsma[j] * log10(ta[j]));
                                double amb = 1. / (tsmb[j] * log10(tb[j]));
                                result += fabs(-ama + amb);
                            }
                            else // method == 2
                            {
                                double ama = tsma[j] * log10(ta[j]);
                                double amb = tsmb[j] * log10(tb[j]);
                                result += fabs(-ama + amb);
                            }
                        }
                    }
                }
            }
            
            // Process remaining moments
            for (; i < 7; i++)
            {
                double ama = fabs(ma[i]);
                double amb = fabs(mb[i]);
                
                if (ama > 0)
                    anyA = true;
                if (amb > 0)
                    anyB = true;
                
                if (ma[i] > 0)
                    sma = 1;
                else if (ma[i] < 0)
                    sma = -1;
                else
                    sma = 0;
                if (mb[i] > 0)
                    smb = 1;
                else if (mb[i] < 0)
                    smb = -1;
                else
                    smb = 0;
                
                if (ama > eps && amb > eps)
                {
                    if (method == 1)
                    {
                        ama = 1. / (sma * log10(ama));
                        amb = 1. / (smb * log10(amb));
                        result += fabs(-ama + amb);
                    }
                    else // method == 2
                    {
                        ama = sma * log10(ama);
                        amb = smb * log10(amb);
                        result += fabs(-ama + amb);
                    }
                }
            }
        }
        else
        {
            // Fall back to scalar implementation
            goto scalar_impl;
        }
    }
    else
    {
scalar_impl:
#endif // CV_SIMD
    switch (method)
    {
    case 1:
        for( i = 0; i < 7; i++ )
        {
            double ama = fabs( ma[i] );
            double amb = fabs( mb[i] );

            if (ama > 0)
                anyA = true;
            if (amb > 0)
                anyB = true;

            if( ma[i] > 0 )
                sma = 1;
            else if( ma[i] < 0 )
                sma = -1;
            else
                sma = 0;
            if( mb[i] > 0 )
                smb = 1;
            else if( mb[i] < 0 )
                smb = -1;
            else
                smb = 0;

            if( ama > eps && amb > eps )
            {
                ama = 1. / (sma * log10( ama ));
                amb = 1. / (smb * log10( amb ));
                result += fabs( -ama + amb );
            }
        }
        break;

    case 2:
        for( i = 0; i < 7; i++ )
        {
            double ama = fabs( ma[i] );
            double amb = fabs( mb[i] );

            if (ama > 0)
                anyA = true;
            if (amb > 0)
                anyB = true;

            if( ma[i] > 0 )
                sma = 1;
            else if( ma[i] < 0 )
                sma = -1;
            else
                sma = 0;
            if( mb[i] > 0 )
                smb = 1;
            else if( mb[i] < 0 )
                smb = -1;
            else
                smb = 0;

            if( ama > eps && amb > eps )
            {
                ama = sma * log10( ama );
                amb = smb * log10( amb );
                result += fabs( -ama + amb );
            }
        }
        break;

    case 3:
        for( i = 0; i < 7; i++ )
        {
            double ama = fabs( ma[i] );
            double amb = fabs( mb[i] );

            if (ama > 0)
                anyA = true;
            if (amb > 0)
                anyB = true;

            if( ma[i] > 0 )
                sma = 1;
            else if( ma[i] < 0 )
                sma = -1;
            else
                sma = 0;
            if( mb[i] > 0 )
                smb = 1;
            else if( mb[i] < 0 )
                smb = -1;
            else
                smb = 0;

            if( ama > eps && amb > eps )
            {
                ama = sma * log10( ama );
                amb = smb * log10( amb );
                mmm = fabs( (ama - amb) / ama );
                if( result < mmm )
                    result = mmm;
            }
        }
        break;
    default:
        CV_Error( cv::Error::StsBadArg, "Unknown comparison method" );
    }
#if CV_SIMD
    }
#endif

    //If anyA and anyB are both true, the result is correct.
    //If anyA and anyB are both false, the distance is 0, perfect match.
    //If only one is true, then it's a false 0 and return large error.
    if (anyA != anyB)
        result = DBL_MAX;

    return result;
}


CV_IMPL  double
cvMatchShapes( const void* _contour1, const void* _contour2,
               int method, double parameter )
{
    cv::AutoBuffer<double> abuf1, abuf2;
    cv::Mat contour1 = cv::cvarrToMat(_contour1, false, false, 0, &abuf1);
    cv::Mat contour2 = cv::cvarrToMat(_contour2, false, false, 0, &abuf2);

    return cv::matchShapes(contour1, contour2, method, parameter);
}

/* End of file. */
