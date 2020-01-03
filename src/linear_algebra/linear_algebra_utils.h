// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// Written by J.-L. Fattebert, D. Osei-Kuffuor and I.S. Dunn.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_LINEAR_ALGEBRA_UTILS_H
#define MGMOL_LINEAR_ALGEBRA_UTILS_H

#include "memory_space.h"

/* mixed-precision vector dot-product. Accumulates results
 * in double precision and stores as double precision.
 */
double MPdot(const int len, const float* __restrict__ xptr,
    const float* __restrict__ yptr);
double MPdot(const int len, const double* const xptr, const double* const yptr);
template <typename T1, typename T2>
double MPdot(const int len, const T1* const xptr, const T2* const yptr);

/* mixed-precision vector times scalar plus vector. Accumulates results
 * in double precision and stores in single precision.
 */
void MPaxpy(const int len, double scal, const double* __restrict__ xptr,
    double* __restrict__ yptr);
template <typename T1, typename T2>
void MPaxpy(const int len, double scal, const T1* __restrict__ xptr,
    T2* __restrict__ yptr);

template <typename MemorySpaceType>
struct LinearAlgebraUtils
{
    /* mixed-precision scalar times vector. Accumulates results
     * in double precision and stores as single precision.
     */
    static void MPscal(const int len, const double scal, double* dptr);
    static void MPscal(const int len, const double scal, float* dptr);

    template <typename T1, typename T2, typename T3>
    static void MPgemm(const char transa, const char transb, const int m,
        const int n, const int k, const double alpha, const T1* const a,
        const int lda, const T2* const b, const int ldb, const double beta,
        T3* const c, const int ldc);

    template <typename T1, typename T2, typename T3>
    static void MPgemmNN(const int m, const int n, const int k,
        const double alpha, const T1* const a, const int lda, const T2* const b,
        const int ldb, const double beta, T3* const c, const int ldc);
};

#endif
