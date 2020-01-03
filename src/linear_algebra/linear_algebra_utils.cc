// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// Written by J.-L. Fattebert, D. Osei-Kuffuor and I.S. Dunn.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "linear_algebra_utils.h"
#include "MGmol_blas1.h"
#include "Timer.h"
#include "blas2_c.h"
#include "blas3_c.h"
#include "magma_singleton.h"

#ifdef HAVE_MAGMA
#include <magma_v2.h>
#endif

#include <vector>

Timer dgemm_tm("dgemm");
Timer mpgemm_tm("mpgemm");
Timer tttgemm_tm("tttgemm");

Timer mpdot_tm("mpdot");
Timer ttdot_tm("ttdot");

template <typename T1, typename T2>
void MPaxpy(const int len, double scal, const T1* __restrict__ xptr,
    T2* __restrict__ yptr)
{
    for (int k = 0; k < len; k++)
    {
        yptr[k] += (T2)(scal * (double)xptr[k]);
    }
}

void MPaxpy(const int len, double scal, const double* __restrict__ xptr,
    double* __restrict__ yptr)
{
    const int one = 1;
    DAXPY(&len, &scal, xptr, &one, yptr, &one);
}

double MPdot(const int len, const float* __restrict__ xptr,
    const float* __restrict__ yptr)
{
    mpdot_tm.start();

#ifdef BGQ
    const int one = 1;
    double dot    = (double)SDOT(&len, xptr, &one, yptr, &one);
#else
    double dot = 0.;
    for (int k = 0; k < len; k++)
    {
        double val1 = (double)xptr[k];
        double val2 = (double)yptr[k];
        dot += val1 * val2;
    }
#endif

    mpdot_tm.stop();

    return dot;
}

double MPdot(const int len, const double* const xptr, const double* const yptr)
{
    const int one = 1;
    return DDOT(&len, xptr, &one, yptr, &one);
}

template <typename T1, typename T2>
double MPdot(
    const int len, const T1* __restrict__ xptr, const T2* __restrict__ yptr)
{
    ttdot_tm.start();

    double dot = 0.;
    for (int k = 0; k < len; k++)
    {
        double val1 = (double)xptr[k];
        double val2 = (double)yptr[k];
        dot += val1 * val2;
    }

    ttdot_tm.stop();

    return dot;
}

//////////////////////////////
//          MPscal          //
//////////////////////////////
// MemorySpace::Host
template <>
void LinearAlgebraUtils<MemorySpace::Host>::MPscal(
    const int len, const double scal, double* dptr)
{
    const int one = 1;
    DSCAL(&len, &scal, dptr, &one);
}

template <>
void LinearAlgebraUtils<MemorySpace::Host>::MPscal(
    const int len, const double scal, float* dptr)
{
    if (scal == 1.)
        return;
    else if (scal == 0.)
    {
        memset(dptr, 0, len * sizeof(float));
    }
    else
    {
        for (int k = 0; k < len; k++)
        {
            double val = static_cast<double>(dptr[k]);
            dptr[k]    = static_cast<float>(scal * val);
        }
    }
}

// MemorySpace::Device
template <>
void LinearAlgebraUtils<MemorySpace::Device>::MPscal(
    const int len, const double scal, double* dptr)
{
    int const increment   = 1;
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();
    magma_dscal(len, scal, dptr, increment, magma_singleton.queue_);
}

template <>
void LinearAlgebraUtils<MemorySpace::Device>::MPscal(
    const int len, const double scal, float* dptr)
{
    if (scal == 1.)
        return;
    else
    {
        // There is no function in MAGMA that can perfor this operation. If we
        // have OpenMP with offloading we can perform the operation directly on
        // the GPU. Otherwise, we need to move the data to the CPU and then,
        // move it back to the GPU.
#ifdef HAVE_OPENMP_OFFLOAD
#pragma omp target teams distribute parallel for is_device_ptr(dptr)
        for (int k = 0; k < len; k++)
        {
            double val = static_cast<double>(dptr[k]);
            dptr[k]    = static_cast<float>(scal * val);
        }
#else
        std::vector<float> dptr_host(len);
        MemorySpace::copy_to_host(dptr, dptr_host);
#pragma omp parallel for
        for (int k = 0; k < len; k++)
        {
            double val   = static_cast<double>(dptr_host[k]);
            dptr_host[k] = static_cast<float>(scal * val);
        }
        MemorySpace::copy_to_dev(dptr_host, dptr);
#endif
    }
}

////////////////////////////////
//          MPgemmNN          //
////////////////////////////////
// MemorySpace::Host
template <>
template <typename T1, typename T2, typename T3>
void LinearAlgebraUtils<MemorySpace::Host>::MPgemm(const char transa,
    const char transb, const int m, const int n, const int k,
    const double alpha, const T1* const a, const int lda, const T2* const b,
    const int ldb, const double beta, T3* const c, const int ldc)
{
    tttgemm_tm.start();
    // if(onpe0)cout<<"template MPgemm..."<<endl;

    if (beta == 1. && (alpha == 0. || m == 0 || n == 0 || k == 0)) return;

    /* case transb == 'N' and transa == 'N' */
    if (transb == 'N' || transb == 'n')
    {
        if (transa == 'N' || transa == 'n')
        {
            /* buffer to hold accumulation in double */
            std::vector<double> buff(m);
            for (int j = 0; j < n; j++)
            {
                std::fill(buff.begin(), buff.end(), 0.);
                for (int l = 0; l < k; l++)
                {
                    /* pointer to beginning of column l in matrix a */
                    const T1* colL = a + lda * l;
                    /* get multiplier */
                    double mult = (double)(alpha * b[ldb * j + l]);
                    MPaxpy(m, mult, colL, buff.data());
                }
                /* Update col j of of result matrix C. */
                /* Get pointer to beginning of column j in C. */
                T3* cj = c + ldc * j;
                MPscal(m, beta, cj);
                for (int i = 0; i < m; i++)
                {
                    cj[i] += (T3)buff[i];
                }
            }
        }
        else /* transa == 'T'/'C' */
        {
            for (int j = 0; j < n; j++)
            {
                const T2* __restrict__ bj = b + ldb * j;
                for (int i = 0; i < m; i++)
                {
                    const int pos             = ldc * j + i;
                    double bc                 = (double)c[pos] * beta;
                    const T1* __restrict__ ai = a + lda * i;
                    c[pos] = (T3)(alpha * MPdot(k, ai, bj) + bc);
                }
            }
        }
    }
    else /* transb == 'T'/'C' */
    {
        if (transa == 'N' || transa == 'n')
        {
            /* buffer to hold accumulation in double */
            std::vector<double> buff(m);
            for (int j = 0; j < n; j++)
            {
                std::fill(buff.begin(), buff.end(), 0.);
                for (int l = 0; l < k; l++)
                {
                    /* pointer to beginning of column l in matrix a */
                    const T1* colL = a + lda * l;
                    /* get multiplier */
                    double mult = (double)(alpha * b[ldb * l + j]);
                    MPaxpy(m, mult, colL, buff.data());
                }
                /* Update col j of of result matrix C. */
                /* Get pointer to beginning of column j in C. */
                T3* cj = c + ldc * j;
                MPscal(m, beta, cj);
                for (int i = 0; i < m; i++)
                {
                    cj[i] += (T3)buff[i];
                }
            }
        }
        else /* transa == 'T'/'C' */
        {
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    const int pos = ldc * j + i;
                    const T1* ai  = a + lda * i;
                    double sum    = 0.;
                    for (int l = 0; l < k; l++)
                    {
                        sum += alpha * ai[l] * b[ldb * l + j];
                    }
                    sum += (double)(beta * c[pos]);
                    c[pos] = (T3)sum;
                }
            }
        }
    }

    tttgemm_tm.stop();
}

// input/output in double, computation in double
template <>
template <>
void LinearAlgebraUtils<MemorySpace::Host>::MPgemm<double, double, double>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const double* const a, const int lda,
    const double* const b, const int ldb, const double beta, double* const c,
    const int ldc)
{
    dgemm_tm.start();
    DGEMM(
        &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    dgemm_tm.stop();
}

// input/output in float, computation in double
template <>
template <>
void LinearAlgebraUtils<MemorySpace::Host>::MPgemm<float, float, float>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const float* const a, const int lda,
    const float* const b, const int ldb, const double beta, float* const c,
    const int ldc)
{
    mpgemm_tm.start();

    if (beta == 1. && (alpha == 0. || m == 0 || n == 0 || k == 0)) return;

    /* case transb == 'N' and transa == 'N' */
    if (transb == 'N' || transb == 'n')
    {
        if (transa == 'N' || transa == 'n')
        {
            /* buffer to hold accumulation in double */
            std::vector<double> buff(m);
            for (int j = 0; j < n; j++)
            {
                std::fill(buff.begin(), buff.end(), 0);
                for (int l = 0; l < k; l++)
                {
                    /* pointer to beginning of column l in matrix a */
                    const float* colL = a + lda * l;
                    /* get multiplier */
                    double mult = (double)(alpha * b[ldb * j + l]);
                    MPaxpy(m, mult, colL, buff.data());
                }
                /* Update col j of of result matrix C. */
                /* Get pointer to beginning of column j in C. */
                float* cj = c + ldc * j;
                MPscal(m, beta, cj);
                for (int i = 0; i < m; i++)
                    cj[i] += (float)buff[i];
            }
        }
        else /* transa == 'T'/'C' */
        {
            for (int j = 0; j < n; j++)
            {
                const float* __restrict__ bj = b + ldb * j;
                for (int i = 0; i < m; i++)
                {
                    const int pos                = ldc * j + i;
                    double bc                    = (double)c[pos] * beta;
                    const float* __restrict__ ai = a + lda * i;
                    c[pos] = (float)(alpha * MPdot(k, ai, bj) + bc);
                }
            }
        }
    }
    else /* transb == 'T'/'C' */
    {
        if (transa == 'N' || transa == 'n')
        {
            /* buffer to hold accumulation in double */
            std::vector<double> buff(m);
            for (int j = 0; j < n; j++)
            {
                std::fill(buff.begin(), buff.end(), 0);
                for (int l = 0; l < k; l++)
                {
                    /* pointer to beginning of column l in matrix a */
                    const float* colL = a + lda * l;
                    /* get multiplier */
                    double mult = (double)(alpha * b[ldb * l + j]);
                    MPaxpy(m, mult, colL, buff.data());
                }
                /* Update col j of of result matrix C. */
                /* Get pointer to beginning of column j in C. */
                float* cj = c + ldc * j;
                MPscal(m, beta, cj);
                for (int i = 0; i < m; i++)
                    cj[i] += (float)buff[i];
            }
        }
        else /* transa == 'T'/'C' */
        {
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    const int pos   = ldc * j + i;
                    const float* ai = a + lda * i;
                    double sum      = 0.;
                    for (int l = 0; l < k; l++)
                    {
                        sum += alpha * ai[l] * b[ldb * l + j];
                    }
                    sum += (double)(beta * c[pos]);
                    c[pos] = (float)sum;
                }
            }
        }
    }

    mpgemm_tm.stop();
}

// MemorySpace::Device
#ifdef HAVE_MAGMA
template <>
template <typename T1, typename T2, typename T3>
void LinearAlgebraUtils<MemorySpace::Device>::MPgemm(const char transa,
    const char transb, const int m, const int n, const int k,
    const double alpha, const T1* const a, const int lda, const T2* const b,
    const int ldb, const double beta, T3* const c, const int ldc)
{
    std::vector<T1> a_host(lda * k);
    std::vector<T2> b_host(ldb * n);
    std::vector<T3> c_host(ldc * n);

    // Move the data to the host
    MemorySpace::copy_to_host(a, a_host);
    MemorySpace::copy_to_host(b, b_host);

    LinearAlgebraUtils<MemorySpace::Host>::MPgemm(transa, transb, m, n, k,
        alpha, a_host.data(), lda, b_host.data(), ldb, beta, c_host.data(),
        ldc);

    // Move the data to the device
    MemorySpace::copy_to_dev(c_host, c);
}

// input/output in double, computation in double
template <>
template <>
void LinearAlgebraUtils<MemorySpace::Device>::MPgemm(const char transa,
    const char transb, const int m, const int n, const int k,
    const double alpha, const double* const a, const int lda,
    const double* const b, const int ldb, const double beta, double* const c,
    const int ldc)
{
    dgemm_tm.start();
    // Transform char to magma_trans_t
    auto convert_to_magma_trans = [](const char trans) {
        if ((trans == 'N') || trans == 'n')
            return MagmaNoTrans;
        else if ((trans == 'T') || trans == 't')
            return MagmaTrans;
        else if ((trans == 'C') || trans == 'c')
            return MagmaConjTrans;
        else
        {
            std::cerr << "Unknown tranpose operation: " << trans << std::endl;
            return MagmaNoTrans;
        }
    };

    magma_trans_t magma_transa = convert_to_magma_trans(transa);
    magma_trans_t magma_transb = convert_to_magma_trans(transb);

    // Perform dgemm
    auto& magma_singleton = MagmaSingleton::get_magma_singleton();
    magmablas_dgemm(magma_transa, magma_transb, m, n, k, alpha, a, lda, b, ldb,
        beta, c, ldc, magma_singleton.queue_);
    dgemm_tm.stop();
}
#endif

////////////////////////////////
//          MPgemmNN          //
////////////////////////////////

template <typename MemorySpaceType>
template <typename T1, typename T2, typename T3>
void LinearAlgebraUtils<MemorySpaceType>::MPgemmNN(const int m, const int n,
    const int k, const double alpha, const T1* const a, const int lda,
    const T2* const b, const int ldb, const double beta, T3* const c,
    const int ldc)
{
    char transa = 'n';
    char transb = 'n';
    LinearAlgebraUtils<MemorySpaceType>::MPgemm(
        transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

///////////////////////////
//          ETI          //
///////////////////////////
template void MPaxpy<float, double>(const int len, const double scal,
    const float* __restrict__ xptr, double* __restrict__ yptr);
template void MPaxpy<float, float>(const int len, const double scal,
    const float* __restrict__ xptr, float* __restrict__ yptr);

template double MPdot<double, float>(const int len,
    const double* __restrict__ xptr, const float* __restrict__ yptr);
template double MPdot<float, double>(const int len,
    const float* __restrict__ xptr, const double* __restrict__ yptr);

template class LinearAlgebraUtils<MemorySpace::Host>;
template void
LinearAlgebraUtils<MemorySpace::Host>::MPgemm<double, float, double>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const double* const a, const int lda,
    const float* const b, const int ldb, const double beta, double* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Host>::MPgemm<float, double, float>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const float* const a, const int lda,
    const double* const b, const int ldb, const double beta, float* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Host>::MPgemm<double, double, float>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const double* const a, const int lda,
    const double* const b, const int ldb, const double beta, float* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Host>::MPgemm<float, float, double>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const float* const a, const int lda,
    const float* const b, const int ldb, const double beta, double* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN<float, double, float>(
    const int m, const int n, const int k, const double alpha,
    const float* const a, const int lda, const double* const b, const int ldb,
    const double beta, float* const c, const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Host>::MPgemmNN<double, double, double>(
    const int m, const int n, const int k, const double alpha,
    const double* const a, const int lda, const double* const b, const int ldb,
    const double beta, double* const c, const int ldc);

#ifdef HAVE_MAGMA
template class LinearAlgebraUtils<MemorySpace::Device>;
template void
LinearAlgebraUtils<MemorySpace::Device>::MPgemm<float, float, float>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const float* const a, const int lda,
    const float* const b, const int ldb, const double beta, float* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Device>::MPgemm<double, float, double>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const double* const a, const int lda,
    const float* const b, const int ldb, const double beta, double* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Device>::MPgemm<float, double, float>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const float* const a, const int lda,
    const double* const b, const int ldb, const double beta, float* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Device>::MPgemm<double, double, float>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const double* const a, const int lda,
    const double* const b, const int ldb, const double beta, float* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Device>::MPgemm<float, float, double>(
    const char transa, const char transb, const int m, const int n, const int k,
    const double alpha, const float* const a, const int lda,
    const float* const b, const int ldb, const double beta, double* const c,
    const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Device>::MPgemmNN<float, double, float>(
    const int m, const int n, const int k, const double alpha,
    const float* const a, const int lda, const double* const b, const int ldb,
    const double beta, float* const c, const int ldc);
template void
LinearAlgebraUtils<MemorySpace::Device>::MPgemmNN<double, double, double>(
    const int m, const int n, const int k, const double alpha,
    const double* const a, const int lda, const double* const b, const int ldb,
    const double beta, double* const c, const int ldc);
#endif
