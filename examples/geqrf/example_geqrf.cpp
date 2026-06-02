/// @file example_geqrf_preprocessed.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// This is based on Golub and Van Loan's "Matrix Computations", 3rd edition,
// section 5.7.1

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/scal.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/unmqr.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<T> tau(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> x_;
    auto x = new_matrix(x_, n, 1);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, 1);
    std::vector<T> b_orig_;
    auto b_orig = new_matrix(b_orig_, m, 1);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<T>(0xDEADBEEF);
        }
        tau[j] = static_cast<T>(0xFFBADD11);
        x(j, 0) = static_cast<T>(0xBAADF00D);
        b(j, 0) = static_cast<T>(0xBAADF00D);
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i)
            A(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        x(j, 0) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }

     if (m >= 3 && n >= 3) {
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                A(i, j) = static_cast<T>(0);

        A(0, 0) = static_cast<T>(1.0);
        A(1, 0) = static_cast<T>(2.0);
        A(2, 0) = static_cast<T>(2.0);
        A(0, 1) = static_cast<T>(-4.0);
        A(1, 1) = static_cast<T>(3.0);
        A(2, 1) = static_cast<T>(2.0);
        A(0, 2) = static_cast<T>(3.0);
        A(1, 2) = static_cast<T>(2.0);
        A(2, 2) = static_cast<T>(1.0);
    }

    // Copy A to A_orig
    tlapack::lacpy(tlapack::GENERAL, A, A_orig);

    // Generate b = A*x so that the solution is known
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                  static_cast<T>(1.0), A, x, static_cast<T>(0.0), b);

    // Copy b to b_orig
    tlapack::lacpy(tlapack::GENERAL, b, b_orig);

    // Calculate the Frobenius norm of b and store it in norm_b
    auto norm_b = tlapack::lange(tlapack::FROB_NORM, b);

    // Print A and b
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A_orig);
        std::cout << std::endl << "b = ";
        printMatrix(b_orig);
        std::cout << std::endl;
    }
    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // Compute the Householder reflector for b

        // 6*m FLOPS
        auto tau_b = static_cast<T>(0.0);
        tlapack::larfg(tlapack::Direction::Backward,
                       tlapack::StoreV::Columnwise, b_, tau_b);

        // Apply the Householder reflector to A to produce HA
        // 4*m*n FLOPS
        tlapack::larf(tlapack::Side::Left, tlapack::Direction::Backward,
                      tlapack::StoreV::Columnwise, b_, tau_b, A);

        // Compute the QR factorization of (HA)'
        // 4*m*n^2/3 FLOPS
        auto HA_T = tlapack::transpose_view(A);
        tlapack::geqrf(HA_T, tau);

        auto k = std::min(m, n);
        auto slice_HA_T = tlapack::slice(HA_T, range(0, k), range(0, k));
        auto slice_tau = tlapack::slice(tau, range(0, k));
        // Compute the last column of Q by applying the Householder reflectors
        // to the last column of the identity matrix

        // set x to be the last column of the identity matrix
        for (size_t i = 0; i < n; ++i)
            x(i, 0) = static_cast<T>(0.0);
        x(n - 1, 0) = static_cast<T>(1.0);

        // Apply the Householder reflectors to x
        // 4*m*n FLOPS
        tlapack::unmqr(tlapack::Side::Left, tlapack::Op::NoTrans, slice_HA_T,
                       slice_tau, x);

        // Scale x by -1.0 * b(m, 0) / A(m, n).  Note, larfg it introduces a
        // negative because the norm is always nonnegative.
        auto scale = static_cast<T>(-1.0) * norm_b / HA_T(n - 1, m - 1);
        // n FLOPS
        tlapack::scal(scale, x_);
    }

    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Compute FLOPS
    // 6m + 4m*n + 4m*n^2/3 + 4m*n + n = 4m*n^2/3 + 8m*n + 6m
    double flopsQR =
        (4.0e+00 / 3.0e+00 * ((double)m) * ((double)n) * ((double)n) +
         8.0e+00 * ((double)m) * ((double)n) + 6.0e+00 * ((double)m)) /
        (elapsedQR.count() * 1.0e-9);

    // Compute ||b - A*x||_F / ||A||_F
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                  static_cast<T>(-1.0), A_orig, x, static_cast<T>(1.0), b_orig);

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::FROB_NORM, A_orig);

    // Compute the relative residual norm ||b - A*x||_F / ||A||_F
    auto norm_residual = tlapack::lange(tlapack::FROB_NORM, b_orig) / normA;

    // Output

    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
              << ",   GFlop/sec = " << flopsQR * 1.0e-9;
    std::cout << std::endl;
    std::cout << "||b - A*x||_F/||A||_F = " << norm_residual;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n;

    // Default arguments
    m = (argc < 2) ? 199 : atoi(argv[1]);
    n = (argc < 3) ? 151 : atoi(argv[2]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", m, n);
    run<long double>(m, n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )", m, n);
    run<std::complex<float>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )", m, n);
    run<std::complex<double>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<long double> >( %d, %d )", m, n);
    run<std::complex<long double>>(m, n);
    printf("-----------------------\n");

    return 0;
}
