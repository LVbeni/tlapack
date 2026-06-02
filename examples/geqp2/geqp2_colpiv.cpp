// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>  //get rid of?
#include <tlapack/lapack/unmqr.hpp>

// C++ headers
#include <algorithm>
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <numeric>
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
void run(size_t m, size_t n, size_t r)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = true;

    // Arrays
    std::vector<T> tau(n);
    n = m;
    r = n;
    // Matrix
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<idx_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<T> A_perm_;
    auto A_perm = new_matrix(A_perm_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> B_;
    auto B = new_matrix(B_, m, n);
    std::vector<T> C_;
    auto C = new_matrix(C_, m, r);
    std::vector<T> D_;
    auto D = new_matrix(D_, r, n);

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<T>(0xDEADBEEF);
        }
        tau[j] = static_cast<T>(0xFFBADD11);
    }

    // Generate two random matricies C, size mxr and D, size rxn,
    // and compute A_orig = C*D, which has rank r.
    // for (size_t j = 0; j < r; ++j) {
    //     for (size_t i = 0; i < m; ++i)
    //         C(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    // }
    // for (size_t j = 0; j < n; ++j) {
    //     for (size_t i = 0; i < r; ++i)
    //         D(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    // }

    tlapack::laset(tlapack::Uplo::General, static_cast<T>(0), static_cast<T>(0),
                   C);
    tlapack::laset(tlapack::Uplo::General, static_cast<T>(0), static_cast<T>(0),
                   D);

    T cosine = static_cast<T>(-1.0 * cos(M_PI / 4));
    T sine = static_cast<T>(sin(M_PI / 4));

    tlapack::laset(tlapack::Uplo::Upper, cosine, static_cast<T>(1.0), D);

    // set the diagonal of C to an increment of (pi/4)^2
    for (size_t i = 0; i < r; ++i)
        C(i, i) = static_cast<T>(pow(sine, i));

    // Print matrices C and D for debugging
    std::cout << std::endl << "Matrix C =";
    printMatrix(C);
    std::cout << std::endl << "Matrix D =";
    printMatrix(D);

    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1), C, D, T(0),
                  A_orig);

    // copy matrix A_orig into matrix A
    tlapack::lacpy(tlapack::GENERAL, A_orig, A);

    // copy matrix A_orig into matrix B
    tlapack::lacpy(tlapack::GENERAL, A_orig, B);

    auto A_j = slice(A, range(0, m), 0);
    auto col_k = slice(A, range(0, m), 0);
    auto A_trailing = slice(A, range(0, m), range(0, n));

    idx_t pivot = 0;
    real_t max_norm = static_cast<real_t>(0);
    real_t norm = static_cast<real_t>(0);

    std::vector<T> s(n);

    // Start of the sorting and permutation loop
    for (size_t k = 0; k < n - 1; ++k) {
        if (verbose) {
            std::cout << std::endl << "A before pivoting =";
            printMatrix(A);
        }

        // Find the pivot column with largest norm in the active block.
        pivot = k;
        max_norm = static_cast<real_t>(0);
        norm = static_cast<real_t>(0);

        if (k == 0) {
            for (idx_t j = k; j < n; ++j) {
                A_j = slice(A, range(0, m), j);
                s[j] = tlapack::nrm2(A_j);
            }
        }
        else {
            for (idx_t j = k; j < n; ++j) {
                s[j] -= (A(k, j) * A(k, j));
            }
        }

        max_norm = s[k];
        pivot = k;
        for (idx_t j = k + 1; j < n; ++j) {
            if (s[j] > max_norm) {
                max_norm = s[j];
                pivot = j;
            }
        }

        if (verbose) {
            std::cout << std::endl
                      << "Pivot selected for column " << k << " = " << pivot
                      << " with norm = " << max_norm;
        }

        //
        if (pivot != k) {
            for (idx_t i = 0; i < m; ++i) {
                std::swap(A(i, k), A(i, pivot));
                std::swap(s[k], s[pivot]);
            }
            std::swap(perm[k], perm[pivot]);
            // Swap the corresponding columns in B as well to keep track of the
            // original matrix
            for (idx_t i = 0; i < static_cast<idx_t>(m); ++i)
                std::swap(B(i, k), B(i, pivot));
            // print B after swapping columns
            if (verbose) {
                std::cout << std::endl << "B after pivoting =";
                printMatrix(B);
            }
        }

        if (verbose) {
            std::cout << std::endl << "A after pivoting =";
            printMatrix(A);
            // add empty line for better readability
            std::cout << std::endl;
        }

        col_k = slice(A, range(k, m), k);
        tlapack::larfg(tlapack::Direction::Forward, tlapack::StoreV::Columnwise,
                       col_k, tau[k]);

        if (k < n - 1) {
            A_trailing = slice(A, range(k, m), range(k + 1, n));
            tlapack::larf(tlapack::Side::Left, tlapack::Direction::Forward,
                          tlapack::StoreV::Columnwise, col_k, tau[k],
                          A_trailing);
        }
    }

    tlapack::lacpy(tlapack::UPPER_TRIANGLE, A_perm, R);

    // Preserve the final permuted matrix AP before forming the QR factors.
    tlapack::lacpy(tlapack::GENERAL, B, A_perm);

    // Compute the QR factorization of the permuted matrix AP using geqr2
    // (unblocked routine). Use geqrf for the blocked alternative.
    tlapack::geqr2(B, tau);

    // Update matrix A with the final QR factorization
    tlapack::lacpy(tlapack::GENERAL, B, A);

    // Extract the upper-triangular R from the QR factorization for comparison.
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, B, R);

    if (verbose) {
        std::cout << std::endl << "AP (final permuted matrix) =";
        printMatrix(A_perm);
        std::cout << std::endl << "R from geqr2 (upper triangle) =";
        printMatrix(R);
        std::cout << std::endl << "Householder form of QR factorization =";
        printMatrix(B);
        std::cout << std::endl;

        printMatrix(A);
        std::cout << std::endl << "---end ------------------------------";
    }
}

int main(int argc, char** argv)
{
    int m, n, r;

    // Default arguments
    m = (argc < 2) ? 6 : atoi(argv[1]);
    n = (argc < 3) ? 6 : atoi(argv[2]);
    r = (argc < 4) ? 3 : atoi(argv[3]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d, %d )", m, n, r);
    run<float>(m, n, r);
    printf("-----------------------\n");

    // printf("run< double >( %d, %d )", m, n);
    // run<double>(m, n);
    // printf("-----------------------\n");

    // printf("run< complex<float> >( %d, %d )", m, n);
    // run<std::complex<float>>(m, n);
    // printf("-----------------------\n");

    // printf("run< complex<double> >( %d, %d )", m, n);
    // run<std::complex<double>>(m, n);
    // printf("-----------------------\n");

    // printf("run< complex<long double> >( %d, %d )", m, n);
    // run<std::complex<long double>>(m, n);
    // printf("-----------------------\n");

    return 0;
}

//