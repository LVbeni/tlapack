
// The Q from Businger and Golub algorithm can be very inaccurate, even for small matrices.
// Th reconstruction error for drmac is wrong

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>  //get rid of?

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
// check function used in Stewarts algorithm
template <typename matrixA_t,
          typename matrixV_t,
          typename vectorT_t,
          typename vectorP_t>
inline void check(matrixA_t& A,
                  matrixV_t& V,
                  const vectorT_t& tau,
                  const vectorP_t& perm)

{
    using T = typename tlapack::type_t<matrixA_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrixA_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    // Functors for creating new matrices
    tlapack::Create<matrixA_t> new_matrix;

    // Permute the columns of A according to the pivot order in perm.
    std::vector<T> A_perm_(m * n);
    auto A_perm = new_matrix(A_perm_, m, n);
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A_perm(i, j) = A(i, perm[j]);
        }
    }

    // Copy the upper-triangular part of V into R.
    // Zero the full matrix first so lower triangle does not contain garbage.
    std::vector<T> R_(n * n);
    auto R = new_matrix(R_, n, n);
    tlapack::laset(tlapack::GENERAL, T(0.0), T(0.0), R);
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, V, R);

    // print the upper-triangular R from qrbg
    // std::cout << std::endl << "R_check =";
    // printMatrix(R);
    // std::cout << std::endl;

    // Form the explicit orthogonal matrix Q from the Householder factors.
    std::vector<T> Q_(m * m);
    auto Q = new_matrix(Q_, m, m);
    tlapack::lacpy(tlapack::GENERAL, V, Q);
    tlapack::ung2r(Q, tau);

    // print the orthogonal Q from qrbg
    std::cout << std::endl << "Q_check =";
    printMatrix(Q);
    std::cout << std::endl;

    // Compute the reconstruction accuracy ||AP - Q*R||.
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(-1.0), Q, R,
                  T(1.0), A_perm);

    real_t recon_error = tlapack::lange(tlapack::FROB_NORM, A_perm);

    // Compute the orthogonality error ||I - Q^H Q||.
    std::vector<T> work_;
    auto work = new_matrix(work_, m, m);
    tlapack::laset(tlapack::GENERAL, T(0.0), T(1.0), work);
    tlapack::gemm(tlapack::Op::Trans, tlapack::Op::NoTrans, T(-1.0), Q, Q,
                  T(1.0), work);
    real_t orthogonality_error = tlapack::lange(tlapack::FROB_NORM, work);

    std::cout << std::endl
              << "Reconstruction error ||AP - Q*R|| = " << recon_error
              << std::endl;
    std::cout << "Orthogonality error ||I - Q^H Q|| = " << orthogonality_error
              << std::endl;
}
//------------------------------------------------------------------------------
// Stewart's algorithm for QR factorization with column pivoting
template <typename matrixA_t, typename vectorT_t, typename vectorP_t>
inline void qr_stewart(matrixA_t& A, vectorT_t& tau, vectorP_t& perm)
{
    using T = typename tlapack::type_t<matrixA_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrixA_t> new_matrix;

    // Initialize Stewart's dual vector tracking structures gamma and gamma_init
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);
    std::vector<real_t> gamma(n);
    std::vector<real_t> gamma_init(n);
    for (idx_t j = 0; j < n; ++j) {
        auto A_j = slice(A, range(0, m), j);
        gamma[j] = tlapack::nrm2(A_j);
        gamma_init[j] = gamma[j];
    }
    // print the initial gamma vector
    std::cout << std::endl << "gamma_init =";
    for (idx_t j = 0; j < n; ++j) {
        std::cout << " " << gamma_init[j];
    }
    std::cout << std::endl;

    // Identify the pivot column (maximum remaining norm)
    idx_t pivot = 0;
    real_t max_norm = static_cast<real_t>(0);

    // Standard LAPACK guard thresholds
    const real_t eps = std::numeric_limits<real_t>::epsilon();
    const real_t tau1 = std::sqrt(eps);
    const real_t tau2 = static_cast<real_t>(0.05);

    for (idx_t k = 0; k < std::min(m, n); ++k) {
        // Identify the pivot column (maximum remaining norm)
        pivot = k;
        max_norm = gamma[k];
        for (idx_t j = k + 1; j < n; ++j) {
            if (gamma[j] > max_norm) {
                max_norm = gamma[j];
                pivot = j;
            }
        }

        // Swap columns globally across A, tau, perm, and tracking vectors.
        if (pivot != k) {
            for (idx_t i = 0; i < m; ++i) {
                std::swap(A(i, k), A(i, pivot));
            }
            std::swap(tau[k], tau[pivot]);
            std::swap(perm[k], perm[pivot]);
            std::swap(gamma[k], gamma[pivot]);
            std::swap(gamma_init[k], gamma_init[pivot]);
        }

        // Compute the Householder reflector to zero out under diagonal
        // elements.
        auto col_k = slice(A, range(k, m), k);
        tlapack::larfg(tlapack::Direction::Forward, tlapack::StoreV::Columnwise,
                       col_k, tau[k]);

        // Apply reflector matrix to the trailing matrix A.
        if (k < n - 1) {
            auto A_trailing = slice(A, range(k, m), range(k + 1, n));
            tlapack::larf(tlapack::Side::Left, tlapack::Direction::Forward,
                          tlapack::StoreV::Columnwise, col_k, tau[k],
                          A_trailing);
        }

        // Process Stewart's Dual-Vector Tracking Guard for trailing columns.
        for (idx_t j = k + 1; j < n; ++j) {
            if (gamma[j] > static_cast<real_t>(0.0)) {
                const T r_kj = A(k, j);
                const real_t ratio = tlapack::abs(r_kj) / gamma[j];
                real_t gamma_new = static_cast<real_t>(0.0);

                if (ratio < static_cast<real_t>(1.0)) {
                    gamma_new = gamma[j] * std::sqrt(static_cast<real_t>(1.0) -
                                                     ratio * ratio);
                }

                // TEST 1: Absolute / Historical Decay Guard
                if (gamma_new / gamma_init[j] <= tau1) {
                    auto A_col = slice(A, range(k + 1, m), j);
                    gamma[j] = tlapack::nrm2(A_col);
                    gamma_init[j] = gamma[j];
                }
                // TEST 2: Local Step Perturbation Guard
                else if (gamma_new / gamma[j] <= tau2) {
                    auto A_col = slice(A, range(k + 1, m), j);
                    gamma[j] = tlapack::nrm2(A_col);
                }
                else {
                    gamma[j] = gamma_new;
                }
            }
        }
    }
}
//------------------------------------------------------------------------------
template <typename matrix_t, typename T>
inline void qr_bg(matrix_t& A, std::vector<T>& tau, std::vector<size_t>& perm)
{
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    auto A_j = slice(A, range(0, m), 0);
    auto col_k = slice(A, range(0, m), 0);
    auto A_trailing = slice(A, range(0, m), range(0, n));

    idx_t pivot = 0;
    real_t max_norm = static_cast<real_t>(0);
    real_t norm = static_cast<real_t>(0);

    std::vector<T> s(n);

    // Start of the sorting and permutation loop
    for (size_t k = 0; k < n; ++k) {
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

        //
        if (pivot != k) {
            for (idx_t i = 0; i < m; ++i) {
                std::swap(A(i, k), A(i, pivot));
            }
            std::swap(s[k], s[pivot]);
            std::swap(perm[k], perm[pivot]);
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
}
//------------------------------------------------------------------------------
// Implementation of the laqps
template <typename matrixA_t,
          typename matrixV_t,
          typename vectorT_t,
          typename vectorP_t>
inline void laqps(
    matrixA_t& A, matrixV_t& V, vectorT_t& tau, vectorP_t& perm, size_t offset)
{
    using T = typename tlapack::type_t<matrixA_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    // Machine precision constants
    const real_t eps = std::numeric_limits<real_t>::epsilon();
    const real_t safety_threshold = std::sqrt(eps);  // √e guard from Table 3

    // Tracking norm vectors
    std::vector<real_t> vn1(n);  // Current partial norms (ω)
    std::vector<real_t> vn2(n);  // Original initial column norms (ν)

    // Initialize norms for the active block starting from 'offset'
    for (idx_t j = offset; j < n; ++j) {
        auto A_j = slice(A, range(offset, m), j);
        vn1[j] = tlapack::nrm2(A_j);
        vn2[j] = vn1[j];
    }

    // List to accumulate columns experiencing severe cancellations
    std::vector<idx_t> unresolved_columns;

    for (idx_t k = offset; k < std::min(m, n); ++k) {
        // 1. Identify pivot column from non-unresolved active columns
        idx_t pivot = k;
        real_t max_norm = vn1[k];
        for (idx_t j = k + 1; j < n; ++j) {
            if (vn1[j] > max_norm) {
                max_norm = vn1[j];
                pivot = j;
            }
        }

        // 2. Perform swap if necessary
        if (pivot != k) {
            for (idx_t i = 0; i < m; ++i) {
                std::swap(A(i, k), A(i, pivot));
            }
            std::swap(tau[k], tau[pivot]);
            std::swap(perm[k], perm[pivot]);
            std::swap(vn1[k], vn1[pivot]);
            std::swap(vn2[k], vn2[pivot]);
        }

        // 3. Generate Householder reflector for column k
        auto col_k = slice(A, range(k, m), k);
        tlapack::larfg(tlapack::Direction::Forward, tlapack::StoreV::Columnwise,
                       col_k, tau[k]);

        // 4. Apply reflector matrix to trailing columns
        if (k < n - 1) {
            auto A_trailing = slice(A, range(k, m), range(k + 1, n));
            tlapack::larf(tlapack::Side::Left, tlapack::Direction::Forward,
                          tlapack::StoreV::Columnwise, col_k, tau[k],
                          A_trailing);
        }

        // 5. Update column norms using the Table 3 strategy
        unresolved_columns.clear();
        for (idx_t j = k + 1; j < n; ++j) {
            if (vn1[j] > static_cast<real_t>(0.0)) {
                // t = |A(k,j)| / vn1(j)
                real_t t = tlapack::abs(A(k, j)) / vn1[j];

                // t = max(0, 1 - t^2)
                t = std::max(static_cast<real_t>(0.0),
                             static_cast<real_t>(1.0) - t * t);

                // t2 = t * (vn1(j) / vn2(j))^2
                real_t norm_ratio = vn1[j] / vn2[j];
                real_t t2 = t * (norm_ratio * norm_ratio);

                // Table 3 Safety Check: if t2 <= √e, push to unresolved list
                if (t2 <= safety_threshold) {
                    unresolved_columns.push_back(j);
                }
                else {
                    // Safe to update using the stable scalar formula
                    vn1[j] = vn1[j] * std::sqrt(t);
                }
            }
        }

        // 6. Resolve columns that failed the switch by performing precise
        // recomputation
        for (idx_t j : unresolved_columns) {
            auto A_col = slice(A, range(k + 1, m), j);
            vn1[j] = tlapack::nrm2(A_col);
            vn2[j] = vn1[j];  // Reset base tracking metric
        }
    }
}

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
    n = m;
    r = n;
    std::vector<T> tau(n);
    std::vector<T> tau_bg(n);
    std::vector<T> tau_st(n);
    std::vector<idx_t> perm_bg(n);
    std::vector<idx_t> perm_st(n);
    std::iota(perm_bg.begin(), perm_bg.end(), 0);
    std::iota(perm_st.begin(), perm_st.end(), 0);
    // Matrix
    std::vector<T> A_bg_;
    auto A_bg = new_matrix(A_bg_, m, n);
    std::vector<T> A_st_;
    auto A_st = new_matrix(A_st_, m, n);
    std::vector<T> A_orig_;
    std::vector<T> A_drmac_;
    auto A_drmac = new_matrix(A_drmac_, m, n);
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> A_perm_;
    auto A_perm = new_matrix(A_perm_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> C_;
    auto C = new_matrix(C_, m, r);
    std::vector<T> D_;
    auto D = new_matrix(D_, r, n);
    std::vector<T> tau_qr_bg(n);
    std::vector<T> tau_qr_st(n);
    // std::vector<T> Q

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A_orig(i, j) = static_cast<T>(0xDEADBEEF);
        }
        tau[j] = static_cast<T>(0xFFBADD11);
        tau_bg[j] = static_cast<T>(0xFFBADD11);
        tau_st[j] = static_cast<T>(0xFFBADD11);
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

    T cosine = static_cast<T>(-1.0 * cos(M_PI / 12));
    T sine = static_cast<T>(sin(M_PI / 12));

    tlapack::laset(tlapack::Uplo::Upper, cosine, static_cast<T>(1.0), D);

    // set the diagonal of C to an increment of (pi/4)^2
    for (size_t i = 0; i < r; ++i)
        C(i, i) = static_cast<T>(pow(sine, i));

    // Print matrices C and D for debugging
    // std::cout << std::endl << "Matrix C =";
    // printMatrix(C);
    // std::cout << std::endl << "Matrix D =";
    // printMatrix(D);

    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1), C, D, T(0),
                  A_orig);

    // swap a random column to the front to make the example more interesting
    // using tlapack::swap
    idx_t random_col = rand() % n;
    if (random_col != 0)
        for (idx_t i = 0; i < m; ++i)
            std::swap(A_orig(i, 0), A_orig(i, random_col));

    // Copy matrix A_orig into separate matrices for each factorization.
    tlapack::lacpy(tlapack::GENERAL, A_orig, A_bg);
    tlapack::lacpy(tlapack::GENERAL, A_orig, A_st);
    tlapack::lacpy(tlapack::GENERAL, A_orig, A_drmac);

    // Compute the QR factorization of A with column pivoting using Businger and
    // Golub's algorithm.
    qr_bg(A_bg, tau_bg, perm_bg);
    check(A_orig, A_bg, tau_bg, perm_bg);

    laqps(A_drmac, A_drmac, tau_qr_bg, perm_bg, 0);
    check(A_orig, A_drmac, tau_qr_bg, perm_bg);

    // Compute the QR factorization of A with column pivoting using Stewart's
    // algorithm.
    qr_stewart(A_st, tau_st, perm_st);
    check(A_orig, A_st, tau_st, perm_st);

    // Compute the QR factorization of the Stewart-permuted matrix using geqr2.
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A_perm(i, j) = A_orig(i, perm_st[j]);
        }
    }


    // Compute the QR factorization of the Stewart-permuted matrix using geqr2.
    tlapack::geqr2(A_perm, tau);
    std::cout << std::endl << "Matrix A_perm after geqr2 =";
    printMatrix(A_perm);
    std::cout << std::endl;

    // Extract the upper-triangular R from the QR factorization for comparison.
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, A_perm, R);

    // Form the explicit orthogonal matrix Q from the Householder factors.
    tlapack::ung2r(A_perm, tau);

    // Compute the reconstruction accuracy ||AP - Q*R||.
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1), A_perm, R,
                  T(0), C);
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            C(i, j) = A_perm(i, j) - C(i, j);
        }
    }
    real_t recon_error = tlapack::lange(tlapack::Norm::One, C);

    // Compute the orthogonality error ||I - Q^H Q||.
    tlapack::syrk(tlapack::Uplo::Upper, tlapack::Op::Trans, T(1), A_perm,
                  T(0), D);
    for (idx_t j = 0; j < n; ++j)
        D(j, j) -= static_cast<T>(1);
    real_t orthogonality_error =
        tlapack::lansy(tlapack::Norm::One, tlapack::Uplo::Upper, D);

    if (verbose) {
        std::cout << std::endl << "Q from QR factorization =";
        printMatrix(A_perm);
        std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{
    int m, n, r;

    // Default arguments
    m = (argc < 2) ? 7 : atoi(argv[1]);
    n = (argc < 3) ? 7 : atoi(argv[2]);
    r = (argc < 4) ? 3 : atoi(argv[3]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< double >( %d, %d, %d )\n", m, n, r);
    run<double>(m, n, r);
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