
// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/unmqr.hpp>

#include "geqp2.hpp"

// C++ headers
#include <algorithm>
#include <chrono>  // for high_resolution_clock
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

enum class PermuteTarget { Rows, Columns };

/**
 * @brief Permutes either the rows or columns of a matrix in-place using a
 * SPECIFIC permutation vector.
 *
 * @note The `perm` vector is taken by value because the cycle-following
 * algorithm destructively modifies the tracking array to achieve O(1) auxiliary
 * space.
 */
template <typename MatrixType, typename idx_t>
void permute_matrix(MatrixType& A,
                    idx_t m,
                    idx_t n,
                    PermuteTarget target,
                    std::vector<idx_t> perm)
{
    bool is_row = (target == PermuteTarget::Rows);
    idx_t primary_dim = is_row ? m : n;
    idx_t secondary_dim = is_row ? n : m;

    // Runtime safety check
    if (perm.size() != static_cast<size_t>(primary_dim)) {
        throw std::invalid_argument(
            "Permutation vector size must match the target dimension.");
    }

    // In-place cycle-following loop
    for (idx_t i = 0; i < primary_dim; ++i) {
        while (perm[i] != static_cast<int>(i)) {
            int target_idx = perm[i];

            // Swap entire row or column vector elements
            for (idx_t j = 0; j < secondary_dim; ++j) {
                if (is_row) {
                    std::swap(A(i, j), A(target_idx, j));
                }
                else {
                    std::swap(A(j, i), A(j, target_idx));
                }
            }
            std::swap(perm[i], perm[target_idx]);
        }
    }
}

/**
 * @brief Permutes either the rows or columns of a matrix in-place.
 *
 * @tparam MatrixType A matrix type implementing mutable references via
 * A_orig(i, j)
 * @tparam IdxType    The integer type used for indexing
 * @tparam URBG       A Uniform Random Bit Generator type (e.g., std::mt19937)
 *
 * @param A_orig      The matrix to be permuted in-place
 * @param m           Number of rows
 * @param n           Number of columns
 * @param target      Whether to permute Rows or Columns
 * @param gen         The random number engine passed by the caller
 */
template <typename MatrixType, typename idx_t, typename URBG>
void permute_matrix(
    MatrixType& A, idx_t m, idx_t n, PermuteTarget target, URBG&& gen)
{
    // 1. Determine dimensions based on target
    bool is_row = (target == PermuteTarget::Rows);
    idx_t primary_dim = is_row ? m : n;
    idx_t secondary_dim = is_row ? n : m;

    // 2. Generate random permutation vector using the passed generator
    std::vector<int> p(primary_dim);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), gen);

    // 3. In-place cycle-following loop
    for (idx_t i = 0; i < primary_dim; ++i) {
        while (p[i] != static_cast<int>(i)) {
            int target_idx = p[i];

            // Swap entire row or column vector elements
            for (idx_t j = 0; j < secondary_dim; ++j) {
                if (is_row) {
                    std::swap(A(i, j), A(target_idx, j));
                }
                else {
                    std::swap(A(j, i), A(j, target_idx));
                }
            }
            std::swap(p[i], p[target_idx]);
        }
    }
}

//-----------------------------------------------------------------------
// Print matrix A in the standard output
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
    std::cout << std::endl;
}

//---------------------------------------------------------------------------
template <typename matrixA_t,
          typename matrixV_t,
          typename vectorT_t,
          typename vectorP_t>
inline void check(matrixA_t& A,
                  matrixV_t& V,
                  const vectorT_t& tau,
                  const vectorP_t& perm,
                  const std::string& matrix_name)

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

    // Compute the Frobenius Norm of A
    real_t normA = tlapack::lange(tlapack::FROB_NORM, A_perm);

    // Copy the upper-triangular part of V into R.
    // Zero the full matrix first so lower triangle does not contain garbage.
    std::vector<T> R_(n * n);
    auto R = new_matrix(R_, n, n);
    tlapack::laset(tlapack::GENERAL, T(0.0), T(0.0), R);
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, V, R);

    // Form the explicit orthogonal matrix Q from the Householder factors.
    std::vector<T> Q_(m * n);
    auto Q = new_matrix(Q_, m, n);
    tlapack::lacpy(tlapack::GENERAL, V, Q);
    tlapack::ung2r(Q, tau);

    // Compute the reconstruction accuracy ||AP - Q*R||.
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(-1.0), Q, R,
                  T(1.0), A_perm);

    real_t recon_error = tlapack::lange(tlapack::FROB_NORM, A_perm);

    // Compute the orthogonality error ||I - Q^H Q||.
    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);
    tlapack::laset(tlapack::GENERAL, T(0.0), T(1.0), work);
    tlapack::gemm(tlapack::Op::Trans, tlapack::Op::NoTrans, T(-1.0), Q, Q,
                  T(1.0), work);
    real_t orthogonality_error = tlapack::lange(tlapack::FROB_NORM, work);

    std::cout << std::endl
              << "Reconstruction error ||AP - Q*R||_F / ||A||_F for "
              << matrix_name << " = " << recon_error / normA << std::endl;
    std::cout << "Orthogonality error ||I - Q^H Q||_F for " << matrix_name
              << " = " << orthogonality_error / static_cast<real_t>(n)
              << std::endl;
}

//----------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t r, size_t e)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // 1. Initialize the random engine ONCE (e.g., at application startup)
    std::random_device rd;
    std::mt19937 gen(rd());

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Arrays
    std::vector<T> tau_drmac(n);
    std::vector<idx_t> perm_drmac(n - e);
    std::iota(perm_drmac.begin(), perm_drmac.end(), 0);
    std::vector<idx_t> perm_row(m);
    std::iota(perm_row.begin(), perm_row.end(), 0);
    // creates tau_head that contains columns 1-e
    std::vector<T> tau_head(e);

    std::vector<T> tau_tail(n - e);

    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> A_drmac_;
    auto A_drmac = new_matrix(A_drmac_, m, n);
    std::vector<T> S_;
    auto S = new_matrix(S_, m, r);
    std::vector<T> C_;
    auto C = new_matrix(C_, r, n);
    std::vector<T> W_;
    auto W = new_matrix(W_, m, n);

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A_orig(i, j) = static_cast<T>(0xDEADBEEF);
        }
        tau_drmac[j] = static_cast<T>(0xFFBADD11);
    }

    tlapack::laset(tlapack::Uplo::General, static_cast<T>(0), static_cast<T>(0),
                   S);
    tlapack::laset(tlapack::Uplo::General, static_cast<T>(0), static_cast<T>(0),
                   C);

    // Create random matrices S and C
    for (size_t j = 0; j < r; ++j) {
        for (size_t i = 0; i < m; ++i)
            S(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        for (size_t i = 0; i < n; ++i)
            C(j, i) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }

    // Multiplying Matrices S and C to create the Kahan matrix
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1), S, C, T(0),
                  A_orig);

    // // std::cout << "Before perm: " << std::endl;
    // // printMatrix(A_orig);
    // // Column Permutation
    // std::random_device rand1;
    // std::mt19937 gen_col(rand1());
    // std::vector<int> col_permutaton(n);
    // std::iota(col_permutaton.begin(), col_permutaton.end(), 0);
    // std::shuffle(col_permutaton.begin() + (e + 1), col_permutaton.end(),
    //              gen_col);
    // // Permutation Loop

    // for (idx_t j = e + 1; j < n; ++j) {
    //     while (col_permutaton[j] != j) {
    //         int current_col = col_permutaton[j];
    //         for (idx_t i = 0; i < m; ++i) {
    //             std::swap(A_orig(i, j), A_orig(i, current_col));
    //         }
    //         std::swap(col_permutaton[j], col_permutaton[current_col]);
    //     }
    // }

    // // std::cout << "After perm: " << std::endl;
    // // printMatrix(A_orig);
    // // Row Permutation
    // std::random_device rand2;
    // std::mt19937 gen_row(rand2());
    // std::vector<int> row_permutation(m);
    // std::iota(row_permutation.begin(), row_permutation.end(), 0);
    // std::shuffle(row_permutation.begin(), row_permutation.end(), gen_row);
    // // Permutation Loop
    // for (idx_t i = 0; i < m; ++i) {
    //     while (row_permutation[i] != i) {
    //         int current_row = row_permutation[i];
    //         for (idx_t j = 0; j < n; ++j) {
    //             std::swap(A_orig(i, j), A_orig(current_row, j));
    //         }
    //         std::swap(row_permutation[i], row_permutation[current_row]);
    //     }
    // }

    permute_matrix(A_orig, m, n, PermuteTarget::Columns, gen);
    permute_matrix(A_orig, m, n, PermuteTarget::Rows, gen);

    // Fills up tau_head
    for (idx_t i = 0; i < e; ++i)
        tau_head[i] = tau_drmac[i];
    // Fills up tau_tail
    for (idx_t i = e; i < n; ++i)
        tau_tail[i - e] = tau_drmac[i];

    std::cout << "A = " << std::endl;
    printMatrix(A_orig);
    std::cout << std::endl;

    tlapack::lacpy(tlapack::GENERAL, A_orig, W);

    // std::cout << "W Before QR: " << std::endl;
    // printMatrix(W);
    tlapack::geqr2(W, tau_drmac);
    std::cout << "W after QR: " << std::endl;
    printMatrix(W);
    std::cout << std::endl;
    // std::cout << "After Row perm: " << std::endl;
    // printMatrix(A_orig);

    auto A_1 = slice(A_orig, range(0, m), range(0, e));
    tlapack::geqr2(A_1, tau_head);
    std::cout << "After QR: " << std::endl;
    printMatrix(A_orig);
    std::cout << std::endl;

    auto A_2 = slice(A_orig, range(0, m), range(e, n));
    tlapack::unmqr(tlapack::Side::Left, tlapack::Op::ConjTrans, A_1, tau_head,
                   A_2);
    // std::cout << "after UNMQR" << std::flush;

    std::cout << "After unmqr: " << std::endl;
    printMatrix(A_orig);
    std::cout << std::endl;

    auto A_22 = slice(A_orig, range(e, m), range(e, n));
    std::cout << "A_22 Before drmac" << std::endl;
    printMatrix(A_22);
    std::cout << std::endl;
    geqp2(A_22, tau_tail, perm_drmac);
    std::cout << "A_22 after drmac: " << std::endl;
    printMatrix(A_22);
    std::cout << std::endl;

    // create a slice (e,m) (e,n) then permute it
    auto r_update = slice(A_orig, range(0, e), range(e, n));
    std::cout << "r_update: " << std::endl;
    printMatrix(r_update);
    std::cout << std::endl;

    std::cout << "perm_drmac: ";
    for (idx_t i = 0; i < n - e; i++)
        std::cout << perm_drmac[i] << " ";
    std::cout << std::endl;
    permute_matrix(r_update, e, n-e, PermuteTarget::Columns, perm_drmac);

    // // Apply the same column pivoting from A_22 to the corresponding columns
    // in r_update. for (idx_t j = 0; j < n - e; ++j) {
    //     while (perm_drmac[j] != j) {
    //         int current_col1 = perm_drmac[j];
    //         for (idx_t i = 0; i < e; ++i) {
    //             std::swap(r_update(i, j), r_update(i, current_col1));
    //         }
    //         std::swap(perm_drmac[j], perm_drmac[current_col1]);
    //     }
    // }
    std::cout << std::endl;
    std::cout << "r_update after permutation: " << std::endl;
    printMatrix(r_update);
    std::cout << std::endl;

    auto Y = slice(A_orig, range(e, m), range(e, n));
    std::cout << "Before drmac: " << std::endl;
    printMatrix(Y);
    std::cout << std::endl;
    geqp2(Y, tau_drmac, perm_drmac);
    std::cout << "After drmac: " << std::endl;
    printMatrix(Y);
    std::cout << std::endl;
    // Copy original matrix to the factorization input.
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A_drmac(i, j) = A_orig(i, j);

    // Call to geqp2
    geqp2(A_drmac, tau_drmac, perm_drmac);

    // Calculate the Rank
    unsigned int rank = 0;
    // Set ratio to successive terms. Once that ratio
    const real_t eps = tlapack::ulp<real_t>();

    // tolerance is set to max(m,n) * epsilon * R_11
    real_t tol = std::max(m, n) * eps * std::abs(A_drmac(0, 0));
    for (idx_t i = 0; i < std::min(m, n); i++) {
        if (std::abs(A_drmac(i, i)) > tol) {
            rank++;
        }
        else {
            break;
        }
    }
    std::cout << "rank of A = " << rank << std::endl;

    // Call to check function
    check(A_orig, A_drmac, tau_drmac, perm_drmac, "Drmac");
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, r, e;

    // Default arguments
    m = (argc < 2) ? 6 : atoi(argv[1]);
    n = (argc < 3) ? 6 : atoi(argv[2]);
    r = (argc < 4) ? 4 : atoi(argv[3]);
    e = (argc < 5) ? 3 : atoi(argv[4]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    // printf("run< float  >( %d, %d )", m, n);
    // run<float>(m, n);
    // printf("-----------------------\n");

    printf("run< double >( %d, %d, %d, %d )", m, n, r, e);
    run<double>(m, n, r, e);
    printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", m, n);
    // run<long double>(m, n);
    // printf("-----------------------\n");

    return 0;
}
