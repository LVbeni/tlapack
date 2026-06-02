// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp> //get rid of?
#include <tlapack/lapack/unmqr.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm> 

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
    bool verbose = true;

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
            A_orig(i, j) = static_cast<T>(0xCAFED00D);
        }
        tau[j] = static_cast<T>(0xFFBADD11);
        x(j, 0) = static_cast<T>(0xBAADF00D);
        b(j, 0) = static_cast<T>(0xBAADF00D);
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i)
            A(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        //(j, 0) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }

    x(0, 0) = static_cast<T>(-1.0);
    x(1, 0) = static_cast<T>(1.0);

    if (m >= 3 && n >= 2) {
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                A_orig(i, j) = static_cast<T>(0);

        A_orig(0, 0) = static_cast<T>(1.0);
        A_orig(1, 0) = static_cast<T>(2.0);
        A_orig(2, 0) = static_cast<T>(2.0);
        A_orig(0, 1) = static_cast<T>(-4.0);
        A_orig(1, 1) = static_cast<T>(3.0);
        A_orig(2, 1) = static_cast<T>(2.0);

        tlapack::lacpy(tlapack::GENERAL, A_orig, A);
    }

    // Generate b = A*x so that the solution is known
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                  static_cast<T>(1.0), A_orig, x, static_cast<T>(0.0), b); //static_cast<T>(0.0) is needed for complex types

    // Copy b to b_orig
    tlapack::lacpy(tlapack::GENERAL, b, b_orig);

    // Calculate the F norm of b and store it in norm_b
    auto norm_b = tlapack::lange(tlapack::FROB_NORM, b);

    // Use larfg to create the householder reflectors in B and the tau array
    auto tau_v = static_cast<T>(0.0);
    std::vector<T> v(m);
    // copy b into v
    for (size_t i = 0; i < m; ++i)
        v[i] = b(i, 0);
    
    tlapack::larfg(tlapack::Direction::Backward, tlapack::StoreV::Columnwise, v, tau_v);
    // print v and tau_v
    if (verbose) {
        std::cout << std::endl << "v =";
        for (auto i = 0; i < tlapack::size(v); ++i)
            std::cout << v[i] << " ";
        std::cout << std::endl << "tau_v = " << tau_v;
    }

    tlapack::larf(tlapack::Side::Left, tlapack::Direction::Backward, tlapack::StoreV::Columnwise, v, tau_v, b);
    tlapack::larf(tlapack::Side::Left, tlapack::Direction::Backward, tlapack::StoreV::Columnwise, v, tau_v, A);

    // print A and b after applying the householder reflector
    if (verbose) {
        std::cout << std::endl << "A after applying the householder reflector =";
        printMatrix(A);
    }

    //create HA_T
    std::vector<T> HA_T_;
    auto HA_T = new_matrix(HA_T_, n, m);

    // Transpose the matrix A to get the correct order of the householder reflectors in HA
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i)
            HA_T(j, i) = A(i, j);
    }
    // print HAt
    if (verbose) {
        std::cout << std::endl << "HAT =";
        printMatrix(HA_T);
    }

    //print b
    if (verbose) {
        std::cout << std::endl << "b after applying the householder reflector =";
        printMatrix(b);
    }
    
    for (size_t j = 0; j < n; ++j) {
        auto b_tail = slice(b_orig, range(j + 1, m), 0);
        tlapack::larfg(tlapack::COLUMNWISE_STORAGE, b_orig(j, 0), b_tail, tau[j]);
    }

    /*
    // reverse the order of the householder reflectors and tau array
    std::reverse(b_orig.begin(), b_orig.end());
    std::reverse(tau.begin(), tau.end());

    //print b_ and tau after reversing
    if (verbose) {
        std::cout << std::endl << "tau after reversing =";
        for (auto i = 0; i < tlapack::size(tau); ++i)
            std::cout << tau[i] << " ";
    }

    // apply the householder reflectors using LARF to the matrix A
    for (size_t j = 0; j < n; ++j) {
        auto A_block = slice(A_orig, range(j, m), range(j, n));
        auto b_vec = slice(b_orig, range(j, m), 0);
        tlapack::larf(tlapack::Side::Left, tlapack::Direction::Backward,
                      tlapack::COLUMNWISE_STORAGE, b_vec, tau[j], A_block);
    }

    // Compute the QR factorization of A using the householder reflectors in B and the tau array using geqr2
    tlapack::geqr2(A_orig, tau);

    //create the last column of Q using unmqr
    auto Q_last_col = slice(A_orig, range(0, m), range(n - 1, n));
    tlapack::unmqr(tlapack::Side::Left, tlapack::Op::NoTrans, A_orig, tau,
                   Q_last_col);

    // Compute X = (beta/r(n,n)) * Q(:,n) where beta is the first element of the last householder reflector in B and r(n,n) is the last element of R
    auto R_last_col = slice(A_orig, range(0, n), range(n - 1, n));
    auto X = slice(A_orig, range(0, m), n - 1);
    tlapack::scal(b_orig(0, 0) / R_last_col(n - 1, 0), X);

    // print A, b, x, and computed X
    if (verbose) {
        std::cout << std::endl << "A =";
        printMatrix(A_orig);
        std::cout << std::endl << "\nb =";
        printMatrix(b_orig);
        std::cout << std::endl << "\nx =";
        printMatrix(x);
        std::cout << std::endl << "\nComputed X =";
        for (auto i = 0; i < tlapack::size(X); ++i)
            std::cout << X[i] << " ";
        std::cout << std::endl;
    }
        
*/

}

int main(int argc, char** argv)
{
    int m, n;

    // Default arguments
    m = (argc < 2) ? 3 : atoi(argv[1]);
    n = (argc < 3) ? 2 : atoi(argv[2]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n);
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
