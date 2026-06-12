#ifndef TLAPACK_GEQP2_HH
#define TLAPACK_GEQP2_HH

#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/base/utils.hpp"



namespace tlapack {

    /** Creates a QR factorization with Column Pivoting of a matrix A
    * The matrix Q is represented as a product of elementary reflectors
    * 
    * @param[in, out] A m-by-n matrix.
    *
    * @param [out] tau
    *
    * @param [out] p
    *
    * @return 0 if success
    *
    * @ingroup
    */

    template<TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t, TLAPACK_VECTOR idx_vector_t>
    int geqp2(matrix_t& A, vector_t& tau, idx_vector_t& p)
    {
        using idx_t = size_type<matrix_t>;
        using range = pair<idx_t, idx_t>;
        using T = type_t<matrix_t>;
        using real_t = real_type<T>;

        // constants
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);
        const real_t eps = ulp<real_t>();
        const real_t safety_threshold = std::sqrt(eps);

        // Tracking vectors for column norms
        std::vector<real_t> vn1(n);
        std::vector<real_t> vn2(n);

        // Initialize column norms for the active block
        for(idx_t j = 0; j < n; ++j) {
            auto col = slice(A, range{0, m}, j);
            vn1[j] = nrm2(col);
            vn2[j] = vn1[j];
        }

        // Main Computational Loop
        for(idx_t k = 0; k < std::min(m, n); ++k) {
            // Identify pivot column from non-resolved active columns
            idx_t pivot = k;
            real_t max_norm = vn1[k];
            for(idx_t j = k + 1; j < n; ++j) {
                if(vn1[j] > max_norm) {
                    max_norm = vn1[j];
                    pivot = j;
                }
            }
            // Perform swap if necessary
            if(pivot != k) {
                for(idx_t i = 0; i < m; ++i) {
                    std::swap(A(i, k), A(i, pivot));
                }
                std::swap(p[k], p[pivot]);
                std::swap(vn1[k], vn1[pivot]);
                std::swap(vn2[k], vn2[pivot]);
            }
            // Generate Householder reflector for the k-th column
            auto col_k = slice(A, range{k, m}, k);
            larfg(FORWARD, COLUMNWISE_STORAGE, col_k, tau[k]);
            // Apply the reflector to the remaining columns
            if(k < n - 1) {
                auto A_trailing = slice(A, range{k, m}, range{k + 1, n});
                larf(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, col_k, tau[k], A_trailing);
            }
            // Update the norms of the remaining columns
            for (idx_t j = k + 1; j < n; ++j) {
                if (vn1[j] > static_cast<real_t>(0.0)) {
                    // t = |A(k,j)| / vn1[j]
                    real_t t = abs(A(k, j)) / vn1[j];
                    // t = max(0, 1 - t^2)
                    t = std::max(static_cast<real_t>(0.0), (static_cast<real_t>(1.0) - t * t));
                    // t2 = t * (vn1[j] / vn2[j])^2
                    real_t norm_ratio = vn1[j] / vn2[j];
                    real_t t2 = t * (norm_ratio * norm_ratio);
                    // Safety check
                    if (t2 <= safety_threshold) {
                        auto A_col = slice(A, range(k + 1, m), j);
                        vn1[j] = nrm2(A_col);
                        vn2[j] = vn1[j]; // Reset base 
                    }
                    else {
                        vn1[j] *= std::sqrt(t);
                    }
                }
            }
           
        }

    return 0;
    }

} // namespace tlapack

#endif  // TLAPACK_GEQP2_HH