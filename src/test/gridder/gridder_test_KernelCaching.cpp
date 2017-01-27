#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

/**
 * Test generation of cached (offset) kernels, and demonstrate correct usage.
 *
 *  In this test, we assume an oversampling of 5, resulting in
 *  step-widths of 0.2 regular pixels. We then iterate through a bunch of
 *  possible sub-pixel offsets, checking that we pick the nearest (closest to
 *  exact-positioned) cached kernel correctly.
 */

const double eps = 10e-9;

int support = 3;
int n_image = 8;
double half_base_width = 2.5;
const int oversampling = 5;

TEST(GridderKernelCaching, equal)
{
    arma::mat steps = {
        { -0.4, 0.2, 0.0, 0.2, 0.4 }
    };
    arma::mat substeps = arma::linspace(-0.099999, 0.099999, 15);

    Triangle triangle(half_base_width);
    arma::field<arma::mat> kernel_cache = populate_kernel_cache(triangle, support, oversampling);

    for (arma::uword i = 0; i < steps.n_elem; ++i) {
        arma::mat offset = { steps[i], 0.0 };
        arma::mat aligned_exact_kernel = make_kernel_array(triangle, support, offset);
        // Generate an index into the kernel-cache at the precise offset
        // (i.e. a multiple of 0.2-regular-pixel-widths)
        arma::imat aligned_cache_idx = calculate_oversampled_kernel_indices(offset, oversampling);
        arma::mat cached_kernel = kernel_cache(aligned_cache_idx.at(0, 1) + (oversampling / 2), aligned_cache_idx.at(0, 0) + (oversampling / 2));

        EXPECT_TRUE(arma::approx_equal(aligned_exact_kernel, cached_kernel, "absdiff", tolerance));

        for (arma::uword j = 0; j < substeps.n_elem; ++j) {
            arma::mat s_offset = { offset[0] + substeps[j], 0.0 };
            if (std::abs(substeps[j]) > 0.0) {
                arma::mat unaligned_exact_kernel = make_kernel_array(triangle, support, s_offset);

                // Check that the irregular position resolves to the correct nearby aligned position:
                arma::imat unaligned_cache_idx = calculate_oversampled_kernel_indices(s_offset, oversampling);
                EXPECT_TRUE(arma::approx_equal(unaligned_cache_idx, aligned_cache_idx, "absdiff", tolerance));

                // Demonstrate retrieval of the cached kernel:
                arma::mat cached_kernel = kernel_cache(unaligned_cache_idx.at(0, 1) + (oversampling / 2), unaligned_cache_idx.at(0, 0) + (oversampling / 2));
                EXPECT_TRUE(arma::approx_equal(aligned_exact_kernel, cached_kernel, "absdiff", tolerance));

                // Sanity check - we expect the exact-calculated kernel to be different by a small amount
                arma::mat diff = arma::abs(aligned_exact_kernel - unaligned_exact_kernel);
                EXPECT_TRUE(arma::accu(arma::find(diff > eps)) > 0);
            }
        }
    }
}