#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderSampleWeighting : public ::testing::Test {

public:
    int image_size;
    int support;
    double half_base_width;
    bool kernel_exact;
    int oversampling;
    bool pad;
    bool normalize;
    arma::mat uv;
    arma::cx_mat vis;
    arma::mat vis_weights;

    void SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 0.5;
        kernel_exact = true;
        oversampling = 1;
        pad = false;
        normalize = true;
        uv = {
            { -2., 0 },
            { -2., 0 }
        };
    }

    std::pair<arma::Mat<cx_real_t>, arma::Mat<cx_real_t>> run_gridder()
    {
        // The last two parameters (false, false) force the use of the full gridder without uv shifting
        GridderOutput res = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, uv, vis, vis_weights, kernel_exact, oversampling, pad, normalize, false, false);
        return std::make_pair(std::move(static_cast<arma::Mat<cx_real_t>>(res.vis_grid)), std::move(static_cast<arma::Mat<cx_real_t>>(res.sampling_grid)));
    }
};

// Sanity check that the weights are having some effect
TEST_F(GridderSampleWeighting, zero_weighting)
{
    kernel_exact = true;
    oversampling = 1;
    double vis_amplitude = 42.123;
    vis = vis_amplitude * arma::ones<arma::cx_mat>(uv.n_rows, 1);
    vis_weights = arma::zeros<arma::mat>(arma::size(vis));

    // Exact gridding
    std::pair<arma::Mat<cx_real_t>, arma::Mat<cx_real_t>> result = run_gridder();

    arma::Mat<cx_real_t> zeros_array(image_size, image_size);
    zeros_array.zeros();

    EXPECT_TRUE(arma::accu(vis) != 0.);
    EXPECT_TRUE(arma::approx_equal(result.first, zeros_array, "absdiff", fptolerance));

    kernel_exact = false;
    oversampling = 5;
    // kernel-cache
    result = run_gridder();

    EXPECT_TRUE(arma::approx_equal(result.first, zeros_array, "absdiff", fptolerance));
}

// Confirm natural weighting works as expected in most basic non-zero case
TEST_F(GridderSampleWeighting, natural_weighting)
{
    kernel_exact = true;
    oversampling = 1;
    vis = arma::cx_vec{ 3. / 2., 3. };
    vis_weights = arma::vec{ 2., 3. };

    // Exact gridding
    std::pair<arma::Mat<cx_real_t>, arma::Mat<cx_real_t>> result = run_gridder();

    cx_real_t natural_weighted_sum = static_cast<cx_real_t>(arma::accu(arma::dot(vis, vis_weights)) / arma::accu(vis_weights));

    arma::Mat<cx_real_t> expected_sample_locations = {
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 1., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. }
    };

    EXPECT_TRUE(arma::approx_equal(expected_sample_locations, result.second / arma::accu(result.second), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(expected_sample_locations * natural_weighted_sum, result.first / arma::accu(result.second), "absdiff", fptolerance));
}
