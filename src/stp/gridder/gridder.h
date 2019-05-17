/** @file gridder.h
 *  @brief Classes and function prototypes of the gridder.
 */

#ifndef GRIDDER_H
#define GRIDDER_H

#include <cassert>
#include <cfloat>
#include <map>
#include <tbb/tbb.h>

#include "../common/fft.h"
#include "../common/matstp.h"
#include "../convolution/conv_func.h"
#include "../global_macros.h"
#include "../types.h"
#include "aw_projection.h"

#define arc_sec_to_rad(value) ((value / 3600.0) * (M_PI / 180.0))

namespace stp {

/**
 * @brief The gridder output class
 *
 * Stores matrices of visibility grid and sampling grid as well as the total sum of the sampling grid
 */
class GridderOutput {
public:
    /**
     * @brief Default constructor
     */
    GridderOutput() = default;

    /**
     * @brief Constructor - initializes members
     */
    GridderOutput(MatStp<cx_real_t>& in_vis_grid, MatStp<cx_real_t>& in_sampling_grid, double in_sampling_total)
        : vis_grid(std::move(in_vis_grid))
        , sampling_grid(std::move(in_sampling_grid))
        , sample_grid_total(in_sampling_total)
    {
    }

    /**
     * The visibility grid matrix
     */
    MatStp<cx_real_t> vis_grid;
    /**
     * The sampling grid matrix
     */
    MatStp<cx_real_t> sampling_grid;
    /**
     * Sum of sampling grid values
     */
    double sample_grid_total;
};

/** @brief populate_kernel_cache function
 *
 *  Generate a cache of normalised kernels at oversampled-pixel offsets.
 *
 *  @param[in] T& kernel_creator : the kernel creator functor.
 *  @param[in] support (int): See kernel generation routine.
 *  @param[in] oversampling (int): Oversampling ratio value.
 *  @param[in] pad (bool) :  Whether to pad the array by an extra pixel-width.
 *             This is used when generating an oversampled kernel that will be used for interpolation. Default is false.
 *  @param[in] normalize (bool): Whether to normalize generated kernel functions. Default is true.
 *
 *  @return (arma::field<arma::mat>): Mapping oversampling-pixel offsets to normalised kernels.
 */
template <typename T>
arma::field<arma::Mat<cx_real_t>> populate_kernel_cache(const T& kernel_creator, const uint support, const uint oversampling, const bool pad = false, const bool normalize = true)
{
    uint oversampled_pixel = (oversampling / 2);
    uint cache_size = (oversampled_pixel * 2 + 1);

    arma::mat oversampled_pixel_offsets = (arma::linspace(0, cache_size - 1, cache_size) - oversampled_pixel) / oversampling;
    arma::field<arma::Mat<cx_real_t>> cache(cache_size, cache_size); // 2D kernel array cache to be returned
    arma::field<arma::Mat<real_t>> kernel1D_cache(cache_size); // Temporary cache for 1D kernel arrays

    // Fill 1D kernel array cache
    for (uint i = 0; i < cache_size; i++) {
        kernel1D_cache(i) = make_1D_kernel(kernel_creator, support, oversampled_pixel_offsets[i], 1, pad, normalize);
    }
    size_t kernel_size = kernel1D_cache(0).n_elem;

    // Generate 2D kernel array cache based on 1D kernel cache for reduced running times
    for (uint i = 0; i < cache_size; i++) {
        for (uint j = 0; j <= i; j++) {
            cache(j, i).set_size(kernel_size, kernel_size);
            // Perfom dot product
            for (size_t ii = 0; ii < kernel_size; ii++) {
                for (size_t jj = 0; jj < kernel_size; jj++) {
                    cache(j, i).at(jj, ii) = kernel1D_cache(j).at(jj) * kernel1D_cache(i).at(ii);
                }
            }
            if (i != j)
                cache(i, j) = cache(j, i).st();
        }
    }

    return cache;
}

/**
 * @brief Convert visibilities for half-plane gridding and mark the ones that need to be duplicated (includes W-lambda array).
 *
 * @param[in,out] uv_lambda (arma::mat): UV-coordinates of complex visibilities to be converted.
 *                                   2D double array with 2 columns. Assumed ordering is u,v.
 * @param[in,out] w_lambda (arma::vec): W-coordinate of complex visibilities to be converted (1D array).
 * @param[in,out] vis (arma::cx_mat): Complex visibilities to be converted (1D array).
 * @param[in] kernel_support (int): Kernel support.
 * @param[in,out] good_vis (arma::Col<uint>): Array that identifies visibilities to be duplicated.
 */
void convert_to_halfplane_visibilities(arma::mat& uv_lambda, arma::vec& w_lambda, arma::cx_mat& vis, int kernel_support, arma::Col<uint>& good_vis);

/**
 * @brief Convert visibilities for half-plane gridding and mark the ones that need to be duplicated.
 *
 * @param[in,out] uv_lambda (arma::mat): UV-coordinates of complex visibilities to be converted.
 *                                   2D double array with 2 columns. Assumed ordering is u,v.
 * @param[in,out] vis (arma::cx_mat): Complex visibilities to be converted (1D array).
 * @param[in] kernel_support (int): Kernel support.
 * @param[in,out] good_vis (arma::Col<uint>): Identifies visibilities to be duplicated.
 */
void convert_to_halfplane_visibilities(arma::mat& uv_lambda, arma::cx_mat& vis, int kernel_support, arma::Col<uint>& good_vis);

/**
 * @brief Divides the list of w-lambda values into N planes, averaging or determining the median of each function.
 *
 * @param[in] w_lambda (arma::mat): W-lambda values.
 * @param[in] good_vis (arma::Col<uint>): Indicates which visibilities are valid.
 * @param[in] num_wplanes (uint): Number of planes to divide w-lambda list.
 * @param[out] w_avg_values (arma::Col): Average W-value for each plane.
 * @param[out] w_planes_idx (arma::uvec): List of indexes corresponding to the first visibility associated to each plane.
 * @param[in] median (bool): Use median (rather than mean) to compute W-planes.
 */
void average_w_planes(arma::mat w_lambda, const arma::Col<uint>& good_vis, uint num_wplanes, arma::Col<real_t>& w_avg_values, arma::uvec& w_planes_idx, bool median = false);

/**
 * @brief Divides the list of local hour-angle values into N planes, averaging each plane.
 *
 * @param[in] lha (arma::vec): List of local hour-angle values.
 * @param[in] good_vis (arma::Col<uint>): Indicates which visibilities are valid.
 * @param[in] num_timesteps (uint): number of planes to divide lha list.
 * @param[out] lha_planes (arma::vec): Average lha-values for each plane.
 * @param[out] vis_timesteps (arma::ivec): Plane index associated to each visibility.
 */
void average_lha_planes(const arma::vec& lha, const arma::Col<uint>& good_vis, uint num_timesteps, arma::Col<real_t>& lha_planes, arma::ivec& vis_timesteps);

/** @brief bounds_check_kernel_centre_locations function
 *
 *  Vectorized bounds check.
 *
 * @param[in,out] good_vis (arma::Col<uint>): List of indices for 'good' (in-bounds) positions. Note this is a list of integer values.
 * @param[in] kernel_centre_on_grid (arma::Mat<int>): Corresponding array of nearest-pixel grid-locations,
 *            which will be the centre position of a kernel placement.
 *  @param[in] support (int): Kernel support size in regular pixels.
 *  @param[in] image_size (int): Image width in pixels
 */
void bounds_check_kernel_centre_locations(arma::Col<uint>& good_vis, const arma::Mat<int>& kernel_centre_on_grid, int image_size, int support);

/** @brief calculate_oversampled_kernel_indices function
 *
 *  Find the nearest oversampled gridpoint for given sub-pixel offset.
 *
 *  @param[in] subpixel_coord (arma::mat): Array of 'fractional' co-ords, that is the
               subpixel offsets from nearest pixel on the regular grid.
 *  @param[in] oversampling (uint). How many oversampled pixels to one regular pixel.
 *
 *  @return (arma::Mat<int>): Corresponding oversampled pixel indexes
 */
arma::Mat<int> calculate_oversampled_kernel_indices(arma::mat& subpixel_coord, uint oversampling);

/** @brief generate_a_kernel function
 *
 *  Generate A-kernel from spherical harmonics. Kernel to be combined with W-kernel and AA-kernel for A-projection
 *
 *  @param[in] a_proj (A_ProjectionPars): A-projection parameters
 *  @param[in] fov (double): Field of view
 *  @param[in] workarea_size (int): Workarea size
 *  @param[in] rot_angle (double): Rotation angle (default is 0.0)
 *
 *  @return (arma::Mat<real_t>): A-kernel matrix
 */
arma::Mat<real_t> generate_a_kernel(const A_ProjectionPars& a_proj, const double fov, const int workarea_size, double rot_angle = 0.0);

/** @brief Grid visibilities using convolutional gridding.
 *
 *  Returns the **un-normalized** weighted visibilities; the
 *  weights-renormalization factor can be calculated by summing the sample grid.
 *
 *  If 'exact == True' then exact gridding is used, i.e. the kernel is
 *  recalculated for each visibility, with precise sub-pixel offset according to
 *  that visibility's UV co-ordinates. Otherwise, instead of recalculating the
 *  kernel for each sub-pixel location, we pre-generate an oversampled kernel
 *  ahead of time - so e.g. for an oversampling of 5, the kernel is
 *  pre-generated at 0.2 pixel-width offsets. We then pick the pre-generated
 *  kernel corresponding to the sub-pixel offset nearest to that of the
 *  visibility.
 *  Kernel pre-generation results in improved performance, particularly with
 *  large numbers of visibilities and complex kernel functions, at the cost of
 *  introducing minor aliasing effects due to the 'step-like' nature of the
 *  oversampled kernel. This in turn can be minimised (at the cost of longer
 *  start-up times and larger memory usage) by pre-generating kernels with a
 *  larger oversampling ratio, to give finer interpolation.
 *  This function also performs W-projection and A-projection when these
 *  parameters are provided.
 *
 *  @param[in] T& kernel_creator : the kernel creator functor.
 *  @param[in] support (uint) : Defines the 'radius' of the bounding box within
 *              which convolution takes place. `Box width in pixels = 2*support+1`.
 *              (The central pixel is the one nearest to the UV co-ordinates.)
 *              (This is sometimes known as the 'half-support')
 *  @param[in] image_size (int) : Width of the image in pixels. NB we assume
 *              the pixel `[image_size//2,image_size//2]` corresponds to the origin
 *              in UV-space.
 *  @param[in] uv_lambda (arma::mat) : UV-coordinates of input visibilities.
 *  @param[in] vis (arma::cx_mat) : Complex visibilities. 1d array, shape: (n_vis).
 *  @param[in] vis_weights (arma::mat) : Visibility weights. 1d array, shape: (n_vis).
 *  @param[in] kernel_exact (bool) : Calculate exact kernel-values for every UV-sample.
 *  @param[in] oversampling (uint) : Controls kernel-generation if ``exact==False``.
                Larger values give a finer-sampled set of pre-cached kernels.
 *  @param[in] shift_uv (bool) : Shift uv-coordinates before gridding (required when fftshift function is
 *              skipped before fft). Default is true.
 *  @param[in] halfplane_gridding (bool) : Grid only halfplane matrix. Used when halfplane c2r fft will be applied.
 *              Default is true.
 *  @param[in] w_proj (W_ProjectionPars) : W-projection configuration parameters.
 *  @param[in] w_lambda (arma::vec) : W-coordinate of input visibilities.
 *  @param[in] cell_size (double) : Angular-width of a synthesized pixel in the image to be created (arcsecond).
 *  @param[in] analytic_gcf (bool): Compute approximation of image-domain kernel from analytic expression. Default is true.
 *  @param[in] r_fft (FFTRoutine): Selects FFT routine. Default is FFTW_ESTIMATE_FFT.
 *  @param[in] a_proj (A_ProjectionPars) : A-projection configuration parameters.
 *
 *  @return (GridderRes): stores vis_grid and sampling_grid, representing the visibility grid and the
 *                         sampling grid matrices. Includes also value with the total sampling grid sum.
 */
template <bool generateBeam = true, typename T>
GridderOutput convolve_to_grid(
    const T& kernel_creator,
    const uint support,
    int image_size,
    arma::mat uv_lambda,
    arma::cx_mat vis,
    arma::mat vis_weights,
    bool kernel_exact = true,
    uint oversampling = 1,
    bool shift_uv = true,
    bool halfplane_gridding = true,
    const W_ProjectionPars& w_proj = W_ProjectionPars(),
    arma::vec w_lambda = arma::vec(),
    double cell_size = 0.0,
    bool analytic_gcf = true,
    FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT,
    const A_ProjectionPars& a_proj = A_ProjectionPars())
{
    bool use_wproj = w_proj.isEnabled();
    bool use_aproj = a_proj.isEnabled();
    real_t obsdec_rad = real_t(deg2rad(a_proj.obs_dec));
    real_t obsra_rad = real_t(deg2rad(a_proj.obs_ra));

    // Half image size
    const int half_image_size = int(image_size / 2);
    // Set convolution kernel support for gridding
    int conv_support = int(support);

    /* Some checks ***/
    assert(uv_lambda.n_cols == 2);
    assert(uv_lambda.n_rows == vis.n_rows);
#ifdef WPROJECTION
    if (use_wproj)
        assert(w_lambda.n_rows == vis.n_rows);
    else
        w_lambda = arma::zeros<arma::vec>(vis.n_rows);
#endif
    assert(kernel_exact || (oversampling >= 1));
    assert((image_size % 2) == 0);
    assert(vis.n_elem == vis_weights.n_elem);
    if (use_aproj) {
        assert(!kernel_exact);
        assert(use_wproj);
        assert(!w_proj.hankel_opt);
    }

// Gridder options selection
#ifdef WPROJECTION
    if (use_wproj || use_aproj) {
        use_wproj = true;
        conv_support = int(w_proj.max_wpconv_support); // Set convolution kernel support for gridding
        kernel_exact = false; // kernel exact must be false in this case
    }
#endif
    assert(conv_support > 0);

    if (kernel_exact == true) {
        oversampling = 1;
    } else {
        oversampling = (oversampling >> 1) << 1; // must be multiple of 2
        if (oversampling == 0)
            oversampling = 1;
    }

    arma::Col<uint> good_vis(arma::size(vis));
    good_vis.ones(); // Init good_vis with ones
    arma::Mat<int> kernel_centre_on_grid(arma::size(uv_lambda));
    arma::mat uv_frac(arma::size(uv_lambda));
    arma::vec slha;

#ifdef WPROJECTION
    if (use_wproj) {
        /** calculate wplanes and sort w coordinate
         * by absolute value
         */
        // create auxiliary vector to keep track of original lambda data
        // Abs w
        arma::mat uv_aux(arma::size(uv_lambda));
        arma::mat w_aux(arma::size(w_lambda));
        arma::cx_mat vis_aux(arma::size(vis));
        arma::mat vis_weights_aux(arma::size(vis_weights));

        /***** Sort by the module of w coordinate **/
        arma::uvec sorted_idxs = arma::sort_index(arma::abs(w_lambda), "ascend");

        tbb::parallel_for(tbb::blocked_range<size_t>(0, sorted_idxs.n_elem), [&](const tbb::blocked_range<size_t>& r) {
            size_t e = r.end();
            for (size_t i = r.begin(); i < e; ++i) {
                uv_aux.at(i, 0) = uv_lambda.at(sorted_idxs[i], 0);
                uv_aux.at(i, 1) = uv_lambda.at(sorted_idxs[i], 1);
            }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, sorted_idxs.n_elem), [&](const tbb::blocked_range<size_t>& r) {
            size_t e = r.end();
            for (size_t i = r.begin(); i < e; ++i) {
                w_aux[i] = w_lambda[sorted_idxs[i]];
            }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, sorted_idxs.n_elem), [&](const tbb::blocked_range<size_t>& r) {
            size_t e = r.end();
            for (size_t i = r.begin(); i < e; ++i) {
                vis_aux[i] = vis[sorted_idxs[i]];
            }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, sorted_idxs.n_elem), [&](const tbb::blocked_range<size_t>& r) {
            size_t e = r.end();
            for (size_t i = r.begin(); i < e; ++i) {
                vis_weights_aux[i] = vis_weights[sorted_idxs[i]];
            }
        });

        if (use_aproj) {
            slha.set_size(arma::size(a_proj.lha));
            tbb::parallel_for(tbb::blocked_range<size_t>(0, sorted_idxs.n_elem), [&](const tbb::blocked_range<size_t>& r) {
                size_t e = r.end();
                for (size_t i = r.begin(); i < e; ++i) {
                    slha[i] = a_proj.lha[sorted_idxs[i]];
                }
            });
        }

        uv_lambda = std::move(uv_aux);
        w_lambda = std::move(w_aux);
        vis = std::move(vis_aux);
        vis_weights = std::move(vis_weights_aux);
    }
#endif

    // If a visibility point is located in the top half-plane, move it to the bottom half-plane to a symmetric position with respect to the matrix centre (0,0)
    if (halfplane_gridding) {
        if (use_wproj)
            convert_to_halfplane_visibilities(uv_lambda, w_lambda, vis, conv_support, good_vis);
        else
            convert_to_halfplane_visibilities(uv_lambda, vis, conv_support, good_vis);
    }

    // get visibilities to be processed and uv_frac
    for (size_t idx = 0; idx < kernel_centre_on_grid.n_rows; ++idx) {
        int val = int(rint(uv_lambda.at(idx, 0)));
        uv_frac(idx, 0) = uv_lambda.at(idx, 0) - val;
        kernel_centre_on_grid(idx, 0) = val + half_image_size;

        val = int(rint(uv_lambda.at(idx, 1)));
        uv_frac(idx, 1) = uv_lambda.at(idx, 1) - val;
        kernel_centre_on_grid(idx, 1) = val + half_image_size;
    }
    bounds_check_kernel_centre_locations(good_vis, kernel_centre_on_grid, image_size, conv_support);

    STPLIB_DEBUG("stplib", "Gridder: Total # of vis = {}", vis.n_elem);
    STPLIB_DEBUG("stplib", "Gridder: Total # of good vis = {}", arma::accu(good_vis != 0));

    // Number of rows of the output images. This changes if halfplane gridding is used
    int image_rows = int(image_size);
    if (halfplane_gridding) {
        image_rows = half_image_size + 1;
    }

    // Shift positions of the visibilities (to avoid call to fftshift function)
    if (shift_uv) {
        int shift_offset = half_image_size;
        kernel_centre_on_grid.each_row([&](arma::Mat<int>& r) {
            if (r[0] < shift_offset) {
                r[0] += shift_offset;
            } else {
                r[0] -= shift_offset;
            }
            if (r[1] < shift_offset) {
                r[1] += shift_offset;
            } else {
                r[1] -= shift_offset;
            }
        });
    }

    // Compute total sampling grid
    // Used to renormalize the visibilities by the sampled weights total
    double sample_grid_total = 0;
    for (arma::uword vi = 0; vi < good_vis.n_elem; vi++) {
        if (good_vis[vi] != 0)
            sample_grid_total += vis_weights[vi];
    }
    // Total sampling grid value must be doubled because we are using half gridder image
    sample_grid_total *= 2.0;

    // Create matrices for output gridded images
    // Use MatStp class because these images shall be efficiently initialized with zeros
    MatStp<cx_real_t> vis_grid((size_t(image_rows)), size_t(image_size));
    int sampl_image_size = 0;
    int sampl_image_rows = 0;
    if (generateBeam) {
        sampl_image_size = image_size;
        sampl_image_rows = image_rows;
    }

    MatStp<cx_real_t> sampling_grid((size_t(sampl_image_rows)), size_t(sampl_image_size));
    int kernel_size = conv_support * 2 + 1;

    if (kernel_exact == false) {
        // kernel caches declaration
        arma::field<arma::Mat<cx_real_t>> kernel_cache;
        // kernel fft
        arma::Col<real_t> aa_kernel_img;

#ifdef WPROJECTION
        // aux vars
        double scaling_factor = 1.0;
        int workarea_size = image_size;
        STPLIB_DEBUG("stplib", "Gridder: Maximum workarea size (before undersampling opt) = {}", workarea_size);

        // local data
        arma::Col<real_t> w_avg_values; // vector of w_planes
        arma::uvec w_planes_firstidx; // wplanes index tracking

        // Compute W-Planes and convolution kernels for W-Projection
        uint num_wplanes = 1;
        if (use_wproj) {
            num_wplanes = w_proj.num_wplanes;

            // create w plane array and idx tracker
            w_avg_values.set_size(num_wplanes);
            w_planes_firstidx.set_size(num_wplanes);

            // calculate w_planes_avg
            average_w_planes(arma::abs(w_lambda), good_vis, num_wplanes, w_avg_values, w_planes_firstidx, w_proj.wplanes_median);

            /*** Calculate kernel working area size */
            int undersampling_opt = w_proj.undersampling_opt > 0 ? int(std::pow(2, w_proj.undersampling_opt - 1)) : 0;
            if (undersampling_opt > 0) {
                workarea_size = 2;
                while ((workarea_size / undersampling_opt) < kernel_size) {
                    workarea_size *= 2;
                }
            }
            else
            {
                // Undersampling optimisation is 0, set workarea size to the power of 2 above image_size
                workarea_size = 2;
                while (workarea_size < image_size) {
                    workarea_size *= 2;
                }
            }
            // We need to use scaling factor for w-kernel computation, because the workarea size is different than the image size
            scaling_factor = double(image_size) / double(workarea_size);

            // Determine image-domain AA-kernel
            aa_kernel_img = ImgDomKernel(kernel_creator, workarea_size, false, analytic_gcf, r_fft);
        } else {
            // Set some variables when w-projection is not used
            num_wplanes = 1;
            w_planes_firstidx.set_size(1);
            w_planes_firstidx(0) = 0;
#endif
            kernel_cache = populate_kernel_cache(kernel_creator, support, oversampling, /*pad*/ false, /*normalize*/ true);
#ifdef WPROJECTION
        }

        // Create W-kernel object
        WideFieldImaging wide_imaging(uint(workarea_size), arc_sec_to_rad(cell_size), oversampling, scaling_factor, w_proj, r_fft);

        STPLIB_DEBUG("stplib", "Gridder: Workarea size = {}", workarea_size);
#endif

        // get oversampled kernel indexes
        arma::Mat<int> oversampled_offset = calculate_oversampled_kernel_indices(uv_frac, oversampling) + (oversampling / 2);

#ifdef APROJECTION
        arma::ivec vis_timesteps(arma::size(vis));
        arma::Col<real_t> lha_planes;
        arma::Mat<real_t> Akernel;
        uint num_timesteps = 1;
        if (use_aproj) {
            num_timesteps = a_proj.num_timesteps;
            average_lha_planes(slha, good_vis, num_timesteps, lha_planes, vis_timesteps);

            if (a_proj.aproj_opt) {
                double fov = arc_sec_to_rad(cell_size) * double(image_size);
                Akernel = generate_a_kernel(a_proj, fov, workarea_size);
            }
        }
#endif

        TIMESTAMP_IMAGER

        // Init conv kernel generation time
        std::chrono::duration<double> convkernelgentimes(0);

#ifdef SERIAL_GRIDDER
// Single-threaded implementation of oversampled gridder
#ifdef WPROJECTION
        for (int pi = 0; pi < num_wplanes; pi++) {
            arma::field<arma::Mat<cx_real_t>> kernel_cache_conj;
            arma::uword vi_begin = w_planes_firstidx(pi);
            arma::uword vi_end = (pi == (num_wplanes - 1)) ? good_vis.n_elem : w_planes_firstidx(pi + 1);

            if (use_wproj) {
                // Start timestamp
                auto start = std::chrono::high_resolution_clock::now();
#ifdef APROJECTION
                if (use_aproj) {
                    // Generate new AA/W-kernel for A-projection
                    wide_imaging.generate_image_domain_convolution_kernel(w_avg_values(pi), aa_kernel_img);
                } else
#endif
                {
                    // Generate new convolution kernel for W-projection
                    wide_imaging.generate_convolution_kernel_wproj(w_avg_values(pi), aa_kernel_img, w_proj.hankel_proj_slice);
                }
                // End timestamp
                auto end = std::chrono::high_resolution_clock::now();
                convkernelgentimes += (end-start);
            }
#else
        arma::uword vi_begin = 0;
        arma::uword vi_end = good_vis.n_elem;
#endif
#ifdef APROJECTION
            for (int ts = 0; ts < num_timesteps; ts++) {
                if (use_aproj) {
                    // Start timestamp
                    auto start = std::chrono::high_resolution_clock::now();

                    // Generate the AW - kernels
                    real_t pangle = parangle(lha_planes.at(ts), obsdec_rad, obsra_rad);

                    STPLIB_DEBUG("stplib", "Generate conv. kernel with parallatic angle: {} lha: {}", pangle, lha_planes.at(ts));

                    double fov = arc_sec_to_rad(cell_size) * double(image_size);
                    Akernel = generate_a_kernel(a_proj, fov, workarea_size, pangle);
                    // Generate new convolution kernel for A-projection
                    wide_imaging.generate_convolution_kernel_aproj(Akernel);

                    // End timestamp
                    auto end = std::chrono::high_resolution_clock::now();
                    convkernelgentimes += (end-start);
                }
#endif
#ifdef WPROJECTION
                if (use_wproj) {
                    // Start timestamp
                    auto start = std::chrono::high_resolution_clock::now();

                    kernel_cache = wide_imaging.generate_kernel_cache();
                    conv_support = int(wide_imaging.get_trunc_conv_support());
                    assert(conv_support > 0);
                    kernel_size = conv_support * 2 + 1;
                    STPLIB_DEBUG("stplib", "Gridder: W-plane {} = {}, Plane size = {}, Conv kernel support = {}, Conv kernel size = {}", pi, w_avg_values(pi), vi_end - vi_begin, conv_support, kernel_size);

                    kernel_cache_conj.reset();
                    kernel_cache_conj.set_size(arma::size(kernel_cache));

                    // End timestamp
                    auto end = std::chrono::high_resolution_clock::now();
                    convkernelgentimes += (end-start);
                }
#endif
                for (arma::uword vi = vi_begin; vi < vi_end; vi++) {
                    const uint good_vis_val = good_vis[vi];

#ifdef APROJECTION
                    // Skip this visibility if using A-projection and it does not belong to the time interval
                    if (use_aproj)
                        if (vis_timesteps[vi] != ts)
                            continue;
#endif
                    if (good_vis_val == 0)
                        continue;

                    int gc_x = kernel_centre_on_grid(vi, 0);
                    int gc_y = kernel_centre_on_grid(vi, 1);
                    int cp_x = oversampled_offset.at(vi, 0);
                    int cp_y = oversampled_offset.at(vi, 1);
                    cx_real_t vis_val = cx_real_t(vis[vi]);
                    const real_t vis_weight = vis_weights[vi];
#ifdef WPROJECTION
                    double w_lambda_val = w_lambda.at(vi);
#endif

                    // If good_vis[vi] is 2, add also conjugate visibility
                    for (uint gv = 0; gv < good_vis_val; gv++) {

                        if (gv) {
                            gc_x = -kernel_centre_on_grid(vi, 0) + image_size;
                            gc_y = -kernel_centre_on_grid(vi, 1) + image_size;
                            if (oversampling > 1) {
                                cp_x = -oversampled_offset.at(vi, 0) + oversampling;
                                cp_y = -oversampled_offset.at(vi, 1) + oversampling;
                            }
                            vis_val = std::conj(vis_val);
#ifdef WPROJECTION
                            w_lambda_val *= (-1);
#endif
                        }

                        arma::Mat<cx_real_t>* conv_kernel_array = &(kernel_cache(cp_y, cp_x));
#ifdef WPROJECTION
                        if (use_wproj) {
                            if (w_lambda_val < 0.0) {
                                if (kernel_cache_conj(cp_y, cp_x).is_empty()) {
                                    kernel_cache_conj(cp_y, cp_x) = std::move(arma::conj(kernel_cache(cp_y, cp_x)));
                                }
                                conv_kernel_array = &(kernel_cache_conj(cp_y, cp_x));
                            }
                        }
#endif
                        for (int j = 0; j < kernel_size; j++) {
                            int grid_col = gc_x - conv_support + j;
                            if (grid_col < 0) // left/right split of the kernel
                                grid_col += image_size;
                            if (grid_col >= image_size) // left/right split of the kernel
                                grid_col -= image_size;

                            // Use pointers here for faster access
                            cx_real_t* conv_kernel_col = conv_kernel_array->colptr(j);
                            cx_real_t* vis_grid_col = vis_grid.colptr(grid_col);
                            cx_real_t* sampling_grid_col = NULL;
                            if (generateBeam) {
                                sampling_grid_col = sampling_grid.colptr(grid_col);
                            }

                            int grid_row = gc_y - conv_support;
                            for (int i = 0; i < kernel_size; ++i, ++grid_row) {
                                if (grid_row < 0) // top/bottom split of the kernel. Halfplane gridding: kernel points in the negative halfplane are excluded
                                    continue;
                                if (grid_row >= image_size) // top/bottom split of the kernel. Halfplane gridding: kernel points touching the positive halfplane are used
                                    grid_row -= image_size;
                                // The following condition is needed for the case of halfplane gridding, because only the top halfplane visibilities are convolved
                                if (grid_row < image_rows) {
                                    cx_real_t kernel_val = conv_kernel_col[i] * vis_weight;
                                    assert(arma::is_finite(kernel_val));
                                    vis_grid_col[grid_row] += vis_val * kernel_val;
                                    if (generateBeam) {
                                        sampling_grid_col[grid_row] += kernel_val;
                                    }
                                }
                            }
                        }
                    }
                }
#ifdef APROJECTION
            }
#endif
#ifdef WPROJECTION
        }
        times_gridder.push_back(convkernelgentimes);
#endif
#else
        // Multi-threaded implementation of oversampled gridder (20-25% faster)
#ifdef WPROJECTION
        for (uint pi = 0; pi < num_wplanes; pi++) {
            arma::uword vi_begin = w_planes_firstidx(pi);
            arma::uword vi_end = (pi == (num_wplanes - 1)) ? good_vis.n_elem : w_planes_firstidx(pi + 1);

            if (use_wproj) {
                // Start timestamp
                auto start = std::chrono::high_resolution_clock::now();
#ifdef APROJECTION
                if (use_aproj) {
                    // Generate new AA/W-kernel for A-projection
                    if (a_proj.aproj_opt) {
                        wide_imaging.generate_image_domain_convolution_kernel(w_avg_values(pi), aa_kernel_img);
                        // Generate convolution kernel for A-projection (it will be rotated afterwards)
                        wide_imaging.generate_convolution_kernel_aproj(Akernel);
                    } else {
                        wide_imaging.generate_image_domain_convolution_kernel(w_avg_values(pi), aa_kernel_img);
                    }
                } else
#endif
                {
                    // Generate new convolution kernel for W-projection
                    wide_imaging.generate_convolution_kernel_wproj(w_avg_values(pi), aa_kernel_img, w_proj.hankel_proj_slice);
                }

                // End timestamp
                auto end = std::chrono::high_resolution_clock::now();
                convkernelgentimes += (end-start);
            }
#else
        arma::uword vi_begin = 0;
        arma::uword vi_end = good_vis.n_elem;
#endif
#ifdef APROJECTION
            for (uint ts = 0; ts < num_timesteps; ts++) {
                if (use_aproj) {
                    // Start timestamp
                    auto start = std::chrono::high_resolution_clock::now();

                    // Generate the AW - kernels
                    real_t pangle = parangle(lha_planes.at(ts), obsdec_rad, obsra_rad);

                    STPLIB_DEBUG("stplib", "Generate conv. kernel with parallatic angle: {} lha: {}", pangle, lha_planes.at(ts));

                    if (a_proj.aproj_opt) {
                        // TODO: remove shifts and review rotate_matrix function
                        fftshift(wide_imaging.conv_kernel);
                        cx_real_t pbmin = wide_imaging.conv_kernel(0, 0);
                        wide_imaging.conv_kernel = rotate_matrix(wide_imaging.conv_kernel, pangle, pbmin, wide_imaging.conv_kernel.n_cols);
                        fftshift(wide_imaging.conv_kernel);
                    } else {
                        double fov = arc_sec_to_rad(cell_size) * double(image_size);
                        Akernel = generate_a_kernel(a_proj, fov, workarea_size, pangle);
                        // Generate new convolution kernel for A-projection
                        wide_imaging.generate_convolution_kernel_aproj(Akernel);
                    }

                    // End timestamp
                    auto end = std::chrono::high_resolution_clock::now();
                    convkernelgentimes += (end-start);
                }
#endif
#ifdef WPROJECTION
                if (use_wproj) {
                    // Start timestamp
                    auto start = std::chrono::high_resolution_clock::now();

                    kernel_cache = wide_imaging.generate_kernel_cache();
                    conv_support = int(wide_imaging.get_trunc_conv_support());
                    assert(conv_support > 0);
                    kernel_size = conv_support * 2 + 1;
                    STPLIB_DEBUG("stplib", "Gridder: W-plane {} = {}, Plane size = {}, Conv kernel support = {}, Conv kernel size = {}", pi, w_avg_values(pi), vi_end - vi_begin, conv_support, kernel_size);

                    // End timestamp
                    auto end = std::chrono::high_resolution_clock::now();
                    convkernelgentimes += (end-start);
                }
#endif

                tbb::parallel_for(tbb::blocked_range<int>(0, kernel_size, 1), [&](const tbb::blocked_range<int>& r) {
                    for (arma::uword vi = vi_begin; vi < vi_end; vi++) {
                        const uint good_vis_val = good_vis[vi];

#ifdef APROJECTION
                        // Skip this visibility if using A-projection and it does not belong to the time interval
                        if (use_aproj)
                            if (vis_timesteps[vi] != ts)
                                continue;
#endif
                        if (good_vis_val == 0) {
                            continue;
                        }

                        int gc_x = kernel_centre_on_grid(vi, 0);
                        int gc_y = kernel_centre_on_grid(vi, 1);
                        int cp_x = oversampled_offset.at(vi, 0);
                        int cp_y = oversampled_offset.at(vi, 1);
                        cx_real_t vis_val = cx_real_t(vis[vi]);
                        const real_t vis_weight = real_t(vis_weights[vi]);
#ifdef WPROJECTION
                        double w_lambda_val = w_lambda.at(vi);
#endif

                        // If good_vis[vi] is 2, add also conjugate visibility
                        for (uint gv = 0; gv < good_vis_val; gv++) {

                            if (gv) {
                                gc_x = -kernel_centre_on_grid(vi, 0) + image_size;
                                gc_y = -kernel_centre_on_grid(vi, 1) + image_size;
                                if (oversampling > 1) {
                                    cp_x = -oversampled_offset.at(vi, 0) + int(oversampling);
                                    cp_y = -oversampled_offset.at(vi, 1) + int(oversampling);
                                }
                                vis_val = std::conj(vis_val);
#ifdef WPROJECTION
                                w_lambda_val *= (-1);
#endif
                            }

                            int rbegin = r.begin();
                            int conv_col = ((gc_x - rbegin + kernel_size) % kernel_size);
                            if (conv_col > conv_support)
                                conv_col -= kernel_size;
                            int grid_col = (gc_x - conv_col);

                            if ((grid_col < image_size) && (grid_col >= 0)) {

                                assert(std::abs(conv_col) <= conv_support);
                                assert(arma::uword(grid_col % kernel_size) == rbegin);

                                // Use pointer image_rows here for faster access
                                cx_real_t* conv_kernel_array = kernel_cache(size_t(cp_y), size_t(cp_x)).colptr(uint(conv_support - conv_col));
                                cx_real_t* vis_grid_col = vis_grid.colptr(uint(grid_col));
                                cx_real_t* sampling_grid_col = nullptr;
                                if (generateBeam) {
                                    sampling_grid_col = sampling_grid.colptr(uint(grid_col));
                                }

#ifdef WPROJECTION
                                if (use_wproj && (w_lambda_val < 0.0)) {
                                    // Pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
                                    int grid_row = gc_y - conv_support;
                                    for (int i = 0; i < kernel_size; ++i, ++grid_row) {
                                        if (grid_row < 0)
                                            continue;
                                        if (grid_row >= image_size)
                                            grid_row -= image_size;
                                        if (grid_row < image_rows) {
                                            const cx_real_t kernel_val = std::conj(conv_kernel_array[i]) * vis_weight;
                                            vis_grid_col[grid_row] += vis_val * kernel_val;
                                            if (generateBeam) {
                                                sampling_grid_col[grid_row] += kernel_val;
                                            }
                                        }
                                    }
                                } else
#endif
                                {
                                    // Pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
                                    int grid_row = gc_y - conv_support;
                                    for (int i = 0; i < kernel_size; ++i, ++grid_row) {
                                        if (grid_row < 0)
                                            continue;
                                        if (grid_row >= image_size)
                                            grid_row -= image_size;
                                        if (grid_row < image_rows) {
                                            const cx_real_t kernel_val = conv_kernel_array[i] * vis_weight;
                                            vis_grid_col[grid_row] += vis_val * kernel_val;
                                            if (generateBeam) {
                                                sampling_grid_col[grid_row] += kernel_val;
                                            }
                                        }
                                    }
                                }
                            }

                            // This is for the cases when the kernel is split between the left and right margins
                            if ((gc_x >= (image_size - conv_support)) || (gc_x < conv_support)) {
                                if (gc_x >= (image_size - conv_support)) {
                                    gc_x -= image_size;
                                    conv_col = ((gc_x - rbegin + kernel_size * 2) % kernel_size);
                                    if (conv_col > conv_support)
                                        conv_col -= kernel_size;
                                    grid_col = (gc_x - conv_col);
                                    assert(std::abs(conv_col) <= conv_support);
                                } else {
                                    assert(gc_x < conv_support);
                                    gc_x += image_size;
                                    conv_col = ((gc_x - rbegin + kernel_size) % kernel_size);
                                    if (conv_col > conv_support)
                                        conv_col -= kernel_size;
                                    grid_col = (gc_x - conv_col);
                                    assert(std::abs(conv_col) <= conv_support);
                                }

                                if ((grid_col < image_size) && (grid_col >= 0)) {

                                    // Use pointers here for faster access
                                    cx_real_t* conv_kernel_array = kernel_cache(size_t(cp_y), size_t(cp_x)).colptr(uint(conv_support - conv_col));
                                    cx_real_t* vis_grid_col = vis_grid.colptr(uint(grid_col));
                                    cx_real_t* sampling_grid_col = nullptr;
                                    if (generateBeam) {
                                        sampling_grid_col = sampling_grid.colptr(uint(grid_col));
                                    }

#ifdef WPROJECTION
                                    if (use_wproj && (w_lambda_val < 0.0)) {
                                        // Pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
                                        int grid_row = gc_y - conv_support;
                                        for (int i = 0; i < kernel_size; ++i, ++grid_row) {
                                            if (grid_row < 0)
                                                continue;
                                            if (grid_row >= image_size)
                                                grid_row -= image_size;
                                            if (grid_row < image_rows) {
                                                const cx_real_t kernel_val = std::conj(conv_kernel_array[i]) * vis_weight;
                                                vis_grid_col[grid_row] += vis_val * kernel_val;
                                                if (generateBeam) {
                                                    sampling_grid_col[grid_row] += kernel_val;
                                                }
                                            }
                                        }
                                    } else
#endif
                                    {
                                        // Pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
                                        int grid_row = gc_y - conv_support;
                                        for (int i = 0; i < kernel_size; ++i, ++grid_row) {
                                            if (grid_row < 0)
                                                continue;
                                            if (grid_row >= image_size)
                                                grid_row -= image_size;
                                            if (grid_row < image_rows) {
                                                const cx_real_t kernel_val = conv_kernel_array[i] * vis_weight;
                                                vis_grid_col[grid_row] += vis_val * kernel_val;
                                                if (generateBeam) {
                                                    sampling_grid_col[grid_row] += kernel_val;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                    tbb::simple_partitioner());
#ifdef APROJECTION
            }
#endif
#ifdef WPROJECTION
        }
        times_gridder.push_back(convkernelgentimes);
#endif
#endif
    } else {
        // Exact gridder (slower but with more accuracy)
        for (arma::uword vi = 0; vi < good_vis.n_elem; vi++) {
            if (good_vis[vi] == 0) {
                continue;
            }
            int gc_x = kernel_centre_on_grid(vi, 0);
            int gc_y = kernel_centre_on_grid(vi, 1);
            arma::mat frac = uv_frac.row(vi);
            cx_real_t vis_val = cx_real_t(vis[vi]);
            const real_t vis_weight = real_t(vis_weights[vi]);

            // If good_vis[vi] is 2, add also conjugate visibility
            for (arma::uword gv = 0; gv < good_vis[vi]; gv++) {

                if (gv == 1) {
                    gc_x = -kernel_centre_on_grid(vi, 0) + image_size;
                    gc_y = -kernel_centre_on_grid(vi, 1) + image_size;
                    frac *= (-1);
                    vis_val = std::conj(vis_val);
                }

                // Exact gridding is used, i.e. the kernel is recalculated for each visibility, with
                // precise sub-pixel offset according to that visibility's UV co-ordinates.
                arma::Mat<real_t> normed_kernel_array = make_kernel_array(kernel_creator, conv_support, frac);

                for (int j = 0; j < kernel_size; j++) {
                    int grid_col = gc_x - conv_support + j;
                    if (grid_col < 0) // left/right split of the kernel
                        grid_col += image_size;
                    if (grid_col >= image_size) // left/right split of the kernel
                        grid_col -= image_size;

                    for (int i = 0; i < kernel_size; i++) {
                        int grid_row = gc_y - conv_support + i;
                        if (grid_row < 0) // top/bottom split of the kernel. Halfplane gridding: kernel points in the negative halfplane are excluded
                            grid_row += image_size;
                        if (grid_row >= image_size) // top/bottom split of the kernel. Halfplane gridding: kernel points touching the positive halfplane are used
                            grid_row -= image_size;
                        // The following condition is needed for the case of halfplane gridding, because only the top halfplane visibilities are convolved
                        if (grid_row < image_rows) {
                            const cx_real_t kernel_val = vis_weight * normed_kernel_array.at(uint(i), uint(j));
                            vis_grid(uint(grid_row), uint(grid_col)) += (vis_val * kernel_val);
                            if (generateBeam) {
                                sampling_grid(uint(grid_row), uint(grid_col)) += kernel_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return GridderOutput(vis_grid, sampling_grid, sample_grid_total);
}
}

#endif /* GRIDDER_H */
