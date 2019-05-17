/**
* @file stp_python.cpp
* @brief Implementation of stp python wrapper functions
*/

#include "stp_python.h"

#include <pybind11/stl.h>
#include <utility>

namespace stp_python {

// Convenient aliases
using ptr_complex_double = std::complex<double>*;
using ptr_double = double*;

pybind11::tuple image_visibilities_wrapper(
    np_complex_double_array vis,
    np_double_array snr_weights,
    np_double_array uvw_lambda,
    uint image_size,
    double cell_size,
    double padding_factor,
    stp::KernelFunction kernel_func,
    uint kernel_support,
    bool kernel_exact,
    uint oversampling,
    bool generate_beam,
    bool gridding_correction,
    bool analytic_gcf,
    stp::FFTRoutine r_fft,
    std::string fft_wisdom_filename,
    uint num_wplanes,
    bool wplanes_median,
    uint max_wpconv_support,
    bool hankel_opt,
    bool hankel_proj_slice,
    uint undersampling_opt,
    double kernel_trunc_perc,
    stp::InterpType interp_type,
    uint aproj_numtimesteps,
    double obs_dec,
    double obs_ra,
    bool aproj_opt,
    double aproj_mask_perc,
    np_double_array lha,
    np_double_array pbeam_coefs)
{
    assert(vis.request().ndim == 1); // vis is a 1D array
    assert(uvw_lambda.request().ndim == 2); // uv_pixels is a 2D array

    // Set imager parameters
    stp::ImagerPars img_pars(image_size, cell_size, padding_factor, kernel_func, kernel_support, kernel_exact, oversampling,
        generate_beam, gridding_correction, analytic_gcf, r_fft, fft_wisdom_filename);

    // Set W-Projection parameters
    stp::W_ProjectionPars wproj_pars(num_wplanes, max_wpconv_support, undersampling_opt, kernel_trunc_perc,
        hankel_opt, hankel_proj_slice, interp_type, wplanes_median);

    // Image sizes must be multiple of 4
    while ((img_pars.image_size % 4) != 0) {
        img_pars.image_size++;
    }
    img_pars.padded_image_size = static_cast<double>(img_pars.image_size) * img_pars.padding_factor;
    while ((img_pars.padded_image_size % 4) != 0) {
        img_pars.padded_image_size++;
    }

    arma::cx_mat vis_arma(
        static_cast<ptr_complex_double>(vis.request().ptr),
        vis.request().shape[0], // n_rows
        1, // vis is a vector (ndim = 1), so there is no shape[1]
        false, // copy_aux_mem - do not copy memory for better performance (it will not be modified)
        true); // strict

    arma::mat snr_weights_arma(
        static_cast<ptr_double>(snr_weights.request().ptr),
        uvw_lambda.request().shape[0], // n_rows
        1, // n_cols
        false,
        true);

    arma::mat uvw_lambda_arma(
        static_cast<ptr_double>(uvw_lambda.request().ptr),
        uvw_lambda.request().shape[0], // n_rows
        uvw_lambda.request().shape[1], // n_cols
        false,
        true);

    arma::mat lha_arma(
        static_cast<ptr_double>(lha.request().ptr),
        lha.request().shape[0], // n_rows
        1, // n_cols
        false,
        true);

    std::vector<double> pbeam_coefs_vec(static_cast<ptr_double>(pbeam_coefs.request().ptr),
        static_cast<ptr_double>(pbeam_coefs.request().ptr) + size_t(pbeam_coefs.request().shape[0]));

    // Set A-Projection parameters
    stp::A_ProjectionPars aproj_pars(aproj_numtimesteps, obs_dec, obs_ra, aproj_opt, aproj_mask_perc, lha_arma, pbeam_coefs_vec);

    // Run imager
    stp::ImageVisibilities imager(std::move(vis_arma), std::move(snr_weights_arma),
        std::move(uvw_lambda_arma), img_pars, wproj_pars, aproj_pars);

    // The function will output a tuple with two np_complex_array
    pybind11::tuple result(2);

    // Image at index 0
    arma::Mat<real_t>& image = imager.vis_grid;
    ssize_t data_size = sizeof(real_t);
    pybind11::buffer_info image_buffer(
        static_cast<void*>(image.memptr()), // void *ptr
        data_size, // size_t itemsize
        pybind11::format_descriptor<real_t>::format(), // const std::string &format
        2, // size_t ndim
        { ssize_t(image.n_rows), ssize_t(image.n_cols) }, // const std::vector<ssize_t> &shape
        { data_size, ssize_t(image.n_cols) * data_size }); // const std::vector<ssize_t> &strides

    // The memory is copied when initializing the np_real_array
    result[0] = np_real_array(image_buffer);
    // Delete unnecessary matrix
    imager.vis_grid.reset();

    // Beam at index 1
    arma::Mat<real_t>& beam = imager.sampling_grid;
    pybind11::buffer_info beam_buffer(
        static_cast<void*>(beam.memptr()), // void *ptr
        data_size, // size_t itemsize
        pybind11::format_descriptor<real_t>::format(), // const std::string &format
        2, // size_t ndim
        { ssize_t(beam.n_rows), ssize_t(beam.n_cols) }, // const std::vector<ssize_t> &shape
        { data_size, ssize_t(beam.n_cols) * data_size }); // const std::vector<ssize_t> &strides

    // The memory is copied when initializing the np_real_array
    result[1] = np_real_array(beam_buffer);

    return result;
}

std::vector<std::tuple<int, double, int, int, int, stp::Gaussian2dParams, stp::Gaussian2dParams, std::string>> source_find_wrapper(
    np_real_array image_data,
    double detection_n_sigma,
    double analysis_n_sigma,
    double rms_est,
    bool find_negative_sources,
    uint sigma_clip_iters,
    stp::MedianMethod median_method,
    bool gaussian_fitting,
    bool ccl_4connectivity,
    bool generate_labelmap,
    int source_min_area,
    stp::CeresDiffMethod ceres_diffmethod,
    stp::CeresSolverType ceres_solvertype)
{
    assert(image_data.request().ndim == 2);

    arma::Mat<real_t> image_data_arma(
        static_cast<real_t*>(image_data.request().ptr),
        image_data.request().shape[0], // n_rows
        image_data.request().shape[1], // n_cols
        false, // copy_aux_mem - do not copy memory for better performance (it will not be modified)
        true); // strict

    // Call source find function
    stp::SourceFindImage sfimage = stp::SourceFindImage(std::move(image_data_arma), detection_n_sigma, analysis_n_sigma, rms_est,
        find_negative_sources, sigma_clip_iters, median_method, gaussian_fitting, ccl_4connectivity, generate_labelmap,
        source_min_area, ceres_diffmethod, ceres_solvertype);

    // Convert 'vector of stp::island' to 'vector of tuples'
    std::vector<std::tuple<int, double, int, int, int, stp::Gaussian2dParams, stp::Gaussian2dParams, std::string>> v_islands;
    v_islands.reserve(sfimage.islands.size());
    for (auto&& i : sfimage.islands) {
        v_islands.push_back(std::move(std::make_tuple(i.sign, i.extremum_val, i.extremum_x_idx, i.extremum_y_idx, i.num_samples,
            i.moments_fit, i.leastsq_fit, i.ceres_report)));
    }
    return v_islands;
}

PYBIND11_MODULE(stp_python, m)
{
    m.doc() = "The Slow Transients Pipeline";

    // Enum bindings
    pybind11::enum_<stp::KernelFunction>(m, "KernelFunction")
        .value("TopHat", stp::KernelFunction::TopHat)
        .value("Triangle", stp::KernelFunction::Triangle)
        .value("Sinc", stp::KernelFunction::Sinc)
        .value("Gaussian", stp::KernelFunction::Gaussian)
        .value("GaussianSinc", stp::KernelFunction::GaussianSinc)
        .value("PSWF", stp::KernelFunction::PSWF);

    pybind11::enum_<stp::FFTRoutine>(m, "FFTRoutine")
        .value("FFTW_ESTIMATE_FFT", stp::FFTRoutine::FFTW_ESTIMATE_FFT)
        .value("FFTW_MEASURE_FFT", stp::FFTRoutine::FFTW_MEASURE_FFT)
        .value("FFTW_PATIENT_FFT", stp::FFTRoutine::FFTW_PATIENT_FFT)
        .value("FFTW_WISDOM_FFT", stp::FFTRoutine::FFTW_WISDOM_FFT)
        .value("FFTW_WISDOM_INPLACE_FFT", stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT);

    pybind11::enum_<stp::InterpType>(m, "InterpType")
        .value("LINEAR", stp::InterpType::LINEAR)
        .value("COSINE", stp::InterpType::COSINE)
        .value("CUBIC", stp::InterpType::CUBIC); //.value("CUBIC_SMOOTH", stp::InterpType::CUBIC_SMOOTH);

    pybind11::enum_<stp::MedianMethod>(m, "MedianMethod")
        .value("ZEROMEDIAN", stp::MedianMethod::ZEROMEDIAN)
        .value("BINMEDIAN", stp::MedianMethod::BINMEDIAN)
        .value("BINAPPROX", stp::MedianMethod::BINAPPROX)
        .value("NTHELEMENT", stp::MedianMethod::NTHELEMENT);

    pybind11::enum_<stp::CeresDiffMethod>(m, "CeresDiffMethod")
        .value("AutoDiff", stp::CeresDiffMethod::AutoDiff)
        .value("AutoDiff_SingleResBlk", stp::CeresDiffMethod::AutoDiff_SingleResBlk)
        .value("AnalyticDiff", stp::CeresDiffMethod::AnalyticDiff)
        .value("AnalyticDiff_SingleResBlk", stp::CeresDiffMethod::AnalyticDiff_SingleResBlk);

    pybind11::enum_<stp::CeresSolverType>(m, "CeresSolverType")
        .value("LinearSearch_BFGS", stp::CeresSolverType::LinearSearch_BFGS)
        .value("LinearSearch_LBFGS", stp::CeresSolverType::LinearSearch_LBFGS)
        .value("TrustRegion_DenseQR", stp::CeresSolverType::TrustRegion_DenseQR);

    // Gaussian2dParams struct binding
    pybind11::class_<stp::Gaussian2dParams>(m, "Gaussian2dParams")
        .def_readwrite("amplitude", &stp::Gaussian2dParams::amplitude)
        .def_readwrite("x_centre", &stp::Gaussian2dParams::x_centre)
        .def_readwrite("y_centre", &stp::Gaussian2dParams::y_centre)
        .def_readwrite("semimajor", &stp::Gaussian2dParams::semimajor)
        .def_readwrite("semiminor", &stp::Gaussian2dParams::semiminor)
        .def_readwrite("theta", &stp::Gaussian2dParams::theta)
        .def("__repr__", [](const stp::Gaussian2dParams& g) {
            return "Gaussian2dParams=[" + std::to_string(g.amplitude) + ", " + std::to_string(g.x_centre)
                + ", " + std::to_string(g.y_centre) + ", " + std::to_string(g.semimajor) + ", " + std::to_string(g.semiminor)
                + ", " + std::to_string(g.theta) + "]";
        });

    // Function bindings
    m.def("image_visibilities_wrapper", &image_visibilities_wrapper, "Compute image visibilities (gridding + ifft).",
        pybind11::arg("vis"),
        pybind11::arg("snr_weights"),
        pybind11::arg("uvw_lambda"),
        pybind11::arg("image_size"),
        pybind11::arg("cell_size"),
        pybind11::arg("padding_factor") = 1.0,
        pybind11::arg("kernel_func") = stp::KernelFunction::PSWF,
        pybind11::arg("kernel_support") = 3,
        pybind11::arg("kernel_exact") = true,
        pybind11::arg("kernel_oversampling") = 8,
        pybind11::arg("generate_beam") = false,
        pybind11::arg("gridding_correction") = true,
        pybind11::arg("analytic_gcf") = false,
        pybind11::arg("r_fft") = stp::FFTRoutine::FFTW_ESTIMATE_FFT,
        pybind11::arg("fft_wisdom_filename") = std::string(),
        pybind11::arg("num_wplanes") = 0,
        pybind11::arg("wplanes_median") = false,
        pybind11::arg("max_wpconv_support") = 0,
        pybind11::arg("hankel_opt") = false,
        pybind11::arg("hankel_proj_slice") = false,
        pybind11::arg("undersampling_opt") = 1,
        pybind11::arg("kernel_trunc_perc") = 0.0,
        pybind11::arg("interp_type") = stp::InterpType::LINEAR,
        pybind11::arg("aproj_numtimesteps") = 0,
        pybind11::arg("obs_dec") = 0.0,
        pybind11::arg("obs_ra") = 0.0,
        pybind11::arg("aproj_opt") = false,
        pybind11::arg("aproj_mask_perc") = 0.0,
        pybind11::arg("lha") = np_double_array(),
        pybind11::arg("pbeam_coefs") = np_double_array());

    m.def("source_find_wrapper", &source_find_wrapper, "Find connected regions which peak above/below a given threshold.",
        pybind11::arg("image_data"),
        pybind11::arg("detection_n_sigma"),
        pybind11::arg("analysis_n_sigma"),
        pybind11::arg("rms_est") = 0.0,
        pybind11::arg("find_negative_sources") = true,
        pybind11::arg("sigma_clip_iters") = 5,
        pybind11::arg("median_method") = stp::MedianMethod::BINAPPROX,
        pybind11::arg("gaussian_fitting") = false,
        pybind11::arg("ccl_4connectivity") = false,
        pybind11::arg("generate_labelmap") = true,
        pybind11::arg("source_min_area") = 5,
        pybind11::arg("ceres_diffmethod") = stp::CeresDiffMethod::AutoDiff_SingleResBlk,
        pybind11::arg("ceres_solvertype") = stp::CeresSolverType::LinearSearch_BFGS);
}
}
