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
    int image_size,
    double cell_size,
    stp::KernelFunction kernel_func,
    double kernel_trunc_radius,
    int kernel_support,
    bool kernel_exact,
    int kernel_oversampling,
    bool generate_beam,
    stp::FFTRoutine r_fft,
    std::string fft_wisdom_filename)
{
    assert(vis.request().ndim == 1); // vis is a 1D array
    assert(uvw_lambda.request().ndim == 2); // uv_pixels is a 2D array

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

    // (image, beam) tuple
    std::pair<arma::Mat<real_t>, arma::Mat<real_t>> image_and_beam;

    // Since "image_visibilities" is a function template, there's no way to prevent the use of the following switch statement.
    // Inheritance could be used instead, but virtual function calls would impose an unnecessary performance penalty.
    switch (kernel_func) {
    case stp::KernelFunction::TopHat: {
        stp::TopHat kernel(kernel_trunc_radius);
        image_and_beam = stp::image_visibilities(kernel, std::move(vis_arma), std::move(snr_weights_arma),
            std::move(uvw_lambda_arma), image_size, cell_size, kernel_support, kernel_exact, kernel_oversampling,
            generate_beam, r_fft, fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Triangle: {
        stp::Triangle kernel(kernel_trunc_radius);
        image_and_beam = stp::image_visibilities(kernel, std::move(vis_arma), std::move(snr_weights_arma),
            std::move(uvw_lambda_arma), image_size, cell_size, kernel_support, kernel_exact, kernel_oversampling,
            generate_beam, r_fft, fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Sinc: {
        stp::Sinc kernel(kernel_trunc_radius);
        image_and_beam = stp::image_visibilities(kernel, std::move(vis_arma), std::move(snr_weights_arma),
            std::move(uvw_lambda_arma), image_size, cell_size, kernel_support, kernel_exact, kernel_oversampling,
            generate_beam, r_fft, fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Gaussian: {
        stp::Gaussian kernel(kernel_trunc_radius);
        image_and_beam = stp::image_visibilities(kernel, std::move(vis_arma), std::move(snr_weights_arma),
            std::move(uvw_lambda_arma), image_size, cell_size, kernel_support, kernel_exact, kernel_oversampling,
            generate_beam, r_fft, fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::GaussianSinc: {
        stp::GaussianSinc kernel(kernel_trunc_radius);
        image_and_beam = stp::image_visibilities(kernel, std::move(vis_arma), std::move(snr_weights_arma),
            std::move(uvw_lambda_arma), image_size, cell_size, kernel_support, kernel_exact, kernel_oversampling,
            generate_beam, r_fft, fft_wisdom_filename);
    };
        break;
    default:
        assert(0);
        break;
    }

    // The function will output a tuple with two np_complex_array
    pybind11::tuple result(2);

    // Image at index 0
    arma::Mat<real_t>& image = image_and_beam.first;
    size_t data_size = sizeof(real_t);
    pybind11::buffer_info image_buffer(
        static_cast<void*>(image.memptr()), // void *ptr
        data_size, // size_t itemsize
        pybind11::format_descriptor<real_t>::format(), // const std::string &format
        2, // size_t ndim
        { image.n_rows, image.n_cols }, // const std::vector<size_t> &shape
        { data_size, image.n_cols * data_size }); // const std::vector<size_t> &strides

    // The memory is copied when initializing the np_real_array
    result[0] = np_real_array(image_buffer);
    // Delete unnecessary matrix
    image_and_beam.first.reset();

    // Beam at index 1
    arma::Mat<real_t>& beam = image_and_beam.second;
    pybind11::buffer_info beam_buffer(
        static_cast<void*>(beam.memptr()), // void *ptr
        data_size, // size_t itemsize
        pybind11::format_descriptor<real_t>::format(), // const std::string &format
        2, // size_t ndim
        { beam.n_rows, beam.n_cols }, // const std::vector<size_t> &shape
        { data_size, beam.n_cols * data_size }); // const std::vector<size_t> &strides

    // The memory is copied when initializing the np_real_array
    result[1] = np_real_array(beam_buffer);

    return result;
}

std::vector<std::tuple<int, double, int, int, double, double>> source_find_wrapper(
    np_real_array image_data,
    double detection_n_sigma,
    double analysis_n_sigma,
    double rms_est,
    bool find_negative_sources,
    uint sigma_clip_iters,
    bool binapprox_median,
    bool compute_barycentre,
    bool generate_labelmap)
{
    assert(image_data.request().ndim == 2);

    arma::Mat<real_t> image_data_arma(
        static_cast<real_t*>(image_data.request().ptr),
        image_data.request().shape[0], // n_rows
        image_data.request().shape[1], // n_cols
        false, // copy_aux_mem - do not copy memory for better performance (it will not be modified)
        true); // strict

    // Call source find function
    stp::source_find_image sfimage = stp::source_find_image(std::move(image_data_arma), detection_n_sigma, analysis_n_sigma, rms_est,
        find_negative_sources, sigma_clip_iters, binapprox_median, compute_barycentre, generate_labelmap);

    // Convert 'vector of stp::island' to 'vector of tuples'
    std::vector<std::tuple<int, double, int, int, double, double>> v_islands;
    v_islands.reserve(sfimage.islands.size());
    for (auto&& i : sfimage.islands) {
        v_islands.push_back(std::move(std::make_tuple(i.sign, i.extremum_val, i.extremum_x_idx, i.extremum_y_idx, i.xbar, i.ybar)));
    }

    return v_islands;
}

PYBIND11_PLUGIN(stp_python)
{
    pybind11::module m("stp_python", "The Slow Transients Pipeline");

    pybind11::enum_<stp::KernelFunction>(m, "KernelFunction")
        .value("TopHat", stp::KernelFunction::TopHat)
        .value("Triangle", stp::KernelFunction::Triangle)
        .value("Sinc", stp::KernelFunction::Sinc)
        .value("Gaussian", stp::KernelFunction::Gaussian)
        .value("GaussianSinc", stp::KernelFunction::GaussianSinc);

    pybind11::enum_<stp::FFTRoutine>(m, "FFTRoutine")
        .value("FFTW_ESTIMATE_FFT", stp::FFTRoutine::FFTW_ESTIMATE_FFT)
        .value("FFTW_MEASURE_FFT", stp::FFTRoutine::FFTW_MEASURE_FFT)
        .value("FFTW_PATIENT_FFT", stp::FFTRoutine::FFTW_PATIENT_FFT)
        .value("FFTW_WISDOM_FFT", stp::FFTRoutine::FFTW_WISDOM_FFT)
        .value("FFTW_WISDOM_INPLACE_FFT", stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT);

    m.def("image_visibilities_wrapper", &image_visibilities_wrapper, "Compute image visibilities (gridding + ifft).",
        pybind11::arg("vis"), pybind11::arg("snr_weights"), pybind11::arg("uvw_lambda"), pybind11::arg("image_size"), pybind11::arg("cell_size"),
        pybind11::arg("kernel_func") = stp::KernelFunction::GaussianSinc, pybind11::arg("kernel_trunc_radius") = 3.0,
        pybind11::arg("kernel_support") = 3, pybind11::arg("kernel_exact") = true, pybind11::arg("kernel_oversampling") = 9,
        pybind11::arg("generate_beam") = false, pybind11::arg("r_fft") = stp::FFTRoutine::FFTW_ESTIMATE_FFT,
        pybind11::arg("fft_wisdom_filename") = std::string());

    m.def("source_find_wrapper", &source_find_wrapper, "Find connected regions which peak above/below a given threshold.",
        pybind11::arg("image_data"), pybind11::arg("detection_n_sigma"), pybind11::arg("analysis_n_sigma"), pybind11::arg("rms_est") = 0.0,
        pybind11::arg("find_negative_sources") = true, pybind11::arg("sigma_clip_iters") = 5, pybind11::arg("binapprox_median") = false,
        pybind11::arg("compute_barycentre") = true, pybind11::arg("generate_labelmap") = true);

    return m.ptr();
}
}
