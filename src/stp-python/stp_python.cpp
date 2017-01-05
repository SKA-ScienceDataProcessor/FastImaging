#include "stp_python.h"
#include <pybind11/stl.h>
#include <stp.h>
#include <utility>

namespace stp_python {

// Convenient aliases
using ptr_complex_double = std::complex<double>*;
using ptr_double = double*;

pybind11::tuple image_visibilities_wrapper(
    np_complex_array vis,
    np_double_array uv_pixels,
    int image_size,
    KernelFunction kernel_func,
    double kernel_trunc_radius,
    int kernel_support,
    int kernel_oversampling,
    bool normalize)
{
    assert(vis.request().ndim == 1); // vis is a 1D array
    assert(uv_pixels.request().ndim == 2); // uv_pixels is a 2D array

    arma::cx_mat vis_arma(
        static_cast<ptr_complex_double>(vis.request().ptr),
        vis.request().shape[0],
        vis.request().shape[1],
        true, // copy_aux_mem - this prevents the object on the python side from being modified
        true); // strict

    arma::mat ux_pixels_arma(
        static_cast<ptr_double>(uv_pixels.request().ptr),
        uv_pixels.request().shape[1], // dimensions are reversed because numpy
        uv_pixels.request().shape[0], // uses row-major by default
        true,
        true);

    // make sure dimensions are right
    arma::inplace_trans(ux_pixels_arma);

    // (image, beam) tuple
    std::pair<arma::cx_mat, arma::cx_mat> image_and_beam;

    // Since "image_visibilities" is a function template, there's no
    // way to prevent the use of the following switch statement. Inheritance
    // could be used instead, but virtual function calls would impose an
    // unnecessary performance penalty.
    switch (kernel_func) {
    case KernelFunction::TopHat:
        // TODO: TopHat
        break;
    case KernelFunction::Triangle:
        // TODO: Triangle
        break;
    case KernelFunction::Sinc:
        // TODO: Sinc
        break;
    case KernelFunction::Gaussian:
        // TODO: Gaussian
        break;
    case KernelFunction::GaussianSinc: {
        stp::GaussianSinc kernel(kernel_trunc_radius);
        image_and_beam = stp::process_image_visibilities(kernel, kernel_support, image_size, ux_pixels_arma, vis_arma, kernel_oversampling, normalize);
    };
        break;
    default:
        break;
    }

    // The function will output a tuple with two np_complex_array
    pybind11::tuple result(2);

    // Image at index 0
    arma::cx_mat image(std::get<0>(image_and_beam));
    arma::inplace_strans(image); // needed due to arma being column-major
    size_t data_size = sizeof(arma::cx_double);
    pybind11::buffer_info image_buffer(
        static_cast<void*>(image.memptr()), // void *ptr
        data_size, // size_t itemsize
        "Zd", // const std::string &format
        2, // size_t ndim
        { image.n_rows, image.n_cols }, // const std::vector<size_t> &shape
        { image.n_cols * data_size, data_size }); // const std::vector<size_t> &strides
    result[0] = np_complex_array(image_buffer);

    // Beam at index 1
    arma::cx_mat beam(std::get<1>(image_and_beam));
    arma::inplace_strans(beam); // needed due to arma being column-major
    pybind11::buffer_info beam_buffer(
        static_cast<void*>(beam.memptr()), // void *ptr
        data_size, // size_t itemsize
        "Zd", // const std::string &format
        2, // size_t ndim
        { beam.n_rows, beam.n_cols }, // const std::vector<size_t> &shape
        { beam.n_cols * data_size, data_size });
    result[1] = np_complex_array(beam_buffer);

    return result;
}

PYBIND11_PLUGIN(stp_python)
{
    pybind11::module m("stp_python", "The Slow Transients Pipeline");

    pybind11::enum_<KernelFunction>(m, "KernelFunction")
        .value("TopHat", KernelFunction::TopHat)
        .value("Triangle", KernelFunction::Triangle)
        .value("Sinc", KernelFunction::Sinc)
        .value("Gaussian", KernelFunction::Gaussian)
        .value("GaussianSinc", KernelFunction::GaussianSinc);

    m.def("image_visibilities_wrapper", &image_visibilities_wrapper, "Compute image visibilities");

    // TODO: add the remaining binding functions
    return m.ptr();
}
}
