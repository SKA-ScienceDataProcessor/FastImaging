#include <armadillo>
#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace stp_python {

/**
 * @brief The KernelFunction enum
 */
enum struct KernelFunction {
    TopHat = 0,
    Triangle,
    Sinc,
    Gaussian,
    GaussianSinc
};

using np_complex_array = pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast>;
using np_double_array = pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;

/**
 * @brief image_visibilities_wrapper
 * @param image_size
 * @param kernel_func
 * @param kernel_trunc_radius
 * @param kernel_support
 * @param kernel_oversampling
 * @param normalize
 * @return
 */
pybind11::tuple image_visibilities_wrapper(
    np_complex_array vis, // numpy.ndarray
    np_double_array uv_pixels, // numpy.ndarray<np.float_>
    int image_size, // int
    KernelFunction kernel_func, // enum
    double kernel_trunc_radius = 3.0, // double
    int kernel_support = 3, // int
    int kernel_oversampling = 1, // int (or none)
    bool normalize = true); // bool
}
