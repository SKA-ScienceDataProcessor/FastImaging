#include "load_json_config.h"

rapidjson::Document load_json_configuration(std::string& cfg)
{
    std::ifstream file(cfg);
    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    rapidjson::Document aux;
    aux.Parse(str.c_str());

    return aux;
}

KernelFunction parse_kernel_function(const std::string& kernel)
{
    // Convert kernel function string to KernelFunction enum
    KernelFunction k_func = KernelFunction::GaussianSinc;
    if (kernel == "TopHat") {
        k_func = KernelFunction::TopHat;
    } else if (kernel == "Triangle") {
        k_func = KernelFunction::Triangle;
    } else if (kernel == "Sinc") {
        k_func = KernelFunction::Sinc;
    } else if (kernel == "Gaussian") {
        k_func = KernelFunction::Gaussian;
    } else if (kernel == "GaussianSinc") {
        k_func = KernelFunction::GaussianSinc;
    }

    return k_func;
}

stp::fft_routine parse_fft_routine(const std::string& fft)
{
    // Convert fft routine string to enum
    stp::fft_routine r_fft = stp::FFTW_ESTIMATE_FFT;
    if (fft == "FFTW_ESTIMATE_FFT") {
        r_fft = stp::FFTW_ESTIMATE_FFT;
    } else if (fft == "FFTW_MEASURE_FFT") {
        r_fft = stp::FFTW_MEASURE_FFT;
    } else if (fft == "FFTW_PATIENT_FFT") {
        r_fft = stp::FFTW_PATIENT_FFT;
    } else if (fft == "FFTW_WISDOM_FFT") {
        r_fft = stp::FFTW_WISDOM_FFT;
    } else if (fft == "ARMADILLO_FFT") {
        r_fft = stp::ARMADILLO_FFT;
    }

    return r_fft;
}
