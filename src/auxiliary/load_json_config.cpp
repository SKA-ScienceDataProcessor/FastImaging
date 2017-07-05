#include "load_json_config.h"

rapidjson::Document ConfigurationFile::load_json_configuration(const std::string& cfg)
{
    std::ifstream file(cfg);
    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    if (str.empty()) {
        throw std::runtime_error("Failed to read configuration file: " + cfg);
    }

    rapidjson::Document aux;
    aux.Parse(str.c_str());

    return aux;
}

stp::KernelFunction ConfigurationFile::parse_kernel_function(const std::string& kernel)
{
    // Convert kernel function string to KernelFunction enum
    stp::KernelFunction k_func = stp::KernelFunction::GaussianSinc;
    if (kernel == "TopHat") {
        k_func = stp::KernelFunction::TopHat;
    } else if (kernel == "Triangle") {
        k_func = stp::KernelFunction::Triangle;
    } else if (kernel == "Sinc") {
        k_func = stp::KernelFunction::Sinc;
    } else if (kernel == "Gaussian") {
        k_func = stp::KernelFunction::Gaussian;
    } else if (kernel == "GaussianSinc") {
        k_func = stp::KernelFunction::GaussianSinc;
    }

    return k_func;
}

stp::FFTRoutine ConfigurationFile::parse_fft_routine(const std::string& fft)
{
    // Convert fft routine string to enum
    stp::FFTRoutine r_fft = stp::FFTW_ESTIMATE_FFT;
    if (fft == "FFTW_ESTIMATE_FFT") {
        r_fft = stp::FFTW_ESTIMATE_FFT;
    } else if (fft == "FFTW_MEASURE_FFT") {
        r_fft = stp::FFTW_MEASURE_FFT;
    } else if (fft == "FFTW_PATIENT_FFT") {
        r_fft = stp::FFTW_PATIENT_FFT;
    } else if (fft == "FFTW_WISDOM_FFT") {
        r_fft = stp::FFTW_WISDOM_FFT;
    } else if (fft == "FFTW_WISDOM_INPLACE_FFT") {
        r_fft = stp::FFTW_WISDOM_INPLACE_FFT;
    }

    return r_fft;
}
