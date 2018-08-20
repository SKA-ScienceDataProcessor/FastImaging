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
    // Convert string to KernelFunction enum
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
    } else if (kernel == "PSWF") {
        k_func = stp::KernelFunction::PSWF;
    } else {
        assert(0);
    }

    return k_func;
}

stp::FFTRoutine ConfigurationFile::parse_fft_routine(const std::string& fft)
{
    // Convert string to FFTRoutine enum
    stp::FFTRoutine r_fft = stp::FFTRoutine::FFTW_ESTIMATE_FFT;
    if (fft == "FFTW_ESTIMATE_FFT") {
        r_fft = stp::FFTRoutine::FFTW_ESTIMATE_FFT;
    } else if (fft == "FFTW_MEASURE_FFT") {
        r_fft = stp::FFTRoutine::FFTW_MEASURE_FFT;
    } else if (fft == "FFTW_PATIENT_FFT") {
        r_fft = stp::FFTRoutine::FFTW_PATIENT_FFT;
    } else if (fft == "FFTW_WISDOM_FFT") {
        r_fft = stp::FFTRoutine::FFTW_WISDOM_FFT;
    } else if (fft == "FFTW_WISDOM_INPLACE_FFT") {
        r_fft = stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT;
    } else {
        assert(0);
    }

    return r_fft;
}

stp::InterpType ConfigurationFile::parse_interp_type(const std::string& it)
{
    // Convert string to InterpType enum
    stp::InterpType r_it = stp::InterpType::LINEAR;
    if (it == "linear") {
        r_it = stp::InterpType::LINEAR;
    } else if (it == "cosine") {
        r_it = stp::InterpType::COSINE;
    } else if (it == "cubic") {
        r_it = stp::InterpType::CUBIC;
        //    } else if (it == "cubic_smooth") {
        //        r_it = stp::InterpType::CUBIC_SMOOTH;
    } else {
        assert(0);
    }

    return r_it;
}

stp::MedianMethod ConfigurationFile::parse_median_method(const std::string& medianmethod)
{
    // Convert string to MedianMethod enum
    stp::MedianMethod e_median = stp::MedianMethod::BINAPPROX;
    if (medianmethod == "ZEROMEDIAN") {
        e_median = stp::MedianMethod::ZEROMEDIAN;
    } else if (medianmethod == "BINMEDIAN") {
        e_median = stp::MedianMethod::BINMEDIAN;
    } else if (medianmethod == "BINAPPROX") {
        e_median = stp::MedianMethod::BINAPPROX;
    } else if (medianmethod == "NTHELEMENT") {
        e_median = stp::MedianMethod::NTHELEMENT;
    } else {
        assert(0);
    }

    return e_median;
}

stp::CeresDiffMethod ConfigurationFile::parse_ceres_diffmethod(const std::string& diffmet)
{
    // Convert string to CeresDifMethod enum
    stp::CeresDiffMethod e_diffmet = stp::CeresDiffMethod::AutoDiff_SingleResBlk;
    if (diffmet == "AutoDiff") {
        e_diffmet = stp::CeresDiffMethod::AutoDiff;
    } else if (diffmet == "AutoDiff_SingleResBlk") {
        e_diffmet = stp::CeresDiffMethod::AutoDiff_SingleResBlk;
    } else if (diffmet == "AnalyticDiff") {
        e_diffmet = stp::CeresDiffMethod::AnalyticDiff;
    } else if (diffmet == "AnalyticDiff_SingleResBlk") {
        e_diffmet = stp::CeresDiffMethod::AnalyticDiff_SingleResBlk;
    } else {
        assert(0);
    }

    return e_diffmet;
}

stp::CeresSolverType ConfigurationFile::parse_ceres_solvertype(const std::string& solvertype)
{
    // Convert string to CeresSolverType enum
    stp::CeresSolverType e_solvertype = stp::CeresSolverType::LinearSearch_BFGS;
    if (solvertype == "LinearSearch_BFGS") {
        e_solvertype = stp::CeresSolverType::LinearSearch_BFGS;
    } else if (solvertype == "LinearSearch_LBFGS") {
        e_solvertype = stp::CeresSolverType::LinearSearch_LBFGS;
    } else if (solvertype == "TrustRegion_DenseQR") {
        e_solvertype = stp::CeresSolverType::TrustRegion_DenseQR;
    } else {
        assert(0);
    }

    return e_solvertype;
}
