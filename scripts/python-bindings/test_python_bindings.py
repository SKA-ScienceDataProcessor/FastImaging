import stp_python
import numpy as np

# Input simdata file must be located in the current directory
vis_filepath = 'simdata_awproj_nstep10.npz'

# This example is not computing residual visibilities. 'vis' component is directly used as input to the pipeline
with open(vis_filepath, 'rb') as f:
    npz_data_dict = np.load(f)
    uvw_lambda = npz_data_dict['uvw_lambda']
    vis = npz_data_dict['vis']
    snr_weights = npz_data_dict['snr_weights']
    lha = npz_data_dict['lha']

# Parameters of image_visibilities function
image_size = 1024
cell_size = 100
kernel_func_name = stp_python.KernelFunction.PSWF
kernel_support = 3
kernel_exact = False
kernel_oversampling = 8
generate_beam = False
grid_image_correction = True
fft_routine = stp_python.FFTRoutine.FFTW_ESTIMATE_FFT
fft_wisdom_filename = ""
analytic_gcf = False
num_wplanes = 50
wplanes_median = False
max_wpconv_support = 30
hankel_opt = False
undersampling_opt = 1
kernel_trunc_perc = 1.0
interp_type = stp_python.InterpType.LINEAR
aproj_numtimesteps = 0
obs_dec = 0.0
obs_lat = 0.0
mueller_term = np.ones((image_size, image_size))

# Call image_visibilities
cpp_img, cpp_beam = stp_python.image_visibilities_wrapper(
    vis,
    snr_weights,
    uvw_lambda,
    image_size,
    cell_size,
    kernel_func_name,
    kernel_support,
    kernel_exact,
    kernel_oversampling,
    generate_beam,
    grid_image_correction,
    analytic_gcf,
    fft_routine,
    fft_wisdom_filename,
    num_wplanes,
    wplanes_median,
    max_wpconv_support,
    hankel_opt,
    undersampling_opt,
    kernel_trunc_perc,
    interp_type,
    aproj_numtimesteps,
    obs_dec,
    obs_lat,
    lha,
    mueller_term
)
    

# Parameters of source_find function
detection_n_sigma = 50.0
analysis_n_sigma = 50.0
rms_est = 0.0
find_negative = True
sigma_clip_iters = 5
median_method = stp_python.MedianMethod.BINAPPROX  # Other options: stp_python.MedianMethod.ZEROMEDIAN, stp_python.MedianMethod.BINMEDIAN, stp_python.MedianMethod.NTHELEMENT
gaussian_fitting = True
ccl_4connectivity = False
generate_labelmap = False
source_min_area = 5
ceres_diffmethod = stp_python.CeresDiffMethod.AnalyticDiff_SingleResBlk # Other options: stp_python.CeresDiffMethod.AnalyticDiff, stp_python.CeresDiffMethod.AutoDiff_SingleResBlk, stp_python.CeresDiffMethod.AutoDiff
ceres_solvertype = stp_python.CeresSolverType.LinearSearch_LBFGS # Other options: stp_python.CeresSolverType.LinearSearch_BFGS, stp_python.CeresSolverType.TrustRegion_DenseQR

# Call source_find
islands = stp_python.source_find_wrapper(
    cpp_img, 
    detection_n_sigma, 
    analysis_n_sigma, 
    rms_est, 
    find_negative, 
    sigma_clip_iters, 
    median_method, 
    gaussian_fitting, 
    ccl_4connectivity, 
    generate_labelmap, 
    source_min_area, 
    ceres_diffmethod, 
    ceres_solvertype
)

# Print result
for i in islands:
   print(i)
   print()
