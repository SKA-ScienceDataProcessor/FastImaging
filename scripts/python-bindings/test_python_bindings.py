import stp_python
import numpy as np

# Input simdata file must be located in the current directory
vis_filepath = 'simdata_nstep10.npz'

# This example is not computing residual visibilities. 'vis' component is directly used as input to the pipeline
with open(vis_filepath, 'rb') as f:
    npz_data_dict = np.load(f)
    uvw_lambda = npz_data_dict['uvw_lambda']
    vis = npz_data_dict['vis']
    vis_weights = npz_data_dict['snr_weights']

# Parameters of image_visibilities function
image_size = 8192
cell_size = 0.5
function = stp_python.KernelFunction.GaussianSinc
support = 3
trunc = support
kernel_exact = False
oversampling = 9
generate_beam = False
# Use stp_python.FFTRoutine.FFTW_ESTIMATE_FFT if wisdom files are not available
r_fft = stp_python.FFTRoutine.FFTW_WISDOM_FFT
# The FFTW wisdom files must be located in the current directory
fft_wisdom_filename = '../wisdomfiles/WisdomFile_rob8192x8192.fftw'

# Call image_visibilities
cpp_img, cpp_beam = stp_python.image_visibilities_wrapper(vis, vis_weights, uvw_lambda, image_size, cell_size, function, trunc, support, kernel_exact, oversampling, generate_beam, r_fft, fft_wisdom_filename)

# Parameters of source_find function
detection_n_sigma = 50.0
analysis_n_sigma = 50.0
rms_est = 0.0
find_negative = True
sigma_clip_iters = 5
median_method = 'BINAPPROX'  # Other options: ZEROMEDIAN, BINMEDIAN, NTHELEMENT
gaussian_fitting = True
generate_labelmap = False
ceres_diffmethod = stp_python.CeresDiffMethod.AnalyticDiff_SingleResBlk # Other options: stp_python.CeresDiffMethod.AnalyticDiff, stp_python.CeresDiffMethod.AutoDiff_SingleResBlk, stp_python.CeresDiffMethod.AutoDiff
ceres_solvertype = stp_python.CeresSolverType.LinearSearch_LBFGS # Other options: stp_python.CeresSolverType.LinearSearch_BFGS, stp_python.CeresSolverType.TrustRegion_DenseQR

# Call source_find
islands = stp_python.source_find_wrapper(cpp_img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative, sigma_clip_iters, median_method, gaussian_fitting, generate_labelmap, ceres_diffmethod, ceres_solvertype)

# Print result
for i in islands:
   print(i)
   print()
