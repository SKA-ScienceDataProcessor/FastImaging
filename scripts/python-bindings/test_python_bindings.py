import stp_python
import numpy as np

# Input simdata file must be located in the current directory
vis_filepath = 'simdata_nstep10.npz'

# This example is not computing residual visibilities. 'vis' component is directly used as input to the pipeline
with open(vis_filepath, 'rb') as f:
    npz_data_dict = np.load(f)
    uvw_lambda = npz_data_dict['uvw_lambda']
    vis = npz_data_dict['vis']

# Parameters of image_visibilities function
image_size = 8192
cell_size = 0.5
function = stp_python.KernelFunction.GaussianSinc
support = 3
trunc = support
kernel_exact = False
oversampling = 9
normalize_image = True
normalize_beam = True
r_fft = stp_python.FFTRoutine.FFTW_WISDOM_FFT
# The FFTW wisdom files must be located in the current directory
image_wisdom_filename = 'WisdomFile_rob8192x8192.fftw' 
beam_wisdom_filename = 'WisdomFile_rob8192x8192.fftw'

# Call image_visibilities
cpp_img, cpp_beam = stp_python.image_visibilities_wrapper(vis, uvw_lambda, image_size, cell_size, function, trunc, support, kernel_exact, oversampling, normalize_image, normalize_beam, r_fft, image_wisdom_filename, beam_wisdom_filename)

# Parameters of source_find function
detection_n_sigma = 50.0
analysis_n_sigma = 50.0
rms_est = 0.0
find_negative = True
sigma_clip_iters = 5
binapprox_median = True
compute_barycentre = True
generate_labelmap = False

# Call source_find
islands = stp_python.source_find_wrapper(cpp_img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative, sigma_clip_iters, binapprox_median, compute_barycentre, generate_labelmap)

# Print result
print(islands)
