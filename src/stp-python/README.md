# Python Bindings for Slow Transients Pipeline Prototype
## Procedure for calling image_visibilities function
- Add the location of the stp_python.so shared library to the python path
- Import stp_python, as well as other important modules such as numpy
- Setup the variables and read input files required for the image_visibilities function
- Call the image_visibilities_wrapper (wrapper for C++ implementation of image_visibities function callable from python)

## Example
```
import numpy as np
import astropy.units as u
import stp_python

simple_vis_npz_filepath = './simple_vis.npz'

with open(simple_vis_npz_filepath, 'rb') as f:
	npz_data_dict = np.load(f)
	uvw_lambda = npz_data_dict['uvw_lambda'] # dtype = float64
	vis = npz_data_dict['vis']               # dtype = complex64

trunc = 3.0
support = 3
image_size = 1024 * u.pix
cell_size = 3. * u.arcsec
oversampling = 2

image_size = image_size.to(u.pix)
cpp_img, cpp_beam = stp_python.image_visibilities_wrapper(
	vis,                                    # vis
	uv_in_pixels,                           # uv_pixels
	int(image_size.value),                  # image_size
	stp_python.KernelFunction.GaussianSinc, # kernel_func_name
	trunc,                                  # kernel_trunc_radius
	support,                                # kernel_support
	oversampling,                           # kernel_oversampling
	True                                    # normalize
)
```

