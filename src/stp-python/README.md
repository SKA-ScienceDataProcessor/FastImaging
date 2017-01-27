# Python Bindings for Slow Transients Pipeline Prototype
## Procedure for calling wrapper functions
- Add the location of the stp_python.so shared library to the python path
- Import stp_python, as well as other important modules such as numpy
- Setup the variables and read input files required for the wrapper functions
- Call the wrapper function

## Example
```python
import numpy as np
import stp_python

vis_npz_filepath = '<path>/FastImaging/test-data/pipeline-data/simdata_small.npz'

with open(vis_npz_filepath, 'rb') as f:
    npz_data_dict = np.load(f)
    uvw_lambda = npz_data_dict['uvw_lambda']
    vis = npz_data_dict['vis']
    model = npz_data_dict['model']

res=vis-model
image_size = 1024
cell_size = 1
trunc = 3.0
support = 3
kernel_exact = False
oversampling = 9

# Call image_visibilities_wrapper
cpp_img, cpp_beam = stp_python.image_visibilities_wrapper(res, uvw_lambda, image_size, cell_size, stp_python.KernelFunction.GaussianSinc, trunc, support, kernel_exact, oversampling, True)

# Call source_find_wrapper
isl = stp_python.source_find_wrapper(np.real(cpp_img), 50, 50, 0.0)

print isl
```

