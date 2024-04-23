# Copyright 2023 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

#######################################
## Module             : GeoLlama.io  ##
## Created            : Neville Yee  ##
## Date created       : 02-Oct-2023  ##
## Date last modified : 02-Oct-2023  ##
#######################################


import numpy as np
import numpy.typing as npt

import mrcfile
from skimage.transform import downscale_local_mean as DSLM


def read_mrc(fname: str,
             px_size_nm: float,
             downscale: int=2) -> (npt.NDArray[any], float):
    """
    Function to read in MRC image file.

    Args:
    fname (str) : Path to image file
    px_size_nm (float) : Pixel size of original tomogram in nm
    downscale (int) : Internal binning factor

    Returns:
    ndarray, float
    """

    with mrcfile.open(fname) as f:
        data = f.data

    if downscale > 1:
        data_ds = DSLM(data, (downscale, downscale, downscale))
        return (data_ds, px_size_nm*downscale)

    return (data, px_size_nm)
