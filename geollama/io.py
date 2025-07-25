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
## Date last modified : 22-Jul-2025  ##
#######################################

import logging
from pathlib import Path
from typing import Union

import numpy.typing as npt

import mrcfile
from skimage.transform import downscale_local_mean as DSLM

from geollama import objects


def read_mrc(
    fname: Union[str, Path],
    params: objects.Config,
) -> (npt.NDArray[any], float):
    """
    Read in an mrc image file, then output data as ndarray with essential metadata.
    If user specifies `binning == 0` then determine the internal binning factor such that the shortest dimension of the
    volume is downscaled to a length of about 128 pixels.

    Parameters
    ----------
    fname : str or Path
        Path to image file
    params : Config
        Config object containing all parameters for GeoLlama

    Returns
    -------
    data : ndarray
        - If `binning == 1`, return the original volumetric data
        - If `binning > 1`, return the binned data
    pixel_size : float
        - If `binning == 1`, return the original pixel spacing
        - If `binning > 1`, return the pixel spacing after binning
    original_shape : tuple
        Shape (dimensionality) of original volume
    binning : int
        Actual internal binning factor
    data : ndarray or None
        - If `binning == 1`, return None
        - If `binning > 1`, return original (unbinned) volume
    """
    with mrcfile.open(fname) as f:
        data = f.data
        original_shape = data.shape

    # Determine binning factor if auto-binning used
    binning = params.binning
    if params.binning == 0:
        binning = max(1, min(original_shape[1:]) // 128)
        logging.info(f"AUTOBIN: {fname.name} - Binning factor={binning}")

    if binning > 1:
        data_ds = DSLM(data, (binning, binning, binning))
        return (data_ds, params.pixel_size_nm * binning, original_shape, binning, data)

    return (data, params.pixel_size_nm, original_shape, binning, None)
