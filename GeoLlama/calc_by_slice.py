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

##################################################
## Module             : GeoLlama.calc_by_slice  ##
## Created            : Neville Yee             ##
## Date created       : 02-Oct-2023             ##
## Date last modified : 02-Oct-2023             ##
##################################################


import numpy as np
import numpy.typing as npt


def create_slice_views(volume,
                       coords,
                       sum=10
) -> (npt.NDArray[any], npt.NDArray[any], npt.NDArray[any]):

    sum_half = sum // 2
    z_range = (max(0, coords[0]-sum_half), min(coords[0]+sum-sum_half, volume.shape[0]-1))
    x_range = (max(0, coords[1]-sum_half), min(coords[1]+sum-sum_half, volume.shape[1]-1))
    y_range = (max(0, coords[2]-sum_half), min(coords[2]+sum-sum_half, volume.shape[2]-1))

    view_xy = np.std(volume[z_range[0]: z_range[1], :, :], axis=0)
    view_zy = np.std(volume[:, x_range[0]: x_range[1], :], axis=1)
    view_zx = np.std(volume[:, :, y_range[0]: y_range[1]], axis=2)

    return (view_xy, view_zy, view_zx)
