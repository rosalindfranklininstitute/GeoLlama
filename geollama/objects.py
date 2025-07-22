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

###########################################
## Module             : GeoLlama.config  ##
## Created            : Neville Yee      ##
## Date created       : 03-May-2024      ##
## Date last modified : 21-Jul-2025      ##
###########################################

from pathlib import Path
import os
from dataclasses import dataclass
import typing
import multiprocessing as mp

import numpy.typing as npt


class Config():
    """
    Object containing parameters for GeoLlama, and a self-validation function.

    Parameters
    ----------
    data_path : str
        Path to folder containing tomograms for evaluation
    pixel_size_nm : float
        Pixel spacing of input tomograms (in nm)
    binning : int
        Internal (additional) binning factor for GeoLlama
    autocontrast : bool
        Apply auto-contrast on tomogram slices as preprocessing step before evaluation
    adaptive : bool
        Use adaptive sampling in evaluation
    bandpass : bool
        Apply bandpass filter on tomogram slices as preprocessing step before evaluation
    num_cores : int
        Number of CPUs used for GeoLlama evaluation in parallel
    output_csv_path : str
        Path to output CSV file containing analytical results
    output_star_path : str
        Path to output STAR file containing analytical results
    generate_report : bool
        Generate detailed report after evaluation
    output_mask : bool
        Generate volumetric mask after evaluation
    printout : bool
        Print statistical output after evaluation
    thickness_lower_limit : float
        Lower limit of lamella thickness in nm. (for feature extraction)
    thickness_upper_limit : float
        Upper limit of lamella thickness in nm. (for feature extraction)
    thickness_std_limit : float
        Limit of lamella thickness standard deviation in nm. (for feature extraction)
    xtilt_std_limit : float
        Limit of lamella xtilt standard deviation in degrees. (for feature extraction)
    displacement_limit : float
        Limit of lamella centroid displacement in %. (for feature extraction)
    displacement_std_limit : float
        Limit of lamella centroid displacement standard deviation in %. (for feature extraction)

    Notes
    -----
    The `pixel_size_nm` parameter assumes that all tomograms in the given folder (`data_path`) have the same pixel spacing.
    Moreover, it also assumes that the pixel spacing is consistent among the three axes of a tomogram.
    """

    def __init__(self,
        data_path: str,
        pixel_size_nm: float,
        binning: int,
        autocontrast: bool,
        adaptive: bool,
        bandpass: bool,
        num_cores: int,
        output_csv_path: str,
        output_star_path: str,
        generate_report: bool,
        output_mask: bool,
        printout: bool,
        thickness_lower_limit: float,
        thickness_upper_limit: float,
        thickness_std_limit: float,
        xtilt_std_limit: float,
        displacement_limit: float,
        displacement_std_limit: float,
    ):
        self.data_path = data_path
        self.pixel_size_nm = pixel_size_nm
        self.binning = binning
        self.autocontrast = autocontrast
        self.adaptive = adaptive
        self.bandpass = bandpass
        self.num_cores = num_cores
        self.output_csv_path = output_csv_path
        self.output_star_path = output_star_path
        self.generate_report = generate_report
        self.output_mask = output_mask
        self.printout = printout
        self.thickness_lower_limit = thickness_lower_limit
        self.thickness_upper_limit = thickness_upper_limit
        self.thickness_std_limit = thickness_std_limit
        self.xtilt_std_limit = xtilt_std_limit
        self.displacement_limit = displacement_limit
        self.displacement_std_limit = displacement_std_limit


    def validate(self):
        """
        Self-validate parameters.

        Raises
        ------
        ValueError
            - If `data_path` is not provided
            - If `pixel_size_nm` is not provided or `pixel_size_nm <= 0`
            - If `num_core` is not an integer or `num_core` outside the range of physically accessible number of CPUs
            - If `output_star_path` is not provided when `generate_report == True`
        NotADirectoryError
            - If `data_path` does not point to a folder
        """
        if self.data_path is None:
            raise ValueError("Data path (-p) must be given.")
        elif not os.path.isdir(Path(self.data_path)):
            raise NotADirectoryError("Given path must be a folder.")

        if self.pixel_size_nm is None:
            raise ValueError("Pixel size (-s) must be given.")
        elif self.pixel_size_nm <= 0:
            raise ValueError("Pixel size (-s) must be a positive number.")

        if not isinstance(self.num_cores, int):
            raise ValueError("num_cores (-np) must be an integer.")
        elif not 1 <= self.num_cores <= mp.cpu_count():
            raise ValueError(
                f"num_cores (-np) must be between 1 and # CPUs available ({mp.cpu_count()}). Current settings: {self.num_cores}"
            )

        if self.generate_report and self.output_star_path is None:
            raise ValueError(
                "Output STAR file must be specified for automatic report generation."
            )


@dataclass()
class Result:
    yz_stats: npt.NDArray[any] = None
    xz_stats: npt.NDArray[any] = None
    yz_mean: npt.NDArray[any] = None
    xz_mean: npt.NDArray[any] = None
    yz_sem: npt.NDArray[any] = None
    xz_sem: npt.NDArray[any] = None
    surfaces: typing.Optional[tuple] = None
    binning_factor: typing.Optional[int] = None
    adaptive_triggered: typing.Optional[bool] = None
