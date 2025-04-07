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
## Date last modified : 03-May-2024      ##
###########################################

import os

import typing
from pathlib import Path
import multiprocessing as mp

from geollama import objects


def generate_config(output_path):
    """
    Generates default configuration file

    Args:
    output_path (str) : Path to output yaml file
    """
    import ruamel.yaml

    config_str = """ \
# ESSENTIAL PARAMETERS
# data_path: Path to folder holding all tomograms. (NB. Direct path to individual tomogram file not supported.)
# pixel_size_nm: Pixel size of input tomogram in nm. If input tomograms are binned, use binned pixel sizes.

data_path: null
pixel_size_nm: null


# OPTIONAL PARAMETERS
# binning: Internal binning factor for tomogram evaluation. Recommended target x-y dimensions from (128, 128) to (256, 256). E.g. if input tomogram has shape (2048, 2048, 2048), use -b 8 or -b 16. Use 0 (default) for auto-binning.
# autocontrast: Apply autocontrast to slices prior to evaluation. Recommended.
# adaptive: Use adaptive sampling for slice evaluation. Recommended.
# bandpass: Apply bandpass filter to tomograms prior to evaluation.
# num_cores: Number of CPUs used.
# output_csv_path: Output path for CSV file. Leave blank if no CSV file output required.
# output_star_path: Output path for STAR file. Leave blank if no STAR file output required.
# output_mask: Output volumetric binary masks. If true, masks will be saved in ./volume_masks/ with same filenames as input tomogram
# generate_report: Automatically generate report at the end of calculations.
# printout: Print statistical output after evaluation. Recommended for standalone use of GeoLlama

binning: 0
autocontrast: True
adaptive: True
bandpass: False
num_cores: 1
output_csv_path: null
output_star_path: ./output.star
output_mask: True
generate_report: True
printout: True


# ANALYTICS PARAMETERS for feature extraction
# thickness_lower_limit: Lower limit of lamella thickness in nm (for feature extraction)
# thickness_upper_limit: Upper limit of lamella thickness in nm (for feature extraction)
# thickness_std_limit: Limit of lamella thickness standard deviation in nm (for feature extraction)
# xtilt_std_limit: Limit of lamella xtilt standard deviation in degrees (for feature extraction)
# displacement_limit: Limit of lamella centroid displacement in % (for feature extraction)
# displacement_std_limit: Limit of lamella centroid displacement standard deviation in % (for feature extraction)

thickness_lower_limit: 120
thickness_upper_limit: 300
thickness_std_limit: 15
xtilt_std_limit: 5
displacement_limit : 25
displacement_std_limit : 5
"""

    yaml = ruamel.yaml.YAML()
    data = yaml.load(config_str)

    yaml.default_flow_style = False
    yaml.dump(data, Path(output_path))


def read_config(config_fname: str) -> objects.Config:
    """
    Parse GeoLlama config YAML file

    Args:
    config_fname (str) : Path to config file

    Returns:
    Config
    """
    import ruamel.yaml

    if not os.path.exists(config_fname):
        raise IOError(f"{config_fname} does not exist.")

    yaml = ruamel.yaml.YAML()
    config_dict = yaml.load(Path(config_fname))

    # Initialise a Config object
    params = objects.Config()

    # Check config file has correct dictionary keys
    for key in params.__dict__.keys():
        try:
            params.__setattr__(key, config_dict[key])
        except ValueError:
            print(f"{key} keyword missing in config file.")

    return params


def objectify_user_input(
    autocontrast: typing.Optional[bool],
    adaptive: typing.Optional[bool],
    bandpass: typing.Optional[bool],
    data_path: typing.Optional[str],
    pixel_size_nm: typing.Optional[float],
    binning: typing.Optional[int],
    num_cores: typing.Optional[int],
    output_csv_path: typing.Optional[str],
    output_star_path: typing.Optional[str],
    output_mask: typing.Optional[bool],
    generate_report: typing.Optional[bool],
    printout: typing.Optional[bool],
    thickness_lower_limit: typing.Optional[float],
    thickness_upper_limit: typing.Optional[float],
    thickness_std_limit: typing.Optional[float],
    xtilt_std_limit: typing.Optional[float],
    displacement_limit: typing.Optional[float],
    displacement_std_limit: typing.Optional[float],
) -> objects.Config:
    """
    Objectifying user provided input as a Config object

    Args:
    autocontrast (bool) : Apply autocontrast to slices prior to evaluation
    adaptive (bool) : Use adaptive sampling for slice evaluation
    bandpass (bool) : Apply bandpass filter to tomograms prior to evaluation
    user_path (str) : Path to folder holding all tomograms in batch mode
    pixel_size (float) : Pixel size of input tomogram in nm
    binning (int) : Binning factor for tomogram evaluation
    cpu (int) : Number of CPUs used
    output_csv (str) : Output path for CSV file
    output_star (str) : Output path for STAR file
    output_mask (bool) : Produce 3D mask of estimated lamella region (same shape as input tomogram)
    generate_report (bool) : Automatically generate report at the end of calculations
    thickness_lower_limit (float) : Lower limit of lamella thickness in nm (for feature extraction)
    thickness_upper_limit (float) : Upper limit of lamella thickness in nm (for feature extraction)
    thickness_std_limit (float) : Limit of lamella thickness standard deviation in nm (for feature extraction)
    xtilt_std_limit (float) : Limit of lamella xtilt standard deviation in degrees (for feature extraction)
    displacement_limit (float) : Limit of lamella centroid displacement in % (for feature extraction)
    displacement_std_limit (float) : Limit of lamella centroid displacement standard deviation in % (for feature extraction)

    Returns:
    Config
    """
    params = objects.Config()
    for key in params.__dict__.keys():
        params.__setattr__(key, locals()[key])

    return params


def check_config(params: objects.Config):
    """
    Check datatypes in parameters

    Args:
    config (Config) : Config object storing all parameters
    """

    if params.data_path is None:
        raise ValueError("Data path (-p) must be given.")
    elif not os.path.isdir(Path(params.data_path)):
        raise NotADirectoryError("Given path must be a folder.")

    if params.pixel_size_nm is None:
        raise ValueError("Pixel size (-s) must be given.")
    elif params.pixel_size_nm <= 0:
        raise ValueError("Pixel size (-s) must be a positive number.")

    if not isinstance(params.num_cores, int):
        raise ValueError("num_cores (-np) must be an integer.")
    elif not 1 <= params.num_cores <= mp.cpu_count():
        raise ValueError(
            f"num_cores (-np) must be between 1 and # CPUs available ({mp.cpu_count()}). Current settings: {params.num_cores}"
        )

    if params.generate_report and params.output_star_path is None:
        raise ValueError(
            "Output STAR file must be specified for automatic report generation."
        )
