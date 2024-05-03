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

from GeoLlama import objects


def generate_config(output_path):
    """
    Generates default configuration file

    Args:
    output_path (str) : Path to output yaml file
    """
    import ruamel.yaml

    config_str = """ \
# Essential parameters
data_path: None
pixel_size_nm: None

# Optional parameters
binning: 1
autocontrast: False
adaptive: False
bandpass: False
num_cores: 1
output_csv_path: None
output_star_path: None
"""

    yaml = ruamel.yaml.YAML()
    data = yaml.load(config_str)

    yaml.default_flow_style=False
    yaml.dump(data, Path(output_path))


def read_config(
        config_fname: str
) -> objects.Config:
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
) -> objects.Config:
    """
    Objectifying user provided input as a Config object

    Args:
    autocontrast (bool) : Apply autocontrast to slices prior to evaluation
    adaptive (bool) : Use adaptive sampling for slice evaluation
    bandpass (bool) : Apply bandpass filter to tomograms prior to evaluation
    user_path (str) : Path to folder holding all tomograms in batch mode
    pixel_size (float) : Tomogram pixel size in nm
    binning (int) : Binning factor for tomogram evaluation
    cpu (int) : Number of CPUs used
    out_csv (str) : Output path for CSV file
    out_star (str) : Output path for STAR file

    Returns:
    Config
    """
    params = objects.Config()
    for key in params.__dict__.keys():
        params.__setattr__(key, locals()[key])

    return params


def check_config(
        params: objects.Config
):
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
        raise ValueError(f"num_cores (-np) must be between 1 and # CPUs available ({mp.cpu_count()}).")
