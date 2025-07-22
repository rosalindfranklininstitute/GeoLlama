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
## Date last modified : 22-Jul-2025      ##
###########################################

import os

import typing
from pathlib import Path
import multiprocessing as mp


def generate_config(output_path):
    """
    Generate default configuration YAML file.

    Parameters
    ----------
    output_path : str
        Path to output YAML file
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


def read_config(config_fname: str) -> dict:
    """
    Parse GeoLlama config YAML file and return a dictionary of parameters.

    Parameters
    ----------
    config_fname : str
        Path to YAML configuration file

    Returns
    -------
    config_dict : dict
        Dictionary containing all parameters for GeoLlama
    """
    import ruamel.yaml

    if not os.path.exists(config_fname):
        raise IOError(f"{config_fname} does not exist.")

    yaml = ruamel.yaml.YAML()
    config_dict = yaml.load(Path(config_fname))

    return config_dict
