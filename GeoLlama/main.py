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

#########################################
## Module             : GeoLlama.main  ##
## Created            : Neville Yee    ##
## Date created       : 03-Oct-2023    ##
## Date last modified : 26-Apr-2024    ##
#########################################

import os
from pprint import pprint
from pathlib import Path
import typing
from typing_extensions import Annotated
import sys
import multiprocessing as mp

import typer
import starfile
from tabulate import tabulate

from GeoLlama import io
from GeoLlama import evaluate
from GeoLlama import calc_by_slice as CBS


app = typer.Typer()


def _check_user_input(
        path: typing.Optional[str],
        pixel_size: typing.Optional[float],
        num_cores: typing.Optional[int],
):
    """
    Check user inputs

    Args:
    path (str) : Path to tomogram (non-batch mode) or folder containing tomograms (batch mode)
    pixel_size (float) : Pixel size of original tomogram(s) in nm
    num_cores (int) : Number of cores designated for parallel processing
    """

    if path is None:
        raise ValueError("Data path (-p) must be given.")
    elif not os.path.isdir(Path(path)):
        raise NotADirectoryError("Given path must be a folder.")

    if pixel_size is None:
        raise ValueError("Pixel size (-s) must be given.")
    elif pixel_size <= 0:
        raise ValueError("Pixel size (-s) must be a positive number.")

    if not isinstance(num_cores, int):
        raise ValueError("num_cores (-np) must be an integer.")
    elif not 1 <= num_cores <= mp.cpu_count():
        raise ValueError(f"num_cores (-np) must be between 1 and # CPUs available ({mp.cpu_count()}.")


def _read_config(
        config_fname: str
) -> dict:
    """
    Parse GeoLlama config YAML file

    Args:
    config_fname (str) : Path to config file

    Returns:
    dict
    """
    import ruamel.yaml

    if not os.path.exists(config_fname):
        raise IOError(f"{config_fname} does not exist.")

    yaml = ruamel.yaml.YAML()
    config = yaml.load(Path(config_fname))

    # Check config file has correct dictionary keys
    keys = [
        "data_path", "pixel_size_nm", "binning", "autocontrast",
        "adaptive", "bandpass", "num_cores", "output_csv_path", "output_star_path"
    ]
    for key in keys:
        if not key in config:
            raise ValueError(f"{key} keyword missing in config file.")

    return config


@app.command()
def generate_config(
        output_path: Annotated[
            typing.Optional[str],
            typer.Option(help="Path to output YAML config file.")
        ] = "./config.yaml",
):
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


@app.command()
def main(
        input_config: Annotated[
            typing.Optional[str],
            typer.Option("-i", "--config",
                         help="Input configuration file. (NB. Overrides any parameters provided on command-line)"),
        ] = None,
        autocontrast: Annotated[
            bool,
            typer.Option(help="Apply autocontrast to slices prior to evaluation. Recommended.")
        ] = False,
        bandpass: Annotated[
            bool,
            typer.Option(help="Apply bandpass filter to tomograms prior to evaluation.")
        ] = False,
        user_path: Annotated[
            typing.Optional[str],
            typer.Option("-p", "--path",
                         help="Path to folder holding all tomograms. (NB. Direct path to individual tomogram file not supported.)"),
        ] = None,
        pixel_size: Annotated[
            typing.Optional[float],
            typer.Option(
                "-s", "--pixel_size",
                help="Tomogram pixel size in nm."),
        ] = None,
        binning: Annotated[
            int,
            typer.Option(
            "-b", "--bin",
            help="Internal binning factor for tomogram evaluation. Recommended target x-y dimensions from (256, 256) to (512, 512). E.g. if input tomogram has shape (2048, 2048, 2048), use -b 4 or -b 8."),
        ] = 1,
        cpu: Annotated[
            int,
            typer.Option(
                "-np", "--num_proc",
                help="Number of CPUs used."),
        ] = 1,
        out_csv: Annotated[
            typing.Optional[str],
            typer.Option(
                "-oc", "--csv",
                help="Output path for CSV file."),
        ] = None,
        out_star: Annotated[
            typing.Optional[str],
            typer.Option(
                "-os", "--star",
                help="Output path for STAR file."),
        ] = None,
):
    """
    Main API for running GeoLlama

    Args:
    input_config (str) : Path to input config file
    autocontrast (bool) : Apply autocontrast to slices prior to evaluation
    bandpass (bool) : Apply bandpass filter to tomograms prior to evaluation
    user_path (str) : Path to folder holding all tomograms in batch mode
    pixel_size (float) : Tomogram pixel size in nm
    binning (int) : Binning factor for tomogram evaluation
    cpu (int) : Number of CPUs used
    out_csv (str) : Output path for CSV file
    out_star (str) : Output path for STAR file
    """

    if input_config is not None:
        config = _read_config(input_config)

        # Convert config dictionary keys to internal variables
        user_path = config['data_path']
        pixel_size = config['pixel_size_nm']
        binning = config['binning']
        autocontrast = config['autocontrast']
        adaptive = config['adaptive']
        bandpass = config['bandpass']
        cpu = config['num_cores']
        out_csv = config['output_csv_path']
        out_star = config['output_star_path']

    _check_user_input(
        path=user_path,
        pixel_size=pixel_size,
        num_cores=cpu
    )

    if not Path("./surface_models").is_dir():
        Path("surface_models").mkdir()

    filelist = evaluate.find_files(path=user_path)
    raw_df, show_df = evaluate.eval_batch(
        filelist=filelist,
        pixel_size=pixel_size,
        binning=binning,
        cpu=cpu,
        bandpass=bandpass,
        autocontrast=autocontrast,
    )
    print(tabulate(show_df,
                   headers="keys",
                   tablefmt="pretty",
    ))

    # Print overall statistics
    thickness_mean_of_mean = raw_df['Mean_thickness_nm'].mean()
    thickness_std_of_mean = raw_df['Mean_thickness_nm'].std()

    xtilt_mean_of_mean = raw_df['Mean_X-tilt_degs'].mean()
    xtilt_std_of_mean = raw_df['Mean_X-tilt_degs'].std()

    print(f"Mean/std of thickness across datasets = {thickness_mean_of_mean:.2f} +/- {thickness_std_of_mean:.2f} nm")
    print(f"Mean/std of xtilt across datasets = {xtilt_mean_of_mean:.2f} +/- {xtilt_std_of_mean:.2f} degs")

    if out_csv is not None:
        raw_df.to_csv(out_csv, index=False)
    if out_star is not None:
        starfile.write(raw_df, out_star, overwrite=True)
