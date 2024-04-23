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
## Date last modified : 19-Apr-2024    ##
#########################################

import os
from pprint import pprint
from pathlib import Path
import typing

import typer
import starfile
from tabulate import tabulate

from GeoLlama import io
from GeoLlama import evaluate
from GeoLlama import calc_by_slice as CBS


app = typer.Typer()


def _check_cli_input(path: typing.Optional[str],
                     pixel_size: typing.Optional[float]
):
    """
    Check user inputs for CLI version

    Args:
    path (str) : Path to tomogram (non-batch mode) or folder containing tomograms (batch mode)
    pixel_size (float) : Pixel size of original tomogram(s) in nm
    """

    if path is None:
        raise ValueError("Data path (-p) must be given.")
    if pixel_size is None:
        raise ValueError("Pixel size (-s) must be given.")

    assert(os.path.isdir(path)), \
        "The path given must be a folder in batch mode."


@app.command()
def main(
        autocontrast: bool = typer.Option(
            False,
            help="Apply autocontrast to slices prior to evaluation.",
        ),
        bandpass: bool = typer.Option(
            False,
            help="Apply bandpass filter to tomograms prior to evaluation.",
        ),
        user_path: typing.Optional[str] = typer.Option(
            None, "-p", "--path",
            help="Path to folder holding all tomograms in batch mode.",
        ),
        pixel_size: typing.Optional[float] = typer.Option(
            None, "-s", "--pixel_size",
            help="Tomogram pixel size in nm.",
        ),
        binning: int = typer.Option(
            1, "-b", "--bin",
            help="Binning factor for tomogram evaluation.",
            show_default=True,
        ),
        cpu: int = typer.Option(
            1, "-np", "--num_proc",
            help="Number of CPUs used.",
            show_default=True,
        ),
        out_csv: typing.Optional[str] = typer.Option(
            None, "-oc", "--csv",
            help="Output path for CSV file.",
            show_default=True,
        ),
        out_star: typing.Optional[str] = typer.Option(
            None, "-os", "--star",
            help="Output path for STAR file.",
            show_default=True,
        ),
):
    """
    Main API for running GeoLlama

    Args:
    autocontrast (bool) : Apply autocontrast to slices prior to evaluation
    bandpass (bool) : Apply bandpass filter to tomograms prior to evaluation
    user_path (str) : Path to folder holding all tomograms in batch mode
    pixel_size (float) : Tomogram pixel size in nm
    binning (int) : Binning factor for tomogram evaluation
    cpu (int) : Number of CPUs used
    out_csv (str) : Output path for CSV file
    out_star (str) : Output path for STAR file
    """

    _check_cli_input(
        batch=batch,
        path=user_path,
        pixel_size=pixel_size
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
        raw_df.to_csv(out_csv)
    if out_star is not None:
        starfile.write(raw_df, out_star, overwrite=True)
