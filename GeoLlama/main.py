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
## Date last modified : 05-Oct-2023    ##
#########################################

import os
from pathlib import Path
import typing

import typer

from GeoLlama import io
from GeoLlama import calc_by_slice as CBS


app = typer.Typer()


def GL_single(
        fname: str = typer.Argument(
            help="File path to tomogram (MRC)."
        ),
        pixel_size: float = typer.Argument(
            help="Pixel size (nm) of tomogram."
        ),
        binning: int = typer.Option(
            1, "-b", "--bin",
            help="Binning factor for tomogram evaluation.",
            show_default=True,
            min=1,
        ),
):
    """
    Some docstring
    """

    tomo, pixel_size = io.read_mrc(
        fname=fname,
        px_size_nm=pixel_size,
        downscale=binning
    )

    yz_stats, xz_stats, yz_mean, xz_mean, yz_std, xz_std = CBS.evaluate_full_lamella(
        volume=tomo,
        pixel_size_nm=pixel_size,
    )

    return (yz_stats, xz_stats, yz_mean, xz_mean, yz_std, xz_std)


def _check_cli_input(batch: bool,
                     path: typing.Optional[str],
                     pixel_size: typing.Optional[float]
):
    """
    Check user inputs for CLI version
    """

    if path is None:
        raise ValueError("Data path (-p) must be given.")
    if pixel_size is None:
        raise ValueError("Pixel size (-s) must be given.")

    if batch:
        assert(os.path.isdir(path)), \
            "The path given must be a folder in batch mode."
    else:
        assert(os.path.isfile(path)), \
            "The path given must be a file in non-batch mode."
        assert(Path(path).suffix==".mrc"), \
            "The image given must be in MRC (.mrc) format."


@app.command()
def main(
        gui: bool = typer.Option(
            True,
            help="Use GUI version of GeoLlama.",
        ),
        batch: bool = typer.Option(
            False,
            help="Batch mode of GeoLlama. Finds and evaluates all MRC tomograms in given folder.",
        ),
        user_path: typing.Optional[str] = typer.Option(
            None, "-p", "--path",
            help="Path to folder holding all tomograms in batch mode. Path to single image for evaluation otherwise.",
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
):
    """
    SOME DOCSTRING
    """

    if gui:
        pass

    else:
        _check_cli_input(
            batch=batch,
            path=user_path,
            pixel_size=pixel_size
        )

        if batch:
            pass

        else:
            results = GL_single(
                fname=user_path,
                pixel_size=pixel_size,
                binning=binning)

            yz_stats, xz_stats, yz_mean, xz_mean, yz_std, xz_std = results
