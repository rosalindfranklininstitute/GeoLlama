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
## Date last modified : 25-Jan-2024    ##
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
            False,
            help="Use GUI version of GeoLlama.",
        ),
        batch: bool = typer.Option(
            True,
            help="Batch mode of GeoLlama. Finds and evaluates all MRC tomograms in given folder.",
        ),
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

        if not Path("./surface_models").is_dir():
            Path("surface_models").mkdir()

        if batch:
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

            if out_csv is not None:
                raw_df.to_csv(out_csv)
            if out_star is not None:
                starfile.write(raw_df, out_star, overwrite=True)

        else:
            results = evaluate.eval_single(
                fname=user_path,
                pixel_size=pixel_size,
                binning=binning,
                cpu=cpu,
                autocontrast=autocontrast
            )

            yz_stats, xz_stats, yz_mean, xz_mean, yz_std, xz_std = results
