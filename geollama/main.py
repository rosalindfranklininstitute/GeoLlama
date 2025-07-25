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
## Date last modified : 21-Jul-2025    ##
#########################################

import os
from datetime import datetime as dt
import logging
import re

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import typing
from typing_extensions import Annotated

import typer

from geollama import config
from geollama import evaluate
from geollama import report
from geollama import objects


VERSION = "1.1.0"


def callback():
    pass


app = typer.Typer(callback=callback)


@app.callback()
def callback(
    dev_mode: Annotated[
        bool,
        typer.Option(
            "-d",
            "--dev_mode",
            help=r"""Run GeoLlama in developer mode. If true, verbose error messages and tracebacks 
            (with full printout of local variables) will be enabled.""",
        ),
    ] = False,
):
    """
    Callback for triggering dev-mode in GeoLlama.

    Parameters
    ----------
    dev_mode : bool
        Run GeoLlama in developer mode. If true, verbose error messages and tracebacks
        (with full printout of local variables) will be enabled.
    """
    app.pretty_exceptions_show_locals = dev_mode


@app.command()
def generate_config(
    output_path: Annotated[
        typing.Optional[str], typer.Option(help="Path to output YAML config file.")
    ] = "./config.yaml",
):
    """
    Generate a default configuration YAML file.
    Only used as a CLI entrypoint: cf. config.generate_config() for business logics.

    Parameters
    ----------
    output_path : str, default="./config.yaml"
        Path to output YAML config file.

    Returns
    -------

    """
    config.generate_config(output_path)


@app.command()
def main(
    input_config: Annotated[
        typing.Optional[str],
        typer.Option(
            "-i",
            "--config",
            help="Input configuration file. (NB. Overrides any parameters provided on command-line)",
        ),
    ] = None,
    adaptive: Annotated[
        bool,
        typer.Option(help="Use adaptive sampling for slice evaluation. Recommended."),
    ] = True,
    autocontrast: Annotated[
        bool,
        typer.Option(
            help="Apply autocontrast to slices prior to evaluation. Recommended."
        ),
    ] = True,
    bandpass: Annotated[
        bool,
        typer.Option(help="Apply bandpass filter to tomograms prior to evaluation."),
    ] = False,
    data_path: Annotated[
        typing.Optional[str],
        typer.Option(
            "-p",
            "--data_path",
            help="Path to folder holding all tomograms. (NB. Direct path to individual tomogram file not supported.)",
        ),
    ] = None,
    pixel_size_nm: Annotated[
        typing.Optional[float],
        typer.Option("-s", "--pixel_size", help="Pixel size of input tomogram in nm."),
    ] = None,
    binning: Annotated[
        int,
        typer.Option(
            "-b",
            "--bin",
            help="""Additional binning factor for GeoLlama tomogram evaluation. 
            Overall tomogram binning factor for evaluation is the product of the reconstruction and GeoLlama (this parameter) binning factors. 
            Recommended overall binning factor is 16 or 32 -- e.g. if input tomogram is binned by 4 at reconstruction, 
            recommended parameter would be 4 or 8. Use 0 (default) for auto-binning.""",
        ),
    ] = 0,
    num_cores: Annotated[
        int,
        typer.Option("-np", "--num_cores", help="Number of CPUs used."),
    ] = 1,
    output_csv_path: Annotated[
        typing.Optional[str],
        typer.Option("-oc", "--csv", help="Output path for CSV file."),
    ] = None,
    output_star_path: Annotated[
        typing.Optional[str],
        typer.Option("-os", "--star", help="Output path for STAR file."),
    ] = None,
    output_mask: Annotated[
        bool,
        typer.Option(
            "-m",
            "--mask",
            help="Output volumetric binary masks. If true, masks will be saved in ./volume_masks/ with same filenames as input tomogram.",
        ),
    ] = False,
    thickness_lower_limit: Annotated[
        float,
        typer.Option(
            help="Lower limit of lamella thickness in nm (for feature extraction)."
        ),
    ] = 120,
    thickness_upper_limit: Annotated[
        float,
        typer.Option(
            help="Upper limit of lamella thickness in nm (for feature extraction)."
        ),
    ] = 300,
    thickness_std_limit: Annotated[
        float,
        typer.Option(
            help="Limit of lamella thickness standard deviation in nm (for feature extraction)."
        ),
    ] = 15,
    xtilt_std_limit: Annotated[
        float,
        typer.Option(
            help="Limit of lamella xtilt standard deviation in degrees (for feature extraction)."
        ),
    ] = 5,
    displacement_limit: Annotated[
        float,
        typer.Option(
            help="Limit of lamella centroid displacement in % (for feature extraction)."
        ),
    ] = 25,
    displacement_std_limit: Annotated[
        float,
        typer.Option(
            help="Limit of lamella centroid displacement standard deviation in % (for feature extraction)."
        ),
    ] = 5,
    printout: Annotated[
        bool,
        typer.Option(
            help="Print statistical output after evaluation. Recommended for standalone use of GeoLlama."
        ),
    ] = True,
    profiling: Annotated[
        bool,
        typer.Option(help="Run GeoLlama on profiling mode."),
    ] = False,
    report: Annotated[
        bool,
        typer.Option(
            "--generate-report",
            help="Automatically generate report at the end of calculations.",
        ),
    ] = True,
):
    """
    Main API for running GeoLlama.

    Parameters
    __________
    input_config : str, default=None
        Input configuration file. (NB. Overrides any parameters provided on command-line)
    adaptive : bool, default=True
        Use adaptive sampling for slice evaluation.
    autocontrast : bool, default=True
        Apply autocontrast to slices prior to evaluation.
    adaptive : bool, default=True
        Whether to use adaptive mode. (doubling sampling in second run if anomaly detected)
    bandpass : bool, default=False
        Apply bandpass filter to tomograms prior to evaluation.
    data_path : str, default=None
        Path to folder holding all tomograms in batch mode.
    pixel_size_nm : float, default=None
        Tomogram pixel size in nm.
    binning : int, default=0
        Additional binning factor for GeoLlama tomogram evaluation.
        Overall tomogram binning factor for evaluation is the product of the reconstruction and GeoLlama (this parameter) binning factors.
        Recommended overall binning factor is 16 or 32 -- e.g. if input tomogram is binned by 4 at reconstruction, recommended parameter would be 4 or 8.
        Use 0 for auto-binning.
    num_cores : int, default=1
        Number of CPUs used.
    output_csv_path : str, default=None
        Output path for CSV file.
    output_star_path : str, default=None
        Output path for STAR file.
    output_mask : bool, default=False
        Output volumetric binary masks. If true, masks will be saved in ./volume_masks/ with same filenames as input tomogram.
    thickness_lower_limit : float, default=120
        Lower limit of lamella thickness in nm. (for feature extraction)
    thickness_upper_limit : float, default=120
        Upper limit of lamella thickness in nm. (for feature extraction)
    thickness_std_limit : float, default=15
        Limit of lamella thickness standard deviation in nm. (for feature extraction)
    xtilt_std_limit : float, default=5
        Limit of lamella xtilt standard deviation in degrees. (for feature extraction)
    displacement_limit : float, default=25
        Limit of lamella centroid displacement in %. (for feature extraction)
    displacement_std_limit : float, default=5
        Limit of lamella centroid displacement standard deviation in %. (for feature extraction)
    printout : bool, default=True
        Print statistical output after evaluation.
    profiling : bool, default=False
        Run GeoLlama on profiling mode.
    report : bool, default=True
        Automatically generate report at the end of calculations.
    """

    logging.info("GeoLlama started.")
    # Record application start time
    start_time = dt.now()

    # Library loading
    import starfile
    from tabulate import tabulate
    import pandas as pd

    dir_invoke = os.getcwd()
    logging.info(f"Current directory: {dir_invoke}")
    if input_config is not None:
        config_dict = config.read_config(input_config)
        dir_work = Path(os.path.abspath(input_config)).parents[0]
        logging.info(f"Changing working directory to: {dir_work}")
        os.chdir(dir_work)

    else:
        if data_path is None:
            raise ValueError(
                "Data path (-p) must be given if config file is not provided."
            )
        if pixel_size_nm is None:
            raise ValueError(
                "Pixel size (-s) must be given if config file is not provided."
            )
        config_dict = dict(
            autocontrast=autocontrast,
            adaptive=adaptive,
            bandpass=bandpass,
            data_path=data_path,
            pixel_size_nm=pixel_size_nm,
            binning=binning,
            num_cores=num_cores,
            output_csv_path=output_csv_path,
            output_star_path=output_star_path,
            output_mask=output_mask,
            generate_report=report,
            printout=printout,
            thickness_lower_limit=thickness_lower_limit,
            thickness_upper_limit=thickness_upper_limit,
            thickness_std_limit=thickness_std_limit,
            xtilt_std_limit=xtilt_std_limit,
            displacement_limit=displacement_limit,
            displacement_std_limit=displacement_std_limit,
        )

    params = objects.Config(**config_dict)
    params.validate()
    logging.info("Configuration checks complete.")
    if params.binning == 0:
        logging.info("AUTOBIN: Automatic binning factor activated.")

    if not Path("./surface_models").is_dir():
        Path("surface_models").mkdir()

    if params.output_mask and not Path("./volume_masks").is_dir():
        Path("volume_masks").mkdir()

    if profiling:
        # Only import profiling related libraries when needed
        from pstats import SortKey, Stats
        from pyinstrument import Profiler

        with Profiler(interval=0.01) as profile:
            filelist = evaluate.find_files(path=params.data_path)
            raw_df, analytics_df, show_df, adaptive_count = evaluate.eval_batch(
                filelist=filelist, params=params
            )

            Stats(profile).sort_stats(SortKey.CUMULATIVE).print_stats(
                50, r"\((?!\_).*\)$"
            )
        profile.print()
    else:
        filelist = evaluate.find_files(path=params.data_path)
        raw_df, analytics_df, show_df, adaptive_count = evaluate.eval_batch(
            filelist=filelist, params=params
        )

    # Create bash file for IMOD point2model conversion
    model_filelist = Path("./surface_models/").glob("*.txt")
    text = f"""#!/bin/bash/

for file in {' '.join([f.stem for f in filelist])}
do
\tpoint2model -input ${{file}}.txt -output ${{file}}.mdl -open -thick 3
done
    """
    with open("./surface_models/p2m_convert.sh", "w") as f:
        f.write(text)

    logging.info("All evaluations finished. Preparing statistical outputs...")
    # Record process end time
    end_time = dt.now()

    # Aggregate run metadata
    metadata_df = pd.DataFrame(
        {
            "version": [VERSION],
            "data_source": [str(Path(params.data_path).resolve()) + "/"],
            "model_folder": [str(Path("./surface_models").resolve()) + "/"],
            "start_time": [start_time.astimezone().isoformat(timespec="seconds")],
            "end_time": [end_time.astimezone().isoformat(timespec="seconds")],
            "time_elapsed": [
                re.sub("\s", "", re.sub("day[s]*,", "D", str(end_time - start_time)))
            ],
            "adaptive_count": adaptive_count,
            "thickness_lower_limit": params.thickness_lower_limit,
            "thickness_upper_limit": params.thickness_upper_limit,
            "thickness_std_limit": params.thickness_std_limit,
            "xtilt_std_limit": params.xtilt_std_limit,
            "displacement_limit": params.displacement_limit,
            "displacement_std_limit": params.displacement_std_limit,
        }
    )

    # Print overall statistics
    thickness_mean_of_mean = raw_df["Mean_thickness_nm"].mean()
    thickness_std_of_mean = raw_df["Mean_thickness_nm"].std()

    xtilt_mean_of_mean = raw_df["Mean_X-tilt_degs"].mean()
    xtilt_std_of_mean = raw_df["Mean_X-tilt_degs"].std()

    xtilt_filtered = raw_df["Mean_X-tilt_degs"][
        ~analytics_df["Anom_xtilt_out_of_range"]
    ]
    xtilt_filtered_mean_of_mean = xtilt_filtered.mean()
    xtilt_filtered_std_of_mean = xtilt_filtered.std()

    filtered_count = len(raw_df) - len(xtilt_filtered)

    anomalous_count = analytics_df["Num_possible_anomalies"][
        analytics_df["Num_possible_anomalies"] >= 3
    ].count()

    if params.output_csv_path is not None:
        raw_df.to_csv(params.output_csv_path, index=False)
    if params.output_star_path is not None:
        starfile.write(
            {"metadata": metadata_df, "metrics": raw_df, "analytics": analytics_df},
            params.output_star_path,
            overwrite=True,
        )

    if printout:
        print(
            tabulate(
                show_df,
                headers="keys",
                tablefmt="pretty",
            )
        )
        print(
            f"Mean/std of thickness across datasets = {thickness_mean_of_mean:.2f} +/- {thickness_std_of_mean:.2f} nm"
        )
        print(
            f"Mean/std of xtilt across datasets (Full) = {xtilt_mean_of_mean:.2f} +/- {xtilt_std_of_mean:.2f} degs"
        )
        print(
            f"Mean/std of xtilt across datasets ({filtered_count} outlier(s) excl.) = {xtilt_filtered_mean_of_mean:.2f} +/- {xtilt_filtered_std_of_mean:.2f} degs"
        )
        print(
            f"# Datasets with 3+ potential anomalies detected = {anomalous_count} / {len(raw_df)}"
        )

        if xtilt_filtered_std_of_mean > 15:
            print(
                "\nWARNING: Post-filtering standard deviation of xtilt > 15 degrees. VISUAL INSPECTION OF DATASET RECOMMENDED."
            )

    logging.info("All GeoLlama tasks finished.")

    if params.generate_report:
        logging.info("Generating GeoLlama report...")
        generate_report(
            star_path=params.output_star_path,
            report_path="./GeoLlama_report.ipynb",
            html=True,
        )

    os.chdir(dir_invoke)


@app.command()
def generate_report(
    star_path: Annotated[
        str,
        typer.Argument(help="Path to GeoLlama STAR file for report generation."),
    ],
    report_path: Annotated[
        str,
        typer.Option(help="Target path to save report."),
    ] = "./GeoLlama_report.ipynb",
    html: Annotated[
        bool,
        typer.Option(help="Export report to HTML."),
    ] = True,
):
    """
    API for generating a detailed report using GeoLlama outputs.
    Only used as a CLI entrypoint: cf. report.generate_report() for business logics.

    Parameters
    ----------
    star_path : str
        Path to GeoLlama STAR file for report generation.
    report_path : str
        Target path to save report.
    html : bool
        Export report to HTML.
    """
    try:
        not Path(report_path).exists()
    except:
        logging.warning("Existing report with same name found and will be replaced.")

    report.generate_report(
        report_path=Path(report_path), star_path=Path(star_path), to_html=html
    )
