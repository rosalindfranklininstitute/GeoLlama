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
## Module             : GeoLlama.report  ##
## Created            : Neville Yee      ##
## Date created       : 16-Sep-2024      ##
## Date last modified : 17-Sep-2024      ##
###########################################

import logging

import warnings
import subprocess
from pathlib import Path
import pkg_resources

import json

import papermill as pm


def read_ipynb(
        ipynb_path: Path
) -> dict:
    """
    Reads in a Jupyter notebook and returns the underlying JSON as a dict.

    Args:
    ipynb_path (Path) : Path to Jupyter notebook template

    Returns:
    dict
    """
    nb = pkg_resources.resource_filename("GeoLlama.templates", ipynb_path)
    with open(nb, "r") as f:
        return json.load(f)


def write_ipynb(
        ipynb_path: Path,
        nb_data: dict,
):
    """
    Write out Jupyter notebook with given data / metadata

    Args:
    ipynb_path (Path) : Target path for saving notebook
    nb_data (dict) : Dictionary containing JSON metadata for conversion into report

    Returns:
    None
    """
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f)


def generate_report(
        report_path: Path,
        star_path: Path,
        to_html: bool=False
):
    """
    Generates temporary Jupyter notebook and converts it using papermill

    Args:
    report_path (Path) : Target path for saving completed report
    star_path (Path) : Path to GeoLlama-STAR file for report generation
    to_html (bool) : Whether to convert the executed report into HTML format
    """
    if report_path.exists():
        warnings.warn(f"{report_path} exists. Old report will be rewritten.")
    if report_path.suffix != ".ipynb":
        warnings.warn(f"Output to {report_path.suffix} format is not currently supported. Report will be saved as {report_path.stem}.ipynb.")
        report_path = Path(f"{report_path.parent}/{report_path.stem}.ipynb")

    # Read in report template and write out temporary report for execution
    report = read_ipynb("report_template.ipynb")
    write_ipynb(f"{report_path.parent}/{report_path.stem}_tmp.ipynb", report)

    # Execute temporary report
    pm.execute_notebook(
        f"{report_path.parent}/{report_path.stem}_tmp.ipynb",
        report_path,
        parameters={
            "starfile_path": str(star_path),
        },
        kernel="python3"
    )

    Path(f"{report_path.parent}/{report_path.stem}_tmp.ipynb").unlink()

    # Convert report to HTML format
    if to_html:
        cmd = [
            "jupyter-nbconvert", "--to", "html",
            "--TemplateExporter.exclude_input=True",
            f"{report_path}"
        ]
        run_cmd = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="ascii",
            check=True
        )

        try:
            assert not run_cmd.stderr
        except:
            logging.critical(f"{run_cmd.stderr}")
