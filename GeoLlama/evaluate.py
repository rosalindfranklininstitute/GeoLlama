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

#############################################
## Module             : GeoLlama.evaluate  ##
## Created            : Neville Yee        ##
## Date created       : 05-Oct-2023        ##
## Date last modified : 03-May-2024        ##
#############################################


from pathlib import Path
import typing
import logging
from rich.logging import RichHandler

import numpy as np
from scipy.stats import sem
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

from GeoLlama.prog_bar import (prog_bar, clear_tasks)
from GeoLlama import io
from GeoLlama import calc_by_slice as CBS
from GeoLlama import objects

logging.basicConfig(level=logging.INFO,
                    format="%(message)s",
                    datefmt="%d-%b-%y %H:%M:%S",
                    handlers=[RichHandler()]
)

mpl.use("Agg")


def find_files(path: str) -> list:
    """
    Function to find appropriate files from given path

    Args:
    path (str) : Path to folder holding tomograms

    Returns:
    list
    """

    criterion = Path(path).glob("**/*.mrc")
    filelist = sorted([x for x in criterion if x.is_file()])

    return filelist


def save_figure(surface_info, save_path, binning):
    """
    Export data and interpolated surfaces as PNG file

    Args:
    surface_info (tuple) : Outputs (tuple) of GeoLlama evaluation module
    save_path (str) : Path to PNG file being exported
    binning (int) : Internal binning factor of tomogram
    """

    xx_top, yy_top, surface_top, xx_bottom, yy_bottom, surface_bottom, _, _ = surface_info

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                           figsize=(10, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot_surface(
        surface_top*binning,
        xx_top*binning,
        yy_top*binning,
        rstride=5, cstride=5)
    ax.plot_surface(
        surface_bottom*binning,
        xx_bottom*binning,
        yy_bottom*binning,
        rstride=5, cstride=5, color="r")
    ax.view_init(elev=0, azim=80)

    plt.savefig(save_path)
    plt.close()


def save_text_model(surface_info, save_path, binning):
    """
    Export data and interpolated model as text file for conversion and visualisation in IMOD.

    Args:
    surface_info (tuple) : Outputs (tuple) of GeoLlama evaluation module
    save_path (str) : Path to text file being exported
    binning (int) : Internal binning factor of tomogram
    """
    xx_top, yy_top, surface_top, xx_bottom, yy_bottom, surface_bottom, model_top, model_bottom = surface_info

    contour_top = np.tile(range(1, len(model_top)//2+1), 2)[:, np.newaxis] #np.full((len(model_top),1), 1, dtype=int)
    full_list_top = np.hstack(
        (contour_top, model_top*binning),
        dtype=object
    )

    contour_bottom = np.tile(range(len(model_top)//2+1, len(model_bottom)+1), 2)[:, np.newaxis] #np.full((len(model_bottom),1), 2, dtype=int)
    full_list_bottom = np.hstack(
        (contour_bottom, model_bottom*binning),
        dtype=object
    )

    full_contours = np.vstack((full_list_top, full_list_bottom), dtype=object)
    full_contours[:, 2] += 0.5

    np.savetxt(save_path, full_contours, fmt="%4d %.2f %.2f %.2f")


def eval_single(
        fname: str,
        params: objects.Config
):
    """
    Evaluate geometry of a single tomogram given source file

    Args:
    fname (str) : Path to tomogram file
    params (Config) : Config object holding all parameters

    Returns:
    ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
    """

    tomo, binned_pixel_size = io.read_mrc(
        fname=fname,
        px_size_nm=params.pixel_size_nm,
        downscale=params.binning
    )

    yz_stats, xz_stats, yz_mean, xz_mean, yz_sem, xz_sem, surfaces = CBS.evaluate_full_lamella(
        volume=CBS.filter_bandpass(tomo) if params.bandpass else tomo,
        pixel_size_nm=binned_pixel_size,
        cpu=params.num_cores,
        autocontrast=params.autocontrast,
    )

    # Adaptive mode
    if params.adaptive:
        anomalous = (yz_sem[1] > 20 or yz_sem[2] > 5)
        if anomalous:
            logging.info(f"Adaptive mode triggered for {fname.name}. \nSEM of thickness={yz_sem[1]:.3f}, SEM of xtilt={yz_sem[2]:3f}.")
            yz_stats, xz_stats, yz_mean, xz_mean, yz_sem, xz_sem, surfaces = CBS.evaluate_full_lamella(
                volume=CBS.filter_bandpass(tomo) if params.bandpass else tomo,
                pixel_size_nm=binned_pixel_size,
                cpu=params.num_cores,
                autocontrast=params.autocontrast,
                step_pct=1.25,
            )

    save_figure(surface_info=surfaces,
                save_path=f"./surface_models/{fname.stem}.png",
                binning=params.binning
    )
    save_text_model(surface_info=surfaces,
                    save_path=f"./surface_models/{fname.stem}.txt",
                    binning=params.binning
    )

    return (yz_stats, xz_stats, yz_mean, xz_mean, yz_sem, xz_sem, surfaces)


def eval_batch(
        filelist: list,
        params: objects.Config
) -> (pd.DataFrame, pd.DataFrame):
    """
    Evaluate geometry of tomograms given path to folder containing tomograms, then output statistics as pandas DataFrames.

    Args:
    filelist (list) : List containing paths to tomograms
    params (Config) : Config object holding all parameters

    Returns:
    DataFrame, DataFrame
    """

    thickness_mean_list = []
    xtilt_mean_list = []
    ytilt_mean_list = []

    thickness_sem_list = []
    xtilt_sem_list = []
    ytilt_sem_list = []

    thickness_list = []
    xtilt_list = []
    ytilt_list = []

    with prog_bar as p:
        clear_tasks(p)
        for tomo in p.track(filelist, total=len(filelist)):
            _, _, yz_mean, xz_mean, yz_sem, xz_sem, _ = eval_single(
                fname=tomo,
                params=params
            )

            thickness_mean_list.append(yz_mean[1])
            xtilt_mean_list.append(yz_mean[2])
            ytilt_mean_list.append(xz_mean[2])

            thickness_sem_list.append(yz_sem[1])
            xtilt_sem_list.append(yz_sem[2])
            ytilt_sem_list.append(xz_sem[2])

    for idx, _ in enumerate(filelist):
        thickness_list.append(f"{thickness_mean_list[idx]:.2f} +/- {thickness_sem_list[idx]:.2f}")
        xtilt_list.append(f"{xtilt_mean_list[idx]:.2f} +/- {xtilt_sem_list[idx]:.2f}")
        ytilt_list.append(f"{ytilt_mean_list[idx]:.2f} +/- {ytilt_sem_list[idx]:.2f}")

    # Detect anomalies
    xtilt_mean_of_mean = np.array(xtilt_mean_list).mean()
    xtilt_mean_sem = sem(np.array(xtilt_mean_list))

    thick_anomaly = np.array(thickness_sem_list) > 20
    xtilt_anomaly = np.array(xtilt_sem_list) > 5

    raw_data = pd.DataFrame(
        {"filename": [f.name for f in filelist],
         "Mean_thickness_nm": thickness_mean_list,
         "Thickness_SEM_nm": thickness_sem_list,
         "Mean_X-tilt_degs": xtilt_mean_list,
         "X-tilt_SEM_degs": xtilt_sem_list,
         "Mean_Y-tilt_degs": ytilt_mean_list,
         "Y-tilt_SEM_degs": ytilt_sem_list,
         "thickness_anomaly": thick_anomaly,
         "xtilt_anomaly": xtilt_anomaly,
        }
    )

    show_data = pd.DataFrame(
        {"filename": [f.name for f in filelist],
         "Thickness (nm)": thickness_list,
         "X-tilt (degs)": xtilt_list,
         "Y-tilt (degs)": ytilt_list,
         "thickness_anomaly": thick_anomaly,
         "xtilt_anomaly": xtilt_anomaly,
        }
    )

    return (raw_data, show_data)
