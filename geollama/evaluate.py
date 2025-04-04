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

import inspect

from dataclasses import dataclass
from pathlib import Path
import typing
import logging
import functools
from rich.logging import RichHandler
import multiprocessing as mp

import numpy as np
from scipy.stats import sem, t, skew
from scipy.spatial.transform import Rotation as R
from skimage.transform import downscale_local_mean as DSLM

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import mrcfile

from geollama.prog_bar import prog_bar, clear_tasks
from geollama import io
from geollama import calc_by_slice as CBS
from geollama import objects

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[RichHandler()],
)

mpl.use("Agg")


@dataclass
class Lamella:
    """
    Object encapsulating estimated specifications of lamella
    """

    centroid: list
    thickness: float
    breadth: float
    xtilt: float
    ytilt: float


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

    xx_top, yy_top, surface_top, xx_bottom, yy_bottom, surface_bottom, _, _ = (
        surface_info
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot_surface(
        xx_top * binning, yy_top * binning, surface_top * binning, rstride=5, cstride=5
    )
    ax.plot_surface(
        xx_bottom * binning,
        yy_bottom * binning,
        surface_bottom * binning,
        rstride=5,
        cstride=5,
        color="r",
    )
    ax.view_init(elev=0, azim=-80)

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
    (
        xx_top,
        yy_top,
        surface_top,
        xx_bottom,
        yy_bottom,
        surface_bottom,
        model_top,
        model_bottom,
    ) = surface_info

    contour_top = np.tile(range(1, len(model_top) // 2 + 1), 2)[:, np.newaxis]
    full_list_top = np.hstack((contour_top, model_top * binning), dtype=object)

    contour_bottom = np.tile(range(len(model_top) // 2 + 1, len(model_bottom) + 1), 2)[
        :, np.newaxis
    ]
    full_list_bottom = np.hstack((contour_bottom, model_bottom * binning), dtype=object)

    full_contours = np.vstack((full_list_top, full_list_bottom), dtype=object)
    full_contours[:, 2] += 0.5

    np.savetxt(save_path, full_contours, fmt="%4d %.2f %.2f %.2f")

    return full_contours[:, 1:]


def eval_single(fname: str, params: objects.Config) -> objects.Result:
    """
    Evaluate geometry of a single tomogram given source file

    Args:
    fname (str) : Path to tomogram file
    params (Config) : Config object holding all parameters

    Returns:
    objects.Result
    """

    tomo, binned_pixel_size, unbinned_shape, binning_factor, tomo_orig = io.read_mrc(
        fname=fname, params=params
    )

    yz_stats, xz_stats, yz_mean, xz_mean, yz_sem, xz_sem, surfaces = (
        CBS.evaluate_full_lamella(
            volume=CBS.filter_bandpass(tomo) if params.bandpass else tomo,
            pixel_size_nm=binned_pixel_size,
            cpu=params.num_cores,
            autocontrast=params.autocontrast,
        )
    )

    # Adaptive mode
    adaptive_triggered = False
    if params.adaptive:
        criteria = [yz_sem[0] > 5, yz_sem[2] > 20, yz_sem[3] > 5]
        if np.any(criteria):
            adaptive_triggered = True
            logging.info(
                f"Adaptive mode triggered for {fname.name}. \nthickness = {yz_mean[2]:>3.3f} +/- {yz_sem[2]:>3.3f} nm \nxtilt     = {yz_mean[3]:>3.3f} +/- {yz_sem[3]:>3.3f} degs \ndrift     = {yz_mean[0]:>3.3f} +/- {yz_sem[0]:>3.3f} %"
            )
            yz_stats, xz_stats, yz_mean, xz_mean, yz_sem, xz_sem, surfaces = (
                CBS.evaluate_full_lamella(
                    volume=CBS.filter_bandpass(tomo) if params.bandpass else tomo,
                    pixel_size_nm=binned_pixel_size,
                    cpu=(
                        params.num_cores * 2
                        if params.num_cores <= mp.cpu_count() // 3
                        else params.num_cores
                    ),
                    autocontrast=params.autocontrast,
                    step_pct=1.25,
                )
            )
            logging.info(
                f"Final stats after increased sampling: \nthickness = {yz_mean[2]:>3.3f} +/- {yz_sem[2]:>3.3f} nm \nxtilt     = {yz_mean[3]:>3.3f} +/- {yz_sem[3]:>3.3f} degs \ndrift     = {yz_mean[0]:>3.3f} +/- {yz_sem[0]:>3.3f} %\n"
            )

    contours = save_text_model(
        surface_info=surfaces,
        save_path=f"./surface_models/{fname.stem}.txt",
        binning=binning_factor,
    )

    # Temporarily change ALL axis order from ZXY to XYZ
    tomo_shape = np.roll(np.array(unbinned_shape), -1)
    lamella_centroid = np.moveaxis(contours, 0, -1).mean(axis=1)

    # Define Lamella as object (all length measurements in PIXELS!!)
    lamella = Lamella(
        centroid=lamella_centroid,
        breadth=max(tomo_shape),
        thickness=yz_mean[2] / params.pixel_size_nm,
        xtilt=yz_mean[3],
        ytilt=xz_mean[3],
    )

    # Output of 3D binary mask
    if params.output_mask:
        mask = np.moveaxis(get_intersection_mask(tomo_shape, lamella), -1, 0)
        with mrcfile.new(f"./volume_masks/{fname.stem}.mrc", overwrite=True) as f:
            f.set_data(mask.astype(np.int8))

        with mrcfile.new(
            f"./volume_masks/{fname.stem}_verify.mrc", overwrite=True
        ) as f:
            combined = mask * tomo_orig
            f.set_data(combined.astype(np.float32))

    top_surface, bottom_surface = get_statistical_surfaces(
        lamella_obj=lamella, num_points=100
    )

    save_figure(
        surface_info=surfaces,
        save_path=f"./surface_models/{fname.stem}.png",
        binning=binning_factor,
    )

    single_result = objects.Result(
        yz_stats=yz_stats,
        xz_stats=xz_stats,
        yz_mean=yz_mean,
        xz_mean=xz_mean,
        yz_sem=yz_sem,
        xz_sem=xz_sem,
        surfaces=surfaces,
        binning_factor=binning_factor,
        adaptive_triggered=adaptive_triggered,
    )
    return single_result


def _eval_generator(filelist_in: list, params: objects.Config):
    with prog_bar as p:
        clear_tasks(p)
        for tomo in p.track(filelist_in, total=len(filelist_in)):
            eval_result = eval_single(fname=tomo, params=params)
            yield eval_result


def eval_batch(filelist: list, params: objects.Config) -> (pd.DataFrame, pd.DataFrame):
    """
    Evaluate geometry of tomograms given path to folder containing tomograms, then output statistics as pandas DataFrames.

    Args:
    filelist (list) : List containing paths to tomograms
    params (Config) : Config object holding all parameters

    Returns:
    DataFrame, DataFrame
    """

    _results_list = list(_eval_generator(filelist, params))

    drift_mean_list = [obj.yz_mean[0] for obj in _results_list]
    thickness_mean_list = [obj.yz_mean[2] for obj in _results_list]
    xtilt_mean_list = [obj.yz_mean[3] for obj in _results_list]
    ytilt_mean_list = [obj.xz_mean[3] for obj in _results_list]

    drift_sem_list = [obj.yz_sem[0] for obj in _results_list]
    thickness_sem_list = [obj.yz_sem[2] for obj in _results_list]
    xtilt_sem_list = [obj.yz_sem[3] for obj in _results_list]
    ytilt_sem_list = [obj.xz_sem[3] for obj in _results_list]

    binning_list = [obj.binning_factor for obj in _results_list]
    adaptive_list = [obj.adaptive_triggered for obj in _results_list]

    drift_list, thickness_list, xtilt_list, ytilt_list = [], [], [], []
    for idx, _ in enumerate(filelist):
        drift_list.append(f"{drift_mean_list[idx]:.2f} +/- {drift_sem_list[idx]:.2f}")
        thickness_list.append(
            f"{thickness_mean_list[idx]:.2f} +/- {thickness_sem_list[idx]:.2f}"
        )
        xtilt_list.append(f"{xtilt_mean_list[idx]:.2f} +/- {xtilt_sem_list[idx]:.2f}")
        ytilt_list.append(f"{ytilt_mean_list[idx]:.2f} +/- {ytilt_sem_list[idx]:.2f}")

    # Calculate 90% CI of xtilt values
    xtilt_median = np.median(xtilt_array := np.array(xtilt_mean_list))
    xtilt_head_to_centre = (
        xtilt_median - xtilt_array.min()
        if skew(xtilt_array) > 0
        else xtilt_array.max() - xtilt_median
    )
    xtilt_symmetrise_limits = [
        xtilt_median - xtilt_head_to_centre,
        xtilt_median + xtilt_head_to_centre,
    ]
    xtilt_symmetrised = np.where(
        np.logical_and(
            xtilt_array >= xtilt_symmetrise_limits[0],
            xtilt_array <= xtilt_symmetrise_limits[1],
        )
    )[0]
    xtilt_thres = [
        np.median(xtilt_symmetrised) - 3 * np.std(xtilt_symmetrised),
        np.median(xtilt_symmetrised) + 3 * np.std(xtilt_symmetrised),
    ]

    # Detect potential anomalies
    anom_too_thin = np.array(thickness_mean_list) < params.thickness_lower_limit
    anom_too_thick = np.array(thickness_mean_list) >= params.thickness_upper_limit
    anom_thick_uncertain = np.array(thickness_sem_list) >= params.thickness_std_limit
    anom_xtilt_oor = np.logical_or(
        xtilt_array < max(xtilt_thres[0], xtilt_symmetrise_limits[0]),
        xtilt_array > min(xtilt_thres[1], xtilt_symmetrise_limits[1]),
    )
    anom_xtilt_uncertain = np.array(xtilt_sem_list) >= params.xtilt_std_limit
    anom_centroid_displaced = np.array(drift_mean_list) >= params.displacement_limit
    anom_wild_drift = np.array(drift_sem_list) > params.displacement_std_limit

    anom_collated = np.stack(
        (
            anom_too_thin,
            anom_too_thick,
            anom_thick_uncertain,
            anom_xtilt_oor,
            anom_xtilt_uncertain,
            anom_centroid_displaced,
            anom_wild_drift,
        ),
        axis=1,
    )
    num_anom_categories = anom_collated.sum(axis=1)

    raw_data = pd.DataFrame(
        {
            "filename": [f.name for f in filelist],
            "binning": binning_list,
            "Mean_thickness_nm": thickness_mean_list,
            "Thickness_SEM_nm": thickness_sem_list,
            "Mean_X-tilt_degs": xtilt_mean_list,
            "X-tilt_SEM_degs": xtilt_sem_list,
            "Mean_Y-tilt_degs": ytilt_mean_list,
            "Y-tilt_SEM_degs": ytilt_sem_list,
            "Mean_drift_perc": drift_mean_list,
            "Drift_SEM_perc": drift_sem_list,
        }
    )

    analytics_data = pd.DataFrame(
        {
            "filename": [f.name for f in filelist],
            "Anom_too_thin": anom_too_thin,
            "Anom_too_thick": anom_too_thick,
            "Anom_thick_uncertain": anom_thick_uncertain,
            "Anom_xtilt_out_of_range": anom_xtilt_oor,
            "Anom_xtilt_uncertain": anom_xtilt_uncertain,
            "Anom_centroid_displaced": anom_centroid_displaced,
            "Anom_wild_drift": anom_wild_drift,
            "Num_possible_anomalies": num_anom_categories,
        }
    )

    show_data = pd.DataFrame(
        {
            "filename": [f.name for f in filelist],
            "Thickness (nm)": thickness_list,
            "X-tilt (degs)": xtilt_list,
            "Y-tilt (degs)": ytilt_list,
            "Centroid drift (%)": drift_list,
            "Num_possible_anomalies": num_anom_categories,
        }
    )

    return (raw_data, analytics_data, show_data, np.sum(adaptive_list))


def get_lamella_orientations(lamella_obj: Lamella) -> (np.ndarray, np.ndarray):
    """
    Extracts the coordinates of the reference vertex of the lamella, and calculates the cell vector of the estimated lamella.

    Args:
    lamella_obj (Lamella) : input Lamella object including essential lamella information

    Returns:
    ndarray, ndarray
    """
    rotation_matrix = R.from_euler(
        "yx", [lamella_obj.xtilt, -lamella_obj.ytilt], degrees=True
    )
    lamella_cell_vectors = rotation_matrix.apply(
        np.diag(
            [
                lamella_obj.breadth,
                lamella_obj.breadth,
                lamella_obj.thickness,
            ]
        )
    )

    lamella_ref_vertex = lamella_obj.centroid - 0.5 * lamella_cell_vectors.sum(axis=0)

    return (
        lamella_ref_vertex.astype(np.float32),
        lamella_cell_vectors.astype(np.float32),
    )


def get_intersection_mask(
    tomo_shape: list,
    lamella_obj: Lamella,
) -> np.ndarray:
    """
    Calculate volumetric mask for region estimated to be within lamella.

    args:
    tomo_shape (list) : dimensions of the input tomogram
    lamella_obj (Lamella) : input Lamella object including essential lamella information

    Returns:
    ndarray
    """
    # Calculate full list of tomogram pixel coordinates
    X, Y, Z = np.meshgrid(
        np.arange(tomo_shape[0], dtype=np.int16),
        np.arange(tomo_shape[1], dtype=np.int16),
        np.arange(tomo_shape[2], dtype=np.int16),
    )
    tomo_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Delete intermediate arrays meshgrids to release memory
    del X, Y, Z

    ref_vertex, lamella_vects = get_lamella_orientations(lamella_obj)
    lamella_vects_norm2 = np.linalg.norm(lamella_vects, axis=1) ** 2

    object_vect = np.abs(
        np.inner(np.subtract(tomo_coords, ref_vertex), lamella_vects)
        / lamella_vects_norm2
        - 0.5
    )
    mask = np.all(object_vect <= 0.5, axis=1).reshape(tomo_shape)

    return mask


def get_statistical_surfaces(
    lamella_obj: Lamella,
    num_points: int = 100,
) -> (np.ndarray, np.ndarray):
    """
    Calculate the top and bottom surfaces of lamella using evaluated geometry. Analogous to CBS:interpolate_surfaces.

    Args:
    lamella_obj (Lamella) : input Lamella object including essential lamella information
    num_points (int) : number of points per dimension for interpolation

    Returns:
    ndarray, ndarray
    """
    _, cell_vecs = get_lamella_orientations(lamella_obj=lamella_obj)
    (b1, b2, t) = cell_vecs

    top_ref_pt = lamella_obj.centroid + 0.5 * (-b1 - b2 + t)
    bottom_ref_pt = lamella_obj.centroid - 0.5 * (b1 + b2 + t)

    gridpoints_linspace = np.linspace(0, 1, num_points)
    gp_mesh_1, gp_mesh_2 = np.meshgrid(gridpoints_linspace, gridpoints_linspace)

    b1_mesh = np.repeat(gp_mesh_1[..., np.newaxis], 3, axis=2) @ b1
    b2_mesh = np.repeat(gp_mesh_2[..., np.newaxis], 3, axis=2) @ b2
    t_mesh = b1_mesh + b2_mesh

    top_surface = np.dstack((b1_mesh, b2_mesh, t_mesh)) + top_ref_pt
    bottom_surface = np.dstack((b1_mesh, b2_mesh, t_mesh)) + bottom_ref_pt

    return top_surface.astype(np.float64), bottom_surface.astype(np.float64)
