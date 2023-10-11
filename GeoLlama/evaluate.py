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
## Date last modified : 05-Oct-2023        ##
#############################################


from pathlib import Path

import pandas as pd

from GeoLlama.prog_bar import (prog_bar, clear_tasks)
from GeoLlama import io
from GeoLlama import calc_by_slice as CBS


def find_files(path: str) -> list:
    """
    Some docstrings
    """

    criterion = Path(path).glob("**/*.mrc")
    filelist = sorted([x for x in criterion if x.is_file()])

    return filelist


def eval_single(
        fname: str,
        pixel_size: float,
        binning: int,
        cpu: int
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
        cpu=cpu,
    )

    return (yz_stats, xz_stats, yz_mean, xz_mean, yz_std, xz_std)


def eval_batch(
        filelist: list,
        pixel_size: float,
        binning: int,
        cpu: int,
):
    """
    Some docstring
    """

    thickness_mean_list = []
    xtilt_mean_list = []
    ytilt_mean_list = []

    thickness_std_list = []
    xtilt_std_list = []
    ytilt_std_list = []

    thickness_list = []
    xtilt_list = []
    ytilt_list = []

    with prog_bar as p:
        clear_tasks(p)
        for tomo in p.track(filelist, total=len(filelist)):
            print(tomo)
            _, _, yz_mean, xz_mean, yz_std, xz_std = eval_single(
                fname=tomo,
                pixel_size=pixel_size,
                binning=binning,
                cpu=cpu,
            )

            thickness_mean_list.append(yz_mean[1])
            xtilt_mean_list.append(yz_mean[2])
            ytilt_mean_list.append(xz_mean[2])

            thickness_std_list.append(yz_std[1])
            xtilt_std_list.append(yz_std[2])
            ytilt_std_list.append(xz_std[2])

    for idx, _ in enumerate(filelist):
        thickness_list.append(f"{thickness_mean_list[idx]:.2f} +/- {thickness_std_list[idx]:.2f}")
        xtilt_list.append(f"{xtilt_mean_list[idx]:.2f} +/- {xtilt_std_list[idx]:.2f}")
        ytilt_list.append(f"{ytilt_mean_list[idx]:.2f} +/- {ytilt_std_list[idx]:.2f}")

    data = pd.DataFrame(
        {"filename": [f.name for f in filelist],
         "Thickness (nm)": thickness_list,
         "X-tilt (degs)": xtilt_list,
         "Y-tilt (degs)": ytilt_list,
        }
    )

    return data
