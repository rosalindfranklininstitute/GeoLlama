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

##################################################
## Module             : GeoLlama.calc_by_slice  ##
## Created            : Neville Yee             ##
## Date created       : 02-Oct-2023             ##
## Date last modified : 09-Oct-2023             ##
##################################################

from functools import partial
from pprint import pprint
import multiprocessing as mp

import numpy as np
import numpy.typing as npt

from skimage.filters import threshold_otsu as otsu
from skimage.filters import gaussian
from skimage.filters import difference_of_gaussians as DoG

from skimage.exposure import equalize_adapthist as CLAHE

from sklearn.cluster import (DBSCAN, OPTICS)
from sklearn.decomposition import PCA

from scipy.stats import mode, chi2


def enhance_CLAHE(volume: npt.NDArray[any],
                  clip_limit: float=0.5,
) -> npt.NDArray[any]:
    """
    Enhances image volume contrast using a scikit-image's CLAHE implementation

    Args:
    volume (ndarray) : input 3d image

    Returns:
    ndarray
    """
    volume_norm = (volume - volume.min()) / volume.ptp()
    clahe = CLAHE(volume_norm,
                  clip_limit=clip_limit
    )

    return clahe


def filter_bandpass(image: npt.NDArray[any],
) -> npt.NDArray[any]:
    """
    Bandpass filter for image

    Args:
    image (ndarray) : input image

    Returns:
    ndarray
    """
    filtered_image = DoG(image, min(image.shape)*0.01, max(image.shape)*0.05)

    return filtered_image


def create_slice_views(volume: npt.NDArray[any],
                       coords: list,
                       std_window: int=15,
                       gaussian_sigma: int=None,
                       use_bandpass: bool=True,
) -> (npt.NDArray[any], npt.NDArray[any], npt.NDArray[any]):

    if use_bandpass:
        volume = filter_bandpass(volume)

    std_half = std_window // 2
    z_range = (max(0, coords[0]-std_half), min(coords[0]+std_window-std_half, volume.shape[0]-1))
    x_range = (max(0, coords[1]-std_half), min(coords[1]+std_window-std_half, volume.shape[1]-1))
    y_range = (max(0, coords[2]-std_half), min(coords[2]+std_window-std_half, volume.shape[2]-1))

    # view_xy = np.std(volume[z_range[0]: z_range[1], :, :], axis=0)
    view_zy = np.std(volume[:, x_range[0]: x_range[1], :], axis=1)
    view_zx = np.std(volume[:, :, y_range[0]: y_range[1]], axis=2)

    if gaussian_sigma is not None:
        # view_xy = gaussian(view_xy, sigma=gaussian_sigma)
        view_zy = gaussian(view_zy, sigma=gaussian_sigma)
        view_zx = gaussian(view_zx, sigma=gaussian_sigma)

    return (view_zy, view_zx)


def evaluate_slice(slice_coords: list,
                   volume: npt.NDArray[any],
                   pixel_size_nm: float,
                   use_view: str="YZ",
                   pt_thres: int=100,
) -> (float, float, float):

    view_zy, view_zx = create_slice_views(
        volume=volume,
        coords=slice_coords,
        std_window=15,
        gaussian_sigma=3
    )

    if use_view=="YZ":
        view = view_zy
    elif use_view=="XZ":
        view = view_zx

    # Step 1: Greyvalue thresholding with Otsu method
    thres = otsu(view**2) * 1.25
    mask_s1 = np.argwhere(view**2 >= thres)

    # Skip slice if no pixels masked
    if len(mask_s1) < pt_thres:
        return (0,)*4
    centroid_s1 = mask_s1.mean(axis=0)

    # Step 2: Use Mahalanobis distance to remove potential outliers
    covar_inv = np.linalg.inv(np.cov(mask_s1.T))
    pts_deviation = mask_s1 - centroid_s1
    maha_dist = np.dot(np.dot(pts_deviation, covar_inv), pts_deviation.T).diagonal()
    p_val = 1 - chi2.cdf(np.sqrt(maha_dist), 1)
    mask_s2 = np.delete(mask_s1, np.argwhere(p_val<0.08), axis=0)

    # Skip slice if no pixels masked
    if len(mask_s2) < pt_thres:
        return (0,)*4

    # Step 3: Use distance-based clustering to find sample region
    clusters = DBSCAN(eps=15, min_samples=15).fit_predict(mask_s2)
    mask_s3_args = np.argwhere(clusters==mode(clusters, keepdims=True).mode)
    mask_s3 = mask_s2[mask_s3_args.flatten()]

    # Skip slice if no pixels masked
    if len(mask_s3) < pt_thres:
        return (0,)*4
    centroid_s3 = mask_s3.mean(axis=0)

    # Step 4: Use PCA to find best rectangle fits
    pca = PCA(n_components=2)
    pca.fit(mask_s3)
    eigenvecs = pca.components_
    eigenvals = np.sqrt(pca.explained_variance_)
    rectangle_dims = eigenvals * 4      # 4 times SD to cover nearly all points

    if eigenvecs[1,0] < 0:
        eigenvecs[1] *= -1
    angle = np.rad2deg(np.arctan2(-eigenvecs[np.argmin(eigenvals), 1],
                                  eigenvecs[np.argmin(eigenvals), 0]))

    slice_breadth, slice_thickness = rectangle_dims * pixel_size_nm
    num_points = len(mask_s3)

    return (slice_breadth, slice_thickness, angle, num_points)


def evaluate_full_lamella(volume, pixel_size_nm, cpu=1, clip_limit=None):
    if clip_limit is not None:
        volume = enhance_CLAHE(volume,
                               clip_limit=clip_limit,
        )

    x_slice_list = np.arange(int(volume.shape[1]*0.),
                             int(volume.shape[1]*1.),
                             int(volume.shape[1]*0.025))
    y_slice_list = np.arange(int(volume.shape[2]*0.),
                             int(volume.shape[2]*1.),
                             int(volume.shape[2]*0.025))

    # Evaluation along X axis (YZ-slices)
    x_coords = np.empty((len(x_slice_list), 3), dtype=int)
    x_coords[:, 0] = volume.shape[0]//2
    x_coords[:, 1] = x_slice_list
    x_coords[:, 2] = volume.shape[2]//2
    with mp.Pool(cpu) as p:
        f = partial(evaluate_slice,
                    volume=volume,
                    pixel_size_nm=pixel_size_nm,
                    use_view="YZ")
        yz_full_stats = np.array(p.map(f, x_coords))

    yz_remove_empty = np.array([s for s in yz_full_stats if s[3]>0])
    yz_thres = otsu(yz_remove_empty[:, 3])
    yz_full_stats_refined = np.array([s for s in yz_remove_empty if s[3]>yz_thres])

    # Evaluation along Y axis (XZ-slices)
    y_coords = np.empty((len(y_slice_list), 3), dtype=int)
    y_coords[:, 0] = volume.shape[0]//2
    y_coords[:, 1] = volume.shape[2]//2
    y_coords[:, 2] = y_slice_list
    with mp.Pool(cpu) as p:
        f = partial(evaluate_slice,
                    volume=volume,
                    pixel_size_nm=pixel_size_nm,
                    use_view="XZ")
        xz_full_stats = np.array(p.map(f, y_coords))

    xz_remove_empty = np.array([s for s in xz_full_stats if s[3]>0])
    xz_thres = otsu(xz_remove_empty[:, 3])
    xz_full_stats_refined = np.array([s for s in xz_remove_empty if s[3]>xz_thres])

    # Stat aggregation
    yz_mean = yz_full_stats_refined.mean(axis=0)
    xz_mean = xz_full_stats_refined.mean(axis=0)
    yz_std = yz_full_stats_refined.std(axis=0)
    xz_std = xz_full_stats_refined.std(axis=0)

    return (yz_full_stats, xz_full_stats,
            yz_mean, xz_mean,
            yz_std, xz_std)
