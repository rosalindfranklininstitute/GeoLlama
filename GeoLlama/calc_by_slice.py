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

import itertools
from functools import partial
from pprint import pprint
import multiprocessing as mp

import numpy as np
import numpy.typing as npt

from skimage.filters import threshold_otsu as otsu
from skimage.filters import gaussian
from skimage.filters import difference_of_gaussians as DoG

from sklearn.cluster import (DBSCAN, OPTICS)
from sklearn.decomposition import PCA

from scipy.stats import mode, chi2
import scipy.interpolate as spin
from icecream import ic

import matplotlib.pyplot as plt


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


def autocontrast_slice(image: npt.NDArray[any],
                       target_mean: float=150,
                       target_std: float=40,
                       clip_cutoff: float=3,
) -> npt.NDArray[any]:
    """
    Auto-contrast image slice by clipping and histogram stretching

    Args:
    image (ndarray) : input image
    target_mean (float) : mean grayvalue of the output image
    target_std (float) : standard deviation of grayvalue of the output image
    clip_cutoff (float) : number of SDs to clip before histogram stretching

    Returns:
    ndarray
    """
    modified_z_score = (image - np.mean(image)) / (np.std(image) * clip_cutoff)
    z_clipped = np.clip(modified_z_score, -1, 1)
    z_clipped_std = np.std(z_clipped)

    img_corrected = z_clipped * target_std / z_clipped_std + target_mean

    return img_corrected


def create_slice_views(volume: npt.NDArray[any],
                       coords: list,
                       std_window: int=15,
                       gaussian_sigma: int=None,
) -> (npt.NDArray[any], npt.NDArray[any], npt.NDArray[any]):
    std_half = std_window // 2
    z_range = (max(0, coords[0]-std_half), min(coords[0]+std_window-std_half, volume.shape[0]-1))
    x_range = (max(0, coords[1]-std_half), min(coords[1]+std_window-std_half, volume.shape[1]-1))
    y_range = (max(0, coords[2]-std_half), min(coords[2]+std_window-std_half, volume.shape[2]-1))

    # view_xy = np.std(volume[z_range[0]: z_range[1], :, :], axis=0)
    view_zy = np.std(volume[:, x_range[0]: x_range[1], :], axis=1) * np.sum(volume[:, x_range[0]: x_range[1], :], axis=1)
    view_zx = np.std(volume[:, :, y_range[0]: y_range[1]], axis=2) * np.sum(volume[:, :, y_range[0]: y_range[1]], axis=2)

    if gaussian_sigma is not None:
        view_zy = gaussian(view_zy, sigma=gaussian_sigma)
        view_zx = gaussian(view_zx, sigma=gaussian_sigma)

    return (view_zy, view_zx)


def interpolate_surface(mesh_points: npt.NDArray[any],
                        n_points: int=100,
):
    """
    Interpolates surface using mesh points

    Args:
    mesh_points (ndarray) : input mesh points for reconstructing surface
    n_points (int) : number of points per dimension for interpolation

    Returns:
    ndarray
    """
    x_mesh = np.linspace(np.min(mesh_points[:,1]),
                         np.max(mesh_points[:,1]),
                         n_points)
    y_mesh = np.linspace(np.min(mesh_points[:,2]),
                         np.max(mesh_points[:,2]),
                         n_points)
    xx, yy = np.meshgrid(x_mesh, y_mesh)

    interp = spin.LinearNDInterpolator(mesh_points[:, 1:], mesh_points[:, 0])
    surface = interp(xx, yy)

    return xx, yy, surface


def contour_refine(contour_pts: npt.NDArray[any],
) -> npt.NDArray:
    """
    some docstring
    """
    contour_pts_mean = contour_pts.mean(axis=0)

    pca = PCA(n_components=1)
    pca.fit(contour_pts)
    dir_vec = pca.components_

    distance = np.linalg.norm(
        np.cross(contour_pts-contour_pts_mean, dir_vec),
        axis=1
    )

    # thres = distance.mean() + 3 * np.std(distance)
    # del_args = np.argwhere(distance > thres).flatten()

    dist_std_LOO = np.empty((len(contour_pts)))
    for i in range(len(contour_pts)):
        temp = np.delete(contour_pts, i, axis=0)
        dist_std_LOO[i] = np.std(temp)

    print(dist_std_LOO)

    out = np.delete(contour_pts, del_args, axis=0)

    return out, del_args


def generalised_theil_sen_fit(contour_pts: npt.NDArray[any],
) -> (npt.NDArray, npt.NDArray, float):
    """
    Linear fitting of 3D points using a generalised Theil-Sen algorithm

    Args:
    contour_pts (ndarray) : contour points in 3D space for fitting

    Returns:
    ndarray, ndarray, float
    """
    # Estimate gradient of fitted line
    point_pairs = np.array(list(itertools.combinations(contour_pts, 2)))
    grad_vectors = np.array([p[1]-p[0] for p in point_pairs])
    grad_median = np.median(grad_vectors, axis=0)
    grad_norm = grad_median / np.linalg.norm(grad_median)

    # Estimate offset of fitted line using median of point-to-line vectors
    ptl_vectors = contour_pts - np.dot(contour_pts, grad_norm)[..., np.newaxis] * grad_norm
    ptl_median = np.median(ptl_vectors, axis=0)

    # Final fit + mean-squared error calculation
    fitted_ptl_vectors = (contour_pts-ptl_median) - np.dot(contour_pts-ptl_median, grad_norm)[..., np.newaxis] * grad_norm
    mse = np.mean(np.linalg.norm(fitted_ptl_vectors, axis=0)**2)

    return (grad_norm, ptl_median, mse)


def leave_one_out(contour_pts: npt.NDArray[any],
                  thres: float=0.1,
) -> int | None:
    """
    Get  worst outlier point using leave-one-out (LOO) algorithm

    Args:
    contour_pts (ndarray) : Input ensemble for outlier detection
    thres (float) : Threshold MSE change for outlier detection

    Returns:
    int | None
    """
    mse_perc_change = np.empty(len(contour_pts))

    # Initial fit
    grad_0, passing_0, mse_0 = generalised_theil_sen_fit(contour_pts)

    # Leave-one-out algorithm
    for idx, points in enumerate(contour_pts):
        temp = np.delete(contour_pts, idx, axis=0)
        grad, passing, mse = generalised_theil_sen_fit(temp)
        mse_perc_change[idx] = (mse-mse_0) / mse_0

    # Determine whether worst point is an outlier using given threshold
    # If the point is an outlier, determine its index
    if mse_perc_change.min() < -thres:
        points_out = np.delete(contour_pts, np.argmin(mse_perc_change), axis=0)
        return np.argmin(mse_perc_change)

    return None


def refine_contour_LOO(contour_pts: npt.NDArray[any],
                       max_delete_perc: float=15.,
                       thres: float=0.1,
) -> npt.NDArray:
    """
    Iteratively remove outliers using LOO algorithm

    Args:
    contour_pts (ndarray) : Input ensemble of contour points for refinement
    max_delete_percentage (float) : Maximum percentage of slices allowed to be removed
    thres (float) : Threshold MSE change for outlier detection

    Returns:
    np.ndarray
    """
    max_iter = int(len(contour_pts) * max_delete_perc * 0.01)
    slice_list = [i for i in range(len(contour_pts))]
    remove_list = []

    temp = np.copy(contour_pts)
    for curr_iter in range(max_iter):
        remove_idx = leave_one_out(temp, thres=thres)
        if remove_idx is None:
            return remove_list

        remove_list.append(slice_list.pop(remove_idx))
        temp = np.delete(temp, remove_idx, axis=0)

    return remove_list


def evaluate_slice(slice_coords: list,
                   volume: npt.NDArray[any],
                   pixel_size_nm: float,
                   use_view: str="YZ",
                   pt_thres: int=100,
                   autocontrast: bool=True,
) -> (float, float, float):

    view_zy, view_zx = create_slice_views(
        volume=volume,
        coords=slice_coords,
        std_window=5,
        gaussian_sigma=3
    )

    if use_view=="YZ":
        view = view_zy.T
    elif use_view=="XZ":
        view = view_zx.T

    if autocontrast:
        view = autocontrast_slice(view)

    # Step 1: Greyvalue thresholding with Otsu method
    thres = otsu(view**2) * 1.25
    mask_s1 = np.argwhere(view**2 >= thres)

    # Skip slice if no pixels masked
    # if len(mask_s1) < pt_thres:
    #     return
    # centroid_s1 = 0.5*(mask_s1.max(axis=0) + mask_s1.min(axis=0))
    centroid_s1 = mask_s1.mean(axis=0)

    # Step 2: Use Mahalanobis distance to remove potential outliers
    covar_inv = np.linalg.inv(np.cov(mask_s1.T))
    pts_deviation = mask_s1 - centroid_s1
    maha_dist = np.dot(np.dot(pts_deviation, covar_inv), pts_deviation.T).diagonal()
    p_val = 1 - chi2.cdf(np.sqrt(maha_dist), 1)
    mask_s2 = np.delete(mask_s1, np.argwhere(p_val<0.07), axis=0)

    # Skip slice if no pixels masked
    # if len(mask_s2) < pt_thres:
    #     return

    # Step 3: Use distance-based clustering to find sample region
    clusters = DBSCAN(eps=15, min_samples=15).fit_predict(mask_s2)
    mask_s3_args = np.argwhere(clusters==mode(clusters, keepdims=True).mode)
    mask_s3 = mask_s2[mask_s3_args.flatten()]

    # Skip slice if no pixels masked
    # if len(mask_s3) < pt_thres:
    #     return
    # centroid_s3 = 0.5*(mask_s3.max(axis=0) + mask_s3.min(axis=0))
    centroid_s3 = mask_s3.mean(axis=0)

    # Step 4: Use PCA to find best rectangle fits
    pca = PCA(n_components=2)
    pca.fit(mask_s3)
    eigenvecs = pca.components_
    eigenvals = np.sqrt(pca.explained_variance_)
    rectangle_dims = eigenvals * 2.5      # 3 times SD to cover nearly all points

    if eigenvecs[0,1] < 0:
        eigenvecs[0] *= -1
    if eigenvecs[1,0] < 0:
        eigenvecs[1] *= -1

    angle = np.rad2deg(np.arctan2(-eigenvecs[np.argmin(eigenvals), 1],
                                  eigenvecs[np.argmin(eigenvals), 0]))

    slice_breadth, slice_thickness = 2 * rectangle_dims * pixel_size_nm
    num_points = len(mask_s3)

    cell_vecs = eigenvecs * rectangle_dims.reshape((2, 1))

    # Centralise lamella centroid to middle of slice along long axis
    lamella_centre = centroid_s3

    extrema = np.array(view.shape) - 1
    ref_pt1 = lamella_centre + cell_vecs[1]
    ref_pt2 = lamella_centre - cell_vecs[1]

    top_pt1 = lamella_centre + cell_vecs[1] + cell_vecs[0]
    top_pt2 = lamella_centre + cell_vecs[1] - cell_vecs[0]
    bottom_pt1 = lamella_centre - cell_vecs[1] + cell_vecs[0]
    bottom_pt2 = lamella_centre - cell_vecs[1] - cell_vecs[0]

    if use_view == "YZ":
        surface_top_1 = np.array(
            [top_pt1[0],
             slice_coords[1],
             top_pt1[1]],
        )
        surface_top_2 = np.array(
            [top_pt2[0],
             slice_coords[1],
             top_pt2[1]],
        )
        surface_bottom_1 = np.array(
            [bottom_pt1[0],
             slice_coords[1],
             bottom_pt1[1]]
        )
        surface_bottom_2 = np.array(
            [bottom_pt2[0],
             slice_coords[1],
             bottom_pt2[1]],
        )

    if use_view == "XZ":
        surface_top_1 = np.array(
            [top_pt1[0],
             top_pt1[1],
             slice_coords[2],
            ]
        )
        surface_top_2 = np.array(
            [top_pt2[0],
             top_pt2[1],
             slice_coords[2],
            ],
        )
        surface_bottom_1 = np.array(
            [bottom_pt1[0],
             bottom_pt1[1],
             slice_coords[2],
            ]
        )
        surface_bottom_2 = np.array(
            [bottom_pt2[0],
             bottom_pt2[1],
             slice_coords[2],
            ],
        )
    return (slice_breadth, slice_thickness, angle, num_points,
            surface_top_1, surface_top_2,
            surface_bottom_1, surface_bottom_2)


def evaluate_full_lamella(volume,
                          pixel_size_nm,
                          autocontrast,
                          cpu=1,
):
    x_slice_list = np.arange(int(volume.shape[1]*0.2),
                             int(volume.shape[1]*0.8),
                             int(volume.shape[1]*0.025))
    y_slice_list = np.arange(int(volume.shape[2]*0.2),
                             int(volume.shape[2]*0.8),
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
                    autocontrast=autocontrast,
                    use_view="YZ"
        )
        yz_output = np.array(p.map(f, x_coords), dtype=object)

    yz_output = np.concatenate(yz_output).ravel().reshape((yz_output.size//8, 8))
    yz_full_stats = np.array(yz_output[:, :4], dtype=float)
    yz_surface_top_1 = np.concatenate(yz_output[:,4]).ravel().reshape(len(yz_output), 3)
    yz_surface_top_2 = np.concatenate(yz_output[:,5]).ravel().reshape(len(yz_output), 3)
    yz_surface_bottom_1 = np.concatenate(yz_output[:,6]).ravel().reshape(len(yz_output), 3)
    yz_surface_bottom_2 = np.concatenate(yz_output[:,7]).ravel().reshape(len(yz_output), 3)

    yz_surface_top = np.concatenate((yz_surface_top_1, yz_surface_top_2))
    yz_surface_bottom = np.concatenate((yz_surface_bottom_1, yz_surface_bottom_2))

    # Refine measurements
    t1 = refine_contour_LOO(yz_surface_top_1)
    t2 = refine_contour_LOO(yz_surface_top_2)
    b1 = refine_contour_LOO(yz_surface_bottom_1)
    b2 = refine_contour_LOO(yz_surface_bottom_2)

    removed_slices = list(set(t1 + t2 + b1 + b2))
    yz_full_stats_refined = np.delete(yz_full_stats, removed_slices, axis=0)
    yz_top_contour_1 = np.delete(yz_surface_top_1, removed_slices, axis=0)
    yz_top_contour_2 = np.delete(yz_surface_top_2, removed_slices, axis=0)
    yz_bottom_contour_1 = np.delete(yz_surface_bottom_1, removed_slices, axis=0)
    yz_bottom_contour_2 = np.delete(yz_surface_bottom_2, removed_slices, axis=0)

    # yz_remove_empty = np.array([s for s in yz_full_stats if s[3]>0])
    # yz_thres = otsu(yz_remove_empty[:, 3])
    # yz_full_stats_refined = np.array([s for s in yz_remove_empty if s[3]>yz_thres])

    # Evaluation along Y axis (XZ-slices)
    y_coords = np.empty((len(y_slice_list), 3), dtype=int)
    y_coords[:, 0] = volume.shape[0]//2
    y_coords[:, 1] = volume.shape[1]//2
    y_coords[:, 2] = y_slice_list
    with mp.Pool(cpu) as p:
        f = partial(evaluate_slice,
                    volume=volume,
                    pixel_size_nm=pixel_size_nm,
                    autocontrast=autocontrast,
                    use_view="XZ"
        )
        xz_output = np.array(p.map(f, y_coords), dtype=object)

    xz_full_stats = np.array(xz_output[:, :4], dtype=float)

    xz_remove_empty = np.array([s for s in xz_full_stats if s[3]>0])
    xz_thres = otsu(xz_remove_empty[:, 3])
    xz_full_stats_refined = np.array([s for s in xz_remove_empty if s[3]>xz_thres])

    # Stat aggregation
    yz_mean = yz_full_stats_refined.mean(axis=0)
    xz_mean = xz_full_stats_refined.mean(axis=0)
    yz_std = yz_full_stats_refined.std(axis=0)
    xz_std = xz_full_stats_refined.std(axis=0)

    xx_top, yy_top, surface_interp_top = interpolate_surface(
        np.vstack((yz_top_contour_1, yz_top_contour_2))
    )

    xx_bottom, yy_bottom, surface_interp_bottom = interpolate_surface(
        np.vstack((yz_bottom_contour_1, yz_bottom_contour_2))
    )

    surfaces = (xx_top, yy_top, surface_interp_top,
                xx_bottom, yy_bottom, surface_interp_bottom,
                yz_surface_top, yz_surface_bottom
    )

    return (yz_full_stats, xz_full_stats,
            yz_mean, xz_mean,
            yz_std, xz_std,
            surfaces
    )
