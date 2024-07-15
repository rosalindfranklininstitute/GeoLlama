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
## Date last modified : 25-Apr-2024             ##
##################################################

import itertools
from functools import partial
from pprint import pprint
import multiprocessing as mp

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view as swv

from skimage.filters import threshold_otsu as otsu
from skimage.filters import gaussian
from skimage.filters import difference_of_gaussians as DoG

from sklearn.cluster import (DBSCAN, OPTICS)
from sklearn.decomposition import PCA

from scipy.stats import mode, chi2, t, sem
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
                       sliding_window_width: int=15,
                       gaussian_sigma: int=None,
) -> (npt.NDArray[any], npt.NDArray[any]):
    """
    Create slice views from full tomogram for downstream processes

    Args:
    volume (ndarray) : input 3D image (tomogram)
    sliding_window_width (int) : width of sliding window for metric calculation for 2D slices
    gaussian_sigma (int) : Sigma parameter for Gaussian blurring kernal

    Returns:
    ndarray, ndarray
    """

    # Create sliding window views
    sliding_zy = swv(volume, sliding_window_width, axis=1)
    sliding_zx = swv(volume, sliding_window_width, axis=2)

    # Create views for calculation using sliding window views
    view_zy_raw = sliding_zy.std(axis=-1)
    view_zx_raw = sliding_zx.std(axis=-1)

    # Swap relevant axes to 0 for downstream processes
    view_zy = np.moveaxis(view_zy_raw, 1, 0)
    view_zx = np.moveaxis(view_zx_raw, 2, 0)

    if gaussian_sigma is not None:
        view_zy = gaussian(view_zy, sigma=gaussian_sigma)
        view_zx = gaussian(view_zx, sigma=gaussian_sigma)

    return (view_zy, view_zx)


def interpolate_surface(mesh_points: npt.NDArray[any],
                        n_points: int=100,
) -> (npt.NDArray, npt.NDArray, npt.NDArray):
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
        return int(np.argmin(mse_perc_change))

    return None


def refine_contour_LOO(contour_pts: npt.NDArray[any],
                       max_delete_perc: float=15.,
                       thres: float=0.1,
) -> list:
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


def evaluate_slice(view_input: npt.NDArray[any],
                   pixel_size_nm: float,
                   pt_thres: int=100,
                   autocontrast: bool=True,
) -> (float, float, float, int,
      npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray,
):
    """
    Function for creation and evaluation of one 2D slice from tomogram given coordinates of intersection point

    Args:
    volume (ndarray) : tomogram for evaluation
    pixel_size_nm (float) : pixel size of tomogram (internally binned)
    pt_thres (int) : acceptance limit (number of feature pixels) of slices
    autocontrast (bool) : whether to apply autocontrast on slices before evaluation

    Returns:
    float, float, float, int, ndarray, ndarray, ndarray, ndarray
    """

    view = view_input.T

    if autocontrast:
        view = autocontrast_slice(view)

    # Step 1: Greyvalue thresholding with Otsu method
    thres = otsu(view**2) * 1.25
    mask_s1 = np.argwhere(view**2 >= thres)

    # Skip slice if no pixels masked
    centroid_s1 = mask_s1.mean(axis=0)

    # Step 2: Use Mahalanobis distance to remove potential outliers
    jackknife_dist = np.empty(len(mask_s1))
    for idx in range(len(mask_s1)):
        temp = np.delete(mask_s1, idx, axis=0)
        S_inv = np.linalg.inv(np.cov(temp.T))
        diffs = (mask_s1[idx] - temp.mean(axis=0))[np.newaxis, :]
        jackknife_dist[idx] = np.sqrt( (diffs @ S_inv.T @ diffs.T).diagonal() )

    conf_limit = t.interval(confidence=0.999, df=2,
                            loc=jackknife_dist.mean(), scale=sem(jackknife_dist))[1]
    mask_s2 = np.squeeze(mask_s1[np.argwhere(jackknife_dist<=conf_limit)], axis=1)

    # Step 3: Use distance-based clustering to find sample region
    clusters = DBSCAN(eps=15, min_samples=1000).fit_predict(mask_s2)
    mask_s3_args = np.argwhere(clusters==mode(clusters, keepdims=True).mode)
    mask_s3 = mask_s2[mask_s3_args.flatten()]

    # Skip slice if no pixels masked
    centroid_s3 = mask_s3.mean(axis=0)

    # Step 4: Use PCA to find best rectangle fits
    pca = PCA(n_components=2)
    pca.fit(mask_s3)
    eigenvecs = pca.components_
    eigenvals = np.sqrt(pca.explained_variance_)
    rectangle_dims = eigenvals * 3      # 3 times SD to cover nearly all points

    # Determine breadth (long semi-minor) axis
    breadth_axis = np.argmin(np.abs(eigenvecs[:, 0]))
    if eigenvecs[breadth_axis, 1] < 0:
        eigenvecs[breadth_axis] *= -1 # Ensure breadth axis always points "right"
    if np.cross(eigenvecs[breadth_axis], eigenvecs[1-breadth_axis]) > 0:
        eigenvecs[1-breadth_axis] *= -1 # Ensure thickness axis always points "up"

    angle = 90 + np.rad2deg(np.arctan2(-eigenvecs[np.argmin(eigenvals), 1],
                                       eigenvecs[np.argmin(eigenvals), 0]))

    slice_breadth, slice_thickness = 2 * rectangle_dims * pixel_size_nm
    num_points = len(mask_s3)

    cell_vecs = eigenvecs * rectangle_dims.reshape((2, 1))

    # Centralise lamella centroid to middle of slice along long axis
    lamella_centre = centroid_s3
    lamella_to_slice_dist = np.linalg.norm(lamella_centre - 0.5*np.array(view.shape)) * 200 / max(view.shape)

    top_pt1 = lamella_centre + cell_vecs[1] + cell_vecs[0]
    top_pt2 = lamella_centre + cell_vecs[1] - cell_vecs[0]
    bottom_pt1 = lamella_centre - cell_vecs[1] + cell_vecs[0]
    bottom_pt2 = lamella_centre - cell_vecs[1] - cell_vecs[0]

    return (lamella_to_slice_dist, slice_breadth, slice_thickness, angle, num_points,
            top_pt1, top_pt2, bottom_pt1, bottom_pt2)


def evaluate_full_lamella(volume: npt.NDArray[any],
                          pixel_size_nm: float,
                          autocontrast: bool,
                          cpu: int=1,
                          discard_pct: float=20,
                          step_pct: float=2.5,
) -> (
    npt.NDArray, npt.NDArray, float, float, float, float, tuple
):
    """
    Evaluation of full lamella geometry

    Args:
    volume (ndarray) : Tomogram to be evaluated
    pixel_size_nm (float) : Pixel size (after binning) of tomogram
    autocontrast (bool) : whether to apply autocontrast on slices before evaluation
    cpu (int) : number of cores used in parallel calculation of slices
    discard_pct (float) : % of pixels from either end along an axis to discard.
    step_pct (float) : % of total pixels along an axis per step

    Returns:
    ndarray, ndarray, float, float, float, float, tuple

    """

    # Create slice views
    zy_stack, zx_stack = create_slice_views(
        volume=volume,
        sliding_window_width=5,
        gaussian_sigma=3
    )

    # Create slice stacks for assessment (trimming)
    zy_slice_list = np.arange(int(volume.shape[1]*0.01*discard_pct),
                             int(volume.shape[1]*0.01*(100-discard_pct)),
                             int(volume.shape[1]*0.01*step_pct))
    zx_slice_list = np.arange(int(volume.shape[2]*0.01*discard_pct),
                             int(volume.shape[2]*0.01*(100-discard_pct)),
                             int(volume.shape[2]*0.01*step_pct))

    zy_stack_assessment = zy_stack[zy_slice_list]
    zx_stack_assessment = zx_stack[zx_slice_list]

    # Evaluation along X axis (YZ-slices)
    with mp.Pool(cpu) as p:
        f = partial(evaluate_slice,
                    pixel_size_nm=pixel_size_nm,
                    autocontrast=autocontrast,
        )
        yz_output = np.array(p.map(f, zy_stack_assessment), dtype=object)

    yz_output = np.concatenate(yz_output).ravel().reshape((yz_output.size//9, 9))
    yz_full_stats = np.array(yz_output[:, :5], dtype=float)
    yz_surface_top_1_2d = np.concatenate(yz_output[:,5]).ravel().reshape(len(yz_output), 2)
    yz_surface_top_2_2d = np.concatenate(yz_output[:,6]).ravel().reshape(len(yz_output), 2)
    yz_surface_bottom_1_2d = np.concatenate(yz_output[:,7]).ravel().reshape(len(yz_output), 2)
    yz_surface_bottom_2_2d = np.concatenate(yz_output[:,8]).ravel().reshape(len(yz_output), 2)

    yz_surface_top_1 = np.insert(yz_surface_top_1_2d, 1, zy_slice_list, axis=1)
    yz_surface_top_2 = np.insert(yz_surface_top_2_2d, 1, zy_slice_list, axis=1)
    yz_surface_bottom_1 = np.insert(yz_surface_bottom_1_2d, 1, zy_slice_list, axis=1)
    yz_surface_bottom_2 = np.insert(yz_surface_bottom_2_2d, 1, zy_slice_list, axis=1)

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
    with mp.Pool(cpu) as p:
        f = partial(evaluate_slice,
                    pixel_size_nm=pixel_size_nm,
                    autocontrast=autocontrast,
        )
        xz_output = np.array(p.map(f, zx_stack_assessment), dtype=object)

    xz_full_stats = np.array(xz_output[:, :5], dtype=float)

    # xz_remove_empty = np.array([s for s in xz_full_stats if s[3]>0])
    # xz_thres = otsu(xz_remove_empty[:, 3])
    # xz_full_stats_refined = np.array([s for s in xz_remove_empty if s[3]>xz_thres])

    # Stat aggregation
    yz_mean = yz_full_stats_refined.mean(axis=0)
    xz_mean = xz_full_stats.mean(axis=0)
    yz_sem = sem(yz_full_stats_refined, axis=0)
    xz_sem = sem(xz_full_stats, axis=0)

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
            yz_sem, xz_sem,
            surfaces
    )
