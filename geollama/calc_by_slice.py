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
## Date last modified : 22-Jul-2025             ##
##################################################

import itertools
from typing import Optional
from functools import partial
import multiprocessing as mp

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view as swv

from skimage.filters import threshold_otsu as otsu
from skimage.filters import gaussian

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from scipy.stats import mode, t, sem
import scipy.interpolate as spin
from scipy.signal import convolve2d as c2d


def filter_bandpass(
    image: npt.NDArray[any],
) -> npt.NDArray[any]:
    """
    Perform bandpass filtering on image using scikit-image's difference of Gaussian function.

    Parameters
    ----------
    image : ndarray
        Image to be bandpass-filtered.

    Returns
    -------
    filtered_image : ndarray
        Bandpass-filtered image

    """
    from skimage.filters import difference_of_gaussians as DoG

    filtered_image = DoG(image, min(image.shape) * 0.01, max(image.shape) * 0.05)

    return filtered_image


def autocontrast_slice(
    image: npt.NDArray[any],
    target_mean: float = 150,
    target_std: float = 40,
    clip_cutoff: float = 3,
) -> npt.NDArray[any]:
    """
    Auto-contrast image slice by clipping and histogram stretching.

    The image grey-values are converted to Z-scores first, and then clipped according to given parameters.
    The clipped image histogram is then re-equalised to a target mean and standard deviation.

    Parameters
    ----------
    image : ndarray
        Input image for auto-contrast.
    target_mean : float, default=150,
        Target mean grey-value of contrast-corrected image.
    target_std : float, default=40
        Target mean grey-value of contrast-corrected image.
    clip_cutoff : float, default=3
        Cutoff threshold for clipping of image grey-value Z-scores.

    Returns
    -------
    img_corrected : ndarray
        Output image after auto-contrast.

    """
    modified_z_score = (image - np.mean(image)) / (np.std(image) * clip_cutoff)
    z_clipped = np.clip(modified_z_score, -1, 1)
    z_clipped_std = np.std(z_clipped)

    img_corrected = z_clipped * target_std / z_clipped_std + target_mean

    return img_corrected


def create_slice_views(
    volume: npt.NDArray[any],
    sliding_window_width: int = 15,
    gaussian_sigma: int = None,
) -> (npt.NDArray[any], npt.NDArray[any]):
    """
    Create slice views from full tomogram for downstream processes.

    Sliding windows are first created out of the tomogram along the two "horizontal" axes, along which
    the standard deviation are created.
    The curated (reduced) volumes are than transposed such that the assessment axes are in the first positions. For
    example, the ZXY->XYZ transposition for the ZY-view.
    Gaussian filter will be applied to the curated volumes if the `gaussian_sigma` parameter is specified.

    Parameters
    ----------
    volume : ndarray
        Input 3D image (tomogram).
    sliding_window_width : int, default=15
        Width of sliding window for metric calculation of 2D slices
    gaussian_sigma : int, default=None
        Sigma parameter for Gaussian blurring kernel. Gaussian blurring applied only if parameter is not None.

    Returns
    -------
    view_zy : ndarray
        Stack of 2D images (ZY-planes) for evaluation along the X-axis
    view_zx : ndarray
        Stack of 2D images (ZX-planes) for evaluation along the Y-axis
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


def interpolate_surface(
    mesh_points: npt.NDArray[any],
    n_points: int = 100,
) -> (npt.NDArray, npt.NDArray, npt.NDArray):
    """
    Interpolates surface with given mesh points using scipy.interpolate.

    Parameters
    ----------
    mesh_points : ndarray
        Input mesh points for reconstructing surface
    n_points : int, default=100
        Number of points per dimension for interpolation

    Returns
    -------
    xx, yy : tuple of ndarrays
        Mesh points along X and Y axes as outputs from numpy.meshgrid.
        For detailed explanation, see numpy documentation.
    surface : ndarray
        Interpolated surface height (in Z direction)
    """
    x_mesh = np.linspace(np.min(mesh_points[:, 1]), np.max(mesh_points[:, 1]), n_points)
    y_mesh = np.linspace(np.min(mesh_points[:, 2]), np.max(mesh_points[:, 2]), n_points)
    xx, yy = np.meshgrid(x_mesh, y_mesh)

    interp = spin.LinearNDInterpolator(mesh_points[:, 1:], mesh_points[:, 0])
    surface = interp(xx, yy)

    return xx, yy, surface


def generalised_theil_sen_fit(
    contour_pts: npt.NDArray[any],
) -> (npt.NDArray, npt.NDArray, float):
    """
    Linear fitting of 3D points using a generalised Theil-Sen algorithm.

    Parameters
    ----------
    contour_pts : ndarray
        Contour points in 3D space for fitting.

    Returns
    -------
    grad_norm : ndarray
        Normalised gradient of fitted line
    ptl_median : ndarray
        Offset of fitted line
    mse : float
        Mean-squared error of fit

    References
    ----------
    .. [1] Theil-Sen algorithm, https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator
    """
    # Estimate gradient of fitted line
    point_pairs = np.array(list(itertools.combinations(contour_pts, 2)))
    grad_vectors = np.array([p[1] - p[0] for p in point_pairs])
    grad_median = np.median(grad_vectors, axis=0)
    grad_norm = grad_median / np.linalg.norm(grad_median)

    # Estimate offset of fitted line using median of point-to-line vectors
    ptl_vectors = (
        contour_pts - np.dot(contour_pts, grad_norm)[..., np.newaxis] * grad_norm
    )
    ptl_median = np.median(ptl_vectors, axis=0)

    # Final fit + mean-squared error calculation
    fitted_ptl_vectors = (contour_pts - ptl_median) - np.dot(
        contour_pts - ptl_median, grad_norm
    )[..., np.newaxis] * grad_norm
    mse = np.mean(np.linalg.norm(fitted_ptl_vectors, axis=0) ** 2)

    return (grad_norm, ptl_median, mse)


def leave_one_out(
    contour_pts: npt.NDArray[any],
    thres: float = 0.1,
) -> Optional[int]:
    """
    Determine the worst outlier point from an ensemble using the leave-one-out (LOO) algorithm.
    If by removing the worst candidate outlier point the mean-squared error would change by less than the given threshold,
    then there is no real outlier.

    Parameters
    ----------
    contour_pts : ndarray
        Input ensemble for outlier detection
    thres : float, default=0.1
        Threshold mean-squared error change for outlier detection

    Returns
    -------
    int or None
        Returns the index of the worst outlier point in the given ensemble if applicable, otherwise returns None
    """
    mse_perc_change = np.empty(len(contour_pts))

    # Initial fit
    grad_0, passing_0, mse_0 = generalised_theil_sen_fit(contour_pts)

    # Leave-one-out algorithm
    for idx, points in enumerate(contour_pts):
        temp = np.delete(contour_pts, idx, axis=0)
        grad, passing, mse = generalised_theil_sen_fit(temp)
        mse_perc_change[idx] = (mse - mse_0) / mse_0

    # Determine whether worst point is an outlier using given threshold
    # If the point is an outlier, determine its index
    if mse_perc_change.min() < -thres:
        return int(np.argmin(mse_perc_change))

    return None


def refine_contour_LOO(
    contour_pts: npt.NDArray[any],
    max_delete_perc: float = 15.0,
    thres: float = 0.1,
) -> list:
    """
    Iteratively remove outliers using the leave-one-out algorithm.
    This function is a wrapper function of `leave_one_out`.

    Parameters
    ----------
    contour_pts : ndarray
        Input ensemble of contour points for refinement
    max_delete_perc : float, default=15.0
        Maximum percentage of slices allowed to be removed
    thres : float, default=0.1
        Threshold mean-squared error change for outlier detection

    Returns
    -------
    remove_list : list
        List of indices of slices in the given ensemble for removal
    """
    max_iter = int(len(contour_pts) * max_delete_perc * 0.01)
    slice_list = list(range(len(contour_pts)))
    remove_list = []

    temp = np.copy(contour_pts)
    for curr_iter in range(max_iter):
        remove_idx = leave_one_out(temp, thres=thres)
        if remove_idx is None:
            return remove_list

        remove_list.append(slice_list.pop(remove_idx))
        temp = np.delete(temp, remove_idx, axis=0)

    return remove_list


def evaluate_slice(
    view_input: npt.NDArray[any],
    pixel_size_nm: float,
    autocontrast: bool = True,
) -> (
    float,
    float,
    float,
    float,
    int,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
):
    """
    Evaluates lamella geometry at given position.

    Parameters
    ----------
    view_input : ndarray
        Slice of tomogram for evaluation
    pixel_size_nm : float
        Pixel spacing of tomogram (after internal binning)
    autocontrast : bool, default=True
        Apply auto-contrast on slice before evaluation

    Returns
    -------
    lamella_to_slice_dist : float
        Maximum displacement of centroid of lamella from centre of given image (in %)
    slice_breadth : float
        Breadth of lamella in image (in unit of nm)
    slice_thickness : float
        Thickness of lamella in image (in unit of nm)
    angle : float
        Tilt angle of lamella with respect to image
    num_points : int
        Number of pixels in given image determined as feature from lamella
    top_pt1 : ndarray
        Coordinates of one of the "top" lamella corners in image, calculated from lamella cell vectors
    top_pt2 : ndarray
        Coordinates of the other "top" lamella corner in image, calculated from lamella cell vectors
    bottom_pt1 : ndarray
        Coordinates of one of the "bottom" lamella corners in image, calculated from lamella cell vectors
    bottom_pt2 : ndarray
        Coordinates of the other "bottom" lamella corner in image, calculated from lamella cell vectors
    """
    view = view_input.T

    if autocontrast:
        view = autocontrast_slice(view)

    # Step 1: Greyvalue thresholding with Otsu method
    thres = otsu(view) * 1.12
    mask_s1 = np.argwhere(view >= thres)

    # Step 2: Use Jackknife distance to remove potential outliers
    jackknife_dist = np.empty(len(mask_s1))
    for idx in range(len(mask_s1)):
        temp = np.delete(mask_s1, idx, axis=0)
        S_inv = np.linalg.inv(np.cov(temp.T))
        diffs = (mask_s1[idx] - temp.mean(axis=0))[np.newaxis, :]
        jackknife_dist[idx] = np.linalg.norm((diffs @ S_inv.T @ diffs.T).diagonal())

    conf_limit = t.interval(
        confidence=0.99, df=2, loc=jackknife_dist.mean(), scale=sem(jackknife_dist)
    )[1]
    mask_s2 = np.squeeze(mask_s1[np.argwhere(jackknife_dist <= conf_limit)], axis=1)

    # Step 3: Use distance-based clustering to find sample region
    min_samples = min(350, len(mask_s2))
    eps = np.sqrt(2 * min_samples / np.pi)
    clusters = HDBSCAN(
        cluster_selection_epsilon=eps, min_samples=min_samples
    ).fit_predict(mask_s2)
    mask_s3_args = np.argwhere(clusters == mode(clusters, keepdims=True).mode)
    mask_s3 = mask_s2[mask_s3_args.flatten()]

    # Calculate weight-corrected centroid of lamella
    pointcloud_2d = np.zeros_like(view)
    pointcloud_2d[[tuple(i) for i in mask_s3]] = 1
    pointcloud_density = c2d(pointcloud_2d, np.full((5, 5), 1), "same")
    weights_2d = 1 / (1 + pointcloud_density)
    masked_weights = [weights_2d[tuple(i)] for i in mask_s3]

    lamella_centre = np.average(mask_s3, weights=masked_weights, axis=0)

    # Step 4: Use PCA to find best rectangle fits
    pca = PCA(n_components=2)
    pca.fit(mask_s3 - lamella_centre)
    eigenvecs = pca.components_
    eigenvals = np.sqrt(pca.explained_variance_)
    rectangle_dims = eigenvals * 3  # 3 times SD to cover nearly all points

    # Determine breadth (semi-major) axis
    breadth_axis = np.argmin(np.abs(eigenvecs[:, 0]))
    if eigenvecs[breadth_axis, 1] < 0:
        eigenvecs[breadth_axis] *= -1  # Ensure breadth axis always points "right"
    if np.cross(eigenvecs[breadth_axis], eigenvecs[1 - breadth_axis]) > 0:
        eigenvecs[1 - breadth_axis] *= -1  # Ensure thickness axis always points "up"

    angle = 90 + np.rad2deg(
        np.arctan2(
            -eigenvecs[np.argmin(eigenvals), 1], eigenvecs[np.argmin(eigenvals), 0]
        )
    )

    slice_breadth, slice_thickness = 2 * rectangle_dims * pixel_size_nm
    num_points = len(mask_s3)

    cell_vecs = eigenvecs * rectangle_dims.reshape((2, 1))

    # Centralise lamella centroid to middle of slice along long axis
    lamella_to_slice_dist = 200 * max(
        np.abs(lamella_centre / np.array(view.shape) - 0.5)
    )

    top_pt1 = lamella_centre + cell_vecs[1] + cell_vecs[0]
    top_pt2 = lamella_centre + cell_vecs[1] - cell_vecs[0]
    bottom_pt1 = lamella_centre - cell_vecs[1] + cell_vecs[0]
    bottom_pt2 = lamella_centre - cell_vecs[1] - cell_vecs[0]

    return (
        lamella_to_slice_dist,
        slice_breadth,
        slice_thickness,
        angle,
        num_points,
        top_pt1,
        top_pt2,
        bottom_pt1,
        bottom_pt2,
    )


def evaluate_full_lamella(
    volume: npt.NDArray[any],
    pixel_size_nm: float,
    autocontrast: bool,
    cpu: int = 1,
    discard_pct: float = 20,
    step_pct: float = 2.5,
) -> (
    npt.NDArray,
    npt.NDArray,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    tuple,
):
    """
    Evaluate geometry of lamella from given tomogram.

    Parameters
    ----------
    volume : ndarray
        Tomogram to be evaluated
    pixel_size_nm : float
        Pixel spacing of tomogram (after internal binning)
    autocontrast : bool
        Apply auto-contrast on slice before evaluation
    cpu : int, default=1
        Number of cores used in parallel calculation of slices
    discard_pct : float, default=20
        % of pixels from either end along an axis to discard
    step_pct : float, default=2.5
        % of total pixels along an axis per step (e.g. for an axis of length 100 pixels and the `step_pct=5.0`, the
        absolute step size will be 5 pixels per step)

    Returns
    -------
    yz_full_stats : ndarray
        ndarray storing raw (full) evaluation statistics of tomogram along the X-axis
    xz_full_stats : ndarray
        ndarray storing raw (full) evaluation statistics of tomogram along the Y-axis
    yz_mean : array_like
        array storing all the mean values calculated using `yz_full_stats`
    xz_mean : array_like
        array storing all the mean values calculated using `xz_full_stats`
    yz_sem : array_like
        array storing the standard error of mean of measurements, aggregated from `yz_full_stats`
    xz_sem : array_like
        array storing the standard error of mean of measurements, aggregated from `xz_full_stats`
    surfaces : tuple of ndarrays
        tuple storing all information of interpolated lamella surfaces, used for plotting
    """
    # Create slice views
    zy_stack, zx_stack = create_slice_views(
        volume=volume, sliding_window_width=5, gaussian_sigma=3
    )

    # Create slice stacks for assessment (trimming)
    zy_slice_list = np.arange(
        int(volume.shape[1] * 0.01 * discard_pct),
        int(volume.shape[1] * 0.01 * (100 - discard_pct)),
        int(volume.shape[1] * 0.01 * step_pct),
    )
    zx_slice_list = np.arange(
        int(volume.shape[2] * 0.01 * discard_pct),
        int(volume.shape[2] * 0.01 * (100 - discard_pct)),
        int(volume.shape[2] * 0.01 * step_pct),
    )

    zy_stack_assessment = zy_stack[zy_slice_list]
    zx_stack_assessment = zx_stack[zx_slice_list]

    # Evaluation along X axis (YZ-slices)
    with mp.Pool(cpu) as p:
        f = partial(
            evaluate_slice,
            pixel_size_nm=pixel_size_nm,
            autocontrast=autocontrast,
        )
        yz_output = np.array(p.map(f, zy_stack_assessment), dtype=object)

    yz_output = np.concatenate(yz_output).ravel().reshape((yz_output.size // 9, 9))
    yz_full_stats = np.array(yz_output[:, :5], dtype=float)
    yz_surface_top_1_2d = (
        np.concatenate(yz_output[:, 5]).ravel().reshape(len(yz_output), 2)
    )
    yz_surface_top_2_2d = (
        np.concatenate(yz_output[:, 6]).ravel().reshape(len(yz_output), 2)
    )
    yz_surface_bottom_1_2d = (
        np.concatenate(yz_output[:, 7]).ravel().reshape(len(yz_output), 2)
    )
    yz_surface_bottom_2_2d = (
        np.concatenate(yz_output[:, 8]).ravel().reshape(len(yz_output), 2)
    )

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

    # Evaluation along Y axis (XZ-slices)
    with mp.Pool(cpu) as p:
        f = partial(
            evaluate_slice,
            pixel_size_nm=pixel_size_nm,
            autocontrast=autocontrast,
        )
        xz_output = np.array(p.map(f, zx_stack_assessment), dtype=object)

    xz_full_stats = np.array(xz_output[:, :5], dtype=float)

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

    surfaces = (
        xx_top,
        yy_top,
        surface_interp_top,
        xx_bottom,
        yy_bottom,
        surface_interp_bottom,
        yz_surface_top,
        yz_surface_bottom,
    )

    return (yz_full_stats, xz_full_stats, yz_mean, xz_mean, yz_sem, xz_sem, surfaces)
