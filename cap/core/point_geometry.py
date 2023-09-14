# Copyright (c) Changan Auto. All rights reserved.

# Implements functions which operate on lidar point domain, such as point
# rotation, point counting, etc.
# Due to the sparsity of the point cloud, these functions rely JIT compilation
# to speed up.
from copy import deepcopy
from typing import Tuple

import numba
import numpy as np

__all__ = [
    "rotation_points_single_angle",
    "points_count_convex_polygon_3d_jit",
    "points_in_convex_polygon_jit",
    "points_in_convex_polygon_3d_jit",
    "get_rotation_matrix",
    "coor_transformation",
    "_points_to_truncate_points_all",
    "dropout_points_in_box",
    "points_to_truncate_points",
]


def rotation_points_single_angle(
    points: np.ndarray, angle: float, axis: int = 0
):
    """Rotate points around an axis.

    Args:
        points (np.ndarray): [N, 3] point array.
        angle (float): rotation angle.
        axis (int, optional): axis index. Defaults to 0.

    Raises:
        ValueError: if axis is not in [-1, 0, 1, 2].

    Returns:
        (np.ndarray): rotated points.
    """
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype,
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype,
        )
    else:
        raise ValueError("axis should be in range -1, 0, 1, 2")

    return points @ rot_mat_T


@numba.njit
def _points_count_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    num_surfaces: np.ndarray = None,
):
    """Count points in 3d convex polygons.

    Args:
        points (np.ndarray): [num_points, 3] array of points.
        polygon_surfaces (np.ndarray): [M, S, P, 3] array of surfaces, where:
            M = number of polygons
            S = max surfaces each polygon has. In cuboid case, S = 6.
            P = max points each surface has. In cuboid case, P = 4.
            The last dimension is each point's coordinates.
        normal_vec (np.ndarray): [M, S, 3] normal vector array.
        d (np.ndarray): [M, S] distance matrix.
        num_surfaces (np.ndarray, optional): [M] array describing how many
            surfaces a polygon contains. Defaults to None.

    Returns:
        ret (np.ndarray): number of points in each polygon.
    """
    max_num_surfaces = polygon_surfaces.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.full((num_polygons,), num_points, dtype=np.int64)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[j] -= 1
                    break
    return ret


@numba.jit
def points_count_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    num_surfaces: np.ndarray = None,
):
    """Count number of points in each convex polygon.

    Args:
        points (np.ndarray): [N, >=3] array of point locations.
        polygon_surfaces (np.ndarray): [M, S, P, 3] array of surfaces, where:
            M = number of polygons
            S = max surfaces each polygon has. In cuboid case, S = 6.
            P = max points each surface has. In cuboid case, P = 4.
            The last dimension is each point's coordinates.
        num_surfaces (np.ndarray, optional): number of surfaces.
            Defaults to None.

    Returns:
        (np.ndarray): number of points in each polygon.
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_count_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.njit
def _points_in_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    num_surfaces: np.ndarray = None,
):
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3] array. All surfaces' normal vector
            must direct to internal.
            max_num_points_of_surface must at least 3.
        normal_vec: normal vectors.
        d: distance matrix.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain.
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces = polygon_surfaces.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    num_surfaces: np.ndarray = None,
):
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain.
    Returns:
        [num_points, num_polygon] bool array.
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.njit
def surface_equ_3d_jitv2(surfaces: np.ndarray):
    """Calculate surface normal vectors and distances.

    Args:
        surfaces (np.ndarray): [M, S, 3, 3] tensor of polygon surfaces, where:
            M = number of polygons
            S = max surfaces each polygon has.
            3 = points that define a suface.
            3 = xyz coords of a point.

    Returns:
        normal_vec (np.ndarray): [M, S, 3] tensor of normal vectors.
        d (np.ndarray): [M, S] distance matrix. Note that this distance might
            not be normalized.
    """
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros(
        (num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype
    )
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            # Cross product of 2 surface vectors to get normal vector
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (
                -surfaces[i, j, 0, 0] * normal_vec[i, j, 0]
                - surfaces[i, j, 0, 1] * normal_vec[i, j, 1]
                - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
            )
    return normal_vec, d


@numba.jit
def points_in_convex_polygon_jit(
    points: np.ndarray, polygon: np.ndarray, clockwise: bool = True
):
    """Check points is in 2d convex polygons. True when point in polygon.

    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = (
            polygon
            - polygon[
                :,
                [num_points_of_polygon - 1]
                + list(range(num_points_of_polygon - 1)),
                :,
            ]
        )
    else:
        vec1 = (
            polygon[
                :,
                [num_points_of_polygon - 1]
                + list(range(num_points_of_polygon - 1)),
                :,
            ]
            - polygon
        )
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret


def get_rotation_matrix(theta, translation=None):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    rotation_mat = np.stack(
        [
            np.stack([cos_theta, -sin_theta], axis=-1),
            np.stack([sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )
    if translation is not None:
        padding_size = ((0, 0),) * (len(rotation_mat.shape) - 2)
        expand_rot_mat = np.pad(
            rotation_mat, padding_size + ((0, 1), (0, 1)), mode="constant"
        )
        expand_rot_mat[..., 2, 2] = 1
        expand_rot_mat[..., :2, 2] = np.array(translation)
        return expand_rot_mat
    else:
        return rotation_mat


def coor_transformation(x, theta=None, translation=None):
    y = x.copy()
    if theta is not None:
        rotation_mat = get_rotation_matrix(theta)
        y = np.sum(y[..., None, :] * rotation_mat, axis=-1)
    if translation is not None:
        y += translation
    return y


@numba.njit
def _points_to_truncate_points_all(
    points: np.ndarray,
    angle_range: Tuple[float, float] = (0, 180),
    center: Tuple[float, float, float] = (0, 0, 0),
    axis_rotat: float = 90,
):
    """
    Axis_rotat:Degrees of clockwise rotation.

    Args:
        points: [num_points, 3].
        angle_range : truncate angle.
        center : truncate center point.
        axis_rotat: aixs rotat angle.
    Returns:
        [num_polygon] array.
    """
    keep_list = []
    for i, v in enumerate(points):

        x = v[0] - center[0]
        y = v[1] - center[1]
        if x > 0 and y >= 0:
            rad = np.arctan(y / x)
        elif x <= 0 and y > 0:
            rad = np.pi / 2 - np.arctan(x / y)
        elif x < 0 and y <= 0:
            rad = np.pi + np.arctan(y / x)
        elif x >= 0 and y < 0:
            rad = np.pi * 3 / 2 - np.arctan(x / y)
        angle = (np.rad2deg(rad) + axis_rotat) % 360

        if angle_range[0] <= angle <= angle_range[1]:
            keep_list.append(i)
    return keep_list


@numba.njit
def _filter_points_to_trancate_points_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    angle_range: Tuple[float, float] = (0, 180),
):
    """
    Kickout points in 3d convex polygons which not in the range.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
        [num_polygon] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    sign = 0.0
    delet_point_idx = []
    for i in range(num_points):
        sign_flag = True
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                sign_flag = sign < 0 and sign_flag
            if sign_flag:
                delet_point_idx.append(i)
                break
    return delet_point_idx


def points_to_truncate_points(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    ther_angle: Tuple[float, float] = (0, 180),
):
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
       truncate_poins
    """
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3] [n,6,3]
    # d: [num_polygon, max_num_surfaces]    #
    delet_point_idx = _filter_points_to_trancate_points_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, ther_angle
    )

    points = np.delete(points, delet_point_idx, axis=0)
    return points


def dropout_points_in_box(
    points: np.ndarray, polygon_surfaces: np.ndarray, prob: float = 0.2
):
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
       truncate_poins
    """
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3] [n,6,3]
    # To express each face with three-dimensional vector
    # only three points are needed:
    # (vector representation of three-dimensional plane)
    # d: [num_polygon, max_num_surfaces]
    prob_array = np.random.random((points.shape[0], 1))
    points_temp = deepcopy(points)
    points_temp = np.hstack([points_temp, prob_array])
    # Since the internal operation of numba does
    # not support numpy random numbers,
    # the probability is generated externally

    delet_point_idx = _dropout_points_onbox_polygon_3d_jit(
        points_temp, polygon_surfaces, normal_vec, d, prob
    )

    points = np.delete(points, delet_point_idx, axis=0)

    return points


@numba.njit
def _dropout_points_onbox_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    prob: float,
):
    """Dropout points in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
        [dropout points idx] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    delet_point_idx = []
    for i in range(num_points):
        sign_flag = True
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                sign_flag = sign < 0 and sign_flag
            if sign_flag and points[i, -1] < prob:
                delet_point_idx.append(i)
                break
    return delet_point_idx
