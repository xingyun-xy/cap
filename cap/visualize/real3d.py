import copy

import cv2
import numpy as np


def draw_projected_box3d(
    image, qs, color=(0, 255, 0), thickness=2, show_arrow=True
):
    try:
        qs = qs.astype(np.int32)
    except BaseException:
        return image
    for k in range(0, 4):
        # Ref:
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html  # noqa
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)  # noqa
        cv2.line(
            image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
        )
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(
            image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
        )

        i, j = k, k + 4
        cv2.line(
            image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
        )
    if show_arrow:
        # 4,5,6,7
        p1 = (qs[0, :] + qs[1, :] + qs[2, :] + qs[3, :]) / 4
        p2 = (qs[0, :] + qs[1, :]) / 2
        p3 = p2 + (p2 - p1) * 0.5

        p1 = p1.astype(np.int32)
        p3 = p3.astype(np.int32)
        cv2.line(
            image,
            (p1[0], p1[1]),
            (p3[0], p3[1]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def draw_projected_box2d(
    image, qs, color=(0, 255, 0), thickness=2, show_arrow=True
):
    try:
        qs = qs.astype(np.int32)
    except BaseException:
        return image
    cv2.line(
        image, (qs[0, 0], qs[0, 1]), (qs[1, 0], qs[1, 1]), color, thickness
    )
    cv2.line(
        image, (qs[1, 0], qs[1, 1]), (qs[2, 0], qs[2, 1]), color, thickness
    )
    cv2.line(
        image, (qs[2, 0], qs[2, 1]), (qs[3, 0], qs[3, 1]), color, thickness
    )
    cv2.line(
        image, (qs[3, 0], qs[3, 1]), (qs[0, 0], qs[0, 1]), color, thickness
    )
    if show_arrow:
        p1 = np.mean(qs, axis=0)
        p2 = (qs[0, :] + qs[3, :]) / 2
        p3 = p2 + (p2 - p1) * 0.5

        p1 = p1.astype(np.int32)
        p3 = p3.astype(np.int32)
        cv2.line(
            image,
            (p1[0], p1[1]),
            (p3[0], p3[1]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def compute_box_3d(dim, location, rotation_y, pitch=0.0):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)  # y
    cx, sx = np.cos(pitch), np.sin(pitch)
    Rx = np.array(
        [[1, 0, 0], [0, cx, sx], [0, -sx, cx]], dtype=np.float32
    )  # x
    l, w, h = dim[2], dim[1], dim[0]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(Ry, corners)
    corners_3d = np.dot(Rx, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
        3, 1
    )
    return corners_3d.transpose(1, 0)


def project_to_image(pts_3d, P, dist_coeff=None, fisheye=False):
    P = np.array(P)
    if dist_coeff is not None:
        if not fisheye:
            rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)
            dist_coeff = np.array(dist_coeff)
            image_pts = cv2.projectPoints(
                pts_3d[:, :3], np.array(rvec), tvec, P[:, :3], dist_coeff
            )[0]
            pts_2d = np.squeeze(image_pts)
        else:
            pts_3d = pts_3d[:, :3]
            pts_3d[pts_3d[:, 2] < 0, 2] = 0.001
            pts_3d = np.expand_dims(pts_3d, 0)
            rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)
            dist_coeff = np.array(dist_coeff, dtype=np.float32)
            fx, fy = P[0, 0], P[1, 1]
            u, v = P[0, 2], P[1, 2]
            k_ = np.mat(
                [[fx, 0.0, u], [0.0, fy, v], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            d_ = np.mat(dist_coeff[:4].T, dtype=np.float32)
            image_pts = cv2.fisheye.projectPoints(
                pts_3d, np.array(rvec), tvec, k_, d_
            )[0]
            pts_2d = np.squeeze(image_pts)
    else:
        pts_3d_homo = np.concatenate(
            [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1
        )
        pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d


def compute_2d_points(
    dim, location, yaw, calib, dist_coeff=None, fisheye=False, pitch=0.0
):
    corners3d = compute_box_3d(dim, location, yaw, pitch)
    corners3d[:, 2][corners3d[:, 2] < 0] = 0.1
    corners3d_proj = project_to_image(
        corners3d, calib, dist_coeff=dist_coeff, fisheye=fisheye
    )
    corners3d_proj = corners3d_proj.reshape(-1, 2).astype(np.int32)
    return corners3d_proj


def get_3dboxcorner_in_cam(dim, location, rotation_y):
    """Convert from KITTI label to 3dbox in velo."""
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
        3, 1
    )

    return corners_3d.transpose()


def corners_to_local_rot_y(corners):
    x, y = 0.5 * (
        corners[0, [2, 0]]
        + corners[3, [2, 0]]
        - corners[1, [2, 0]]
        - corners[2, [2, 0]]
    )
    rot_y = np.arctan2(y, x) - np.pi / 2
    if rot_y < -np.pi:
        rot_y += 2 * np.pi
    elif rot_y > np.pi:
        rot_y -= 2 * np.pi
    return rot_y


def project_camera_to_velo(loc_c, dim_c, yaw_c, r_cam2vel, T_vel2cam):
    points = get_3dboxcorner_in_cam(dim_c, loc_c, yaw_c)
    points_vel = np.dot(r_cam2vel, (points - T_vel2cam).T)
    points_vel_ = points_vel[:, (0, 3, 2, 1, 4, 7, 6, 5)]
    yaw_vel = corners_to_local_rot_y(points_vel_.T)
    return yaw_vel


def get_3dboxcorner_in_velo(box3d, with_size=False):
    """Convert from KITTI label to 3dbox in velo."""
    x, y, z, w, l, h, yaw = box3d
    corner = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    rotMat = np.array(
        [
            [np.cos(np.pi / 2 + yaw), np.sin(np.pi / 2 + yaw), 0.0],
            [-np.sin(np.pi / 2 + yaw), np.cos(np.pi / 2 + yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cornerPosInVelo = (
        np.dot(rotMat, corner) + np.tile(np.array([x, y, z]), (8, 1)).T
    )
    box3d = cornerPosInVelo.transpose()
    if with_size:
        return box3d, h, w, l
    return box3d


def camera2velo(loc, yaw, Tr_vel2cam):
    Tr_vel2cam = np.array(Tr_vel2cam)
    T_vel2cam = Tr_vel2cam[:3, -1]
    r_vel2cam = Tr_vel2cam[:3, :3]
    r_cam2vel = np.linalg.inv(r_vel2cam)
    loc_vel = np.dot(r_cam2vel, loc - T_vel2cam)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    yaw_vel = np.arctan2(
        np.dot(r_cam2vel, R)[0, 2], np.dot(r_cam2vel, R)[0, 0]
    )
    return loc_vel, yaw_vel


def compute_2d_box_V2(
    dim,
    location,
    yaw,
    calib,
    distCoeffs=None,
    fisheye=False,
    pitch=0.0,
    img_w=2048,
    img_h=1024,
    ratio=0.5,
):
    points_cam_input = compute_box_3d(dim, location, yaw, pitch)
    points_cam = copy.deepcopy(points_cam_input)
    VALID_Z = 0.1
    EMinX = -ratio * img_w
    EMinY = -ratio * img_h
    EMaxX = (1 + ratio) * img_w
    EMaxY = (1 + ratio) * img_h
    z_valid_index = np.squeeze(np.argwhere(points_cam[:, 2] > VALID_Z))
    if z_valid_index.size <= 2:
        return None, None, 1
    z_valid_points = points_cam[z_valid_index, :]
    image_pts = project_to_image(z_valid_points, calib, distCoeffs, fisheye)
    valid_point_index = np.squeeze(
        np.argwhere(
            np.logical_and(
                np.logical_and(
                    image_pts[:, 0] > EMinX, image_pts[:, 0] < EMaxX
                ),
                np.logical_and(
                    image_pts[:, 1] > EMinY, image_pts[:, 1] < EMaxY
                ),
            )
        )
    )
    if valid_point_index.size <= 2:
        return None, None, 1
    if valid_point_index.size == 8:
        rate_truncate = 0
    else:
        box_h = np.max(points_cam[:, 1]) - np.min(points_cam[:, 1])
        valid_point_original = z_valid_index[valid_point_index]
        valid_point_cam = points_cam[valid_point_original, :]
        valid_points_indx_set = set(valid_point_original)
        if np.pi * 2 / 5 < np.abs(yaw) < np.pi * 3 / 5:
            delta_index = 2
        else:
            delta_index = 0
        for i in range(points_cam.shape[0]):
            if i not in valid_points_indx_set:
                pp = points_cam[i, :]
                candi_anchors = valid_point_cam[
                    np.abs(pp[1] - valid_point_cam[:, 1]) < box_h * 0.4, :
                ]
                if candi_anchors.shape[0] == 0:
                    print("Error, there should be at least one valid point")
                    return None, None, 1
                elif candi_anchors.shape[0] == 1:
                    pp_nearest = candi_anchors[0, :]
                else:
                    dist = np.sum(
                        np.square(candi_anchors[:, [0, 2]] - pp[[0, 2]]),
                        axis=-1,
                    )
                    pp_nearest = candi_anchors[np.argsort(dist)[-2]]
                ll = pp_nearest[delta_index]
                rr = pp[delta_index]
                edge_p = np.array([pp[0], pp[1], pp_nearest[2]]).reshape(
                    (1, 3)
                )
                while np.abs(ll - rr) > 1e-1:
                    mid = (ll + rr) / 2.0
                    if delta_index == 0:
                        edge_p[0, 0] = mid
                        edge_p[0, 2] = (pp[2] - pp_nearest[2]) * (
                            mid - pp[0]
                        ) / (pp[0] - pp_nearest[0]) + pp[2]
                    else:
                        edge_p[0, 0] = (pp[0] - pp_nearest[0]) * (
                            mid - pp[2]
                        ) / (pp[2] - pp_nearest[2]) + pp[0]
                        edge_p[0, 2] = mid
                    if edge_p[0, 2] < VALID_Z:
                        rr = mid
                        continue
                    edge_p_image = project_to_image(edge_p, calib, distCoeffs)
                    if len(edge_p_image.shape) == 1:
                        edge_p_image = np.expand_dims(edge_p_image, 0)
                    if (
                        EMinX < edge_p_image[0, 0] < EMaxX
                        or EMinY < edge_p_image[0, 1] < EMaxY
                    ):
                        ll = mid
                    else:
                        rr = mid
                points_cam[i, :] = edge_p[0, :]

        image_pts = project_to_image(points_cam, calib, distCoeffs, fisheye)
        Fc = np.mean(points_cam[[0, 1, 4, 5], :], axis=0)
        Bc = np.mean(points_cam[[2, 3, 6, 7], :], axis=0)
        Fc_ori = np.mean(points_cam_input[[0, 1, 4, 5], :], axis=0)
        Bc_ori = np.mean(points_cam_input[[2, 3, 6, 7], :], axis=0)
        l_ori = np.sqrt(np.sum(np.square(Bc_ori - Fc_ori)))
        l_truncated = np.sqrt(np.sum(np.square(Bc - Fc)))
        rate_truncate = (l_ori - l_truncated) / l_ori

    corners3d_proj = image_pts.reshape(-1, 2).astype(np.int32)
    bbox2d = np.concatenate(
        [np.min(corners3d_proj, axis=0), np.max(corners3d_proj, axis=0)]
    )
    return corners3d_proj, bbox2d, rate_truncate


def compute_2d_box(dim, location, yaw, calib, dist_coeff=None, fisheye=False):
    corners3d = compute_box_3d(dim, location, yaw)
    corners3d_proj = project_to_image(
        corners3d, calib, dist_coeff=dist_coeff, fisheye=fisheye
    )
    corners3d_proj = corners3d_proj.reshape(-1, 2).astype(np.int32)
    bbox2d = np.concatenate(
        [np.min(corners3d_proj, axis=0), np.max(corners3d_proj, axis=0)]
    )
    return bbox2d
