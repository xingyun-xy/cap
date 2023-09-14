import math

import cv2
import numpy as np


def get_gaussian2D(wh, alpha=0.54, eps=1e-6, sigma=None):
    radius = (wh / 2 * alpha).astype("int32")
    if sigma is None:
        sigma = (radius * 2 + 1) / 6
    else:
        sigma = np.array(sigma)

    n, m = radius
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    sx2, sy2 = np.array(sigma) ** 2
    heatmap = np.exp(-(y * y) / (2 * sy2 + eps) - (x * x) / (2 * sx2 + eps))
    heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
    return heatmap


def get_reg_map(wh, value):
    w, h = wh
    if isinstance(value, (list, tuple, np.ndarray)):
        reg_map = np.tile(value, (h, w, 1))
    else:
        reg_map = np.tile(value, (h, w))
    return reg_map


def xywh_to_x1y1x2y2(bboxes):
    if isinstance(bboxes, (list, tuple)):
        bboxes = np.array(bboxes)

    bboxes[..., 2:] += bboxes[..., :2]
    return bboxes


def convert_alpha(alpha, alpha_in_degree):
    return math.radians(alpha + 45) if alpha_in_degree else alpha


def convert_pred_rot_to_alpha(rot):
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def convert_gt_rot_to_alpha(rot):
    idx = rot[:, 0] > rot[:, 1]
    alpha1 = np.arctan2(np.sin(rot[:, 2]), np.cos(rot[:, 2])) + (-0.5 * np.pi)
    alpha2 = np.arctan2(np.sin(rot[:, 3]), np.cos(rot[:, 3])) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def get_dense_locoffset(
    wh,
    center,
    loc_offset,
    loc,
    dim,
    calib,
    trans_output,
    distCoeffs,
    undistort_2dcenter,
):  # noqa
    w, h = wh
    radius = (w // 2, h // 2)
    n, m = radius
    x, y = int(center[0]), int(center[1])

    focal_x = calib[0][0]
    focal_y = calib[1][1]
    cx = calib[0][2]
    cy = calib[1][2]

    y_grid = np.arange(y - m, y + m + 1)
    x_grid = np.arange(x - n, x + n + 1)
    x_scale_up = trans_output[0, 0]
    y_scale_up = trans_output[1, 1]
    y_offset = trans_output[1, 2]
    if undistort_2dcenter:
        tmp_y = y_grid / y_scale_up - y_offset / y_scale_up
        tmp_x = x_grid / x_scale_up
        x_range = tmp_x.shape[0]
        x_, y_ = np.meshgrid(tmp_x, tmp_y)
        _xy = np.stack((x_, y_), axis=-1)
        _xy = _xy.reshape(-1, 2)
        tmp_xy = np.expand_dims(_xy, axis=0)
        proj_p = cv2.undistortPoints(
            tmp_xy,
            np.array(calib[:3, :3]),
            np.array(distCoeffs),
            None,
            np.array(calib[:3, :3]),
        )  # noqa
        proj_p = proj_p.reshape(-1, 2)
        proj_p = proj_p.astype(np.int)
        tmp_x = proj_p[:x_range, 0].reshape(-1)
        tmp_y = proj_p[::x_range, 1].reshape(-1)
        y_reg = loc[1] - (tmp_y - cy) / focal_y * loc[2] - dim[0] / 2.0
        x_reg = loc[0] - (tmp_x - cx) / focal_x * loc[2]
    else:
        # - dim[0] / 2.0 的目的是从3dbox的底面中心点转换到3dbox的几何中心点    zmj
        y_reg = (
            loc[1]
            - (y_grid / y_scale_up - y_offset / y_scale_up - cy)
            / focal_y
            * loc[2]
            - dim[0] / 2.0
        )
        x_reg = loc[0] - (x_grid / x_scale_up - cx) / focal_x * loc[2]
    y_reg[m] = loc_offset[1]
    x_reg[n] = loc_offset[0]
    xv, yv = np.meshgrid(x_reg, y_reg)
    locreg_map = np.concatenate(
        [xv[:, :, np.newaxis], yv[:, :, np.newaxis]], axis=-1
    )
    return locreg_map


def get_dense_rot(
    wh, center, alpha, loc, rot_y, calib, trans_output, alpha_in_degree
):
    w, h = wh
    radius = (w // 2, h // 2)
    n, m = radius
    x = int(center[0])

    focal_x = calib[0][0]
    cx = calib[0][2]

    x_grid = np.arange(x - n, x + n + 1)
    x_scale_up = trans_output[0, 0]
    x_grid = x_grid / x_scale_up
    alpha_x = rot_y2alpha(rot_y, x_grid, cx, focal_x)
    alpha_x = convert_alpha(alpha_x, alpha_in_degree)
    alpha_x[n] = alpha

    alpha_x = np.tile(alpha_x, (m * 2 + 1, 1))

    rotbin_map = np.zeros(alpha_x.shape + (2,))
    rotres_map = np.zeros(alpha_x.shape + (2,))
    mask = (alpha_x < np.pi / 6.0) | (alpha_x > 5 * np.pi / 6.0)
    inds = np.where(mask) + (0,)
    rotbin_map[inds] = 1
    rotres_map[inds] = alpha_x[mask] - (-0.5 * np.pi)
    mask = (alpha_x > -np.pi / 6.0) | (alpha_x < -5 * np.pi / 6.0)
    inds = np.where(mask) + (1,)
    rotbin_map[inds] = 1
    rotres_map[inds] = alpha_x[mask] - (0.5 * np.pi)
    return rotbin_map, rotres_map


def rot_y2alpha(rot_y, x, cx, fx):
    alpha = rot_y - np.arctan2(x - cx, fx)
    alpha[alpha > np.pi] = alpha[alpha > np.pi] - 2 * np.pi
    alpha[alpha < -np.pi] = alpha[alpha < -np.pi] + 2 * np.pi

    return alpha


def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by (alpha + theta - 180).

    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    rot_y[rot_y > np.pi] -= 2 * np.pi
    rot_y[rot_y < -np.pi] += 2 * np.pi
    return rot_y


def convert_pred_rot_to_alpha_simplified(rot):
    alpha = np.arctan2(rot[:, 0], rot[:, 1])
    alpha -= np.pi / 2
    return alpha


def unproject_2d_to_3d(pt_2d, depth, P):
    # pts_2d: 2
    # depth: 1
    # P: 3 x 4
    # return: 3
    z = depth - P[2, 3]
    x = (pt_2d[:, 0:1] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:, 1:2] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
    pt_3d = np.concatenate([x, y, z], axis=-1)

    return pt_3d


def get_location_rotation(center, alpha, dim, depth, calib):
    locations = unproject_2d_to_3d(center, depth, calib)
    locations[:, 1] += dim[:, 0] / 2
    rotation_y = alpha2rot_y(alpha, center[:, 0], calib[0, 2], calib[0, 0])
    return locations, rotation_y


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=None,
    inv=0,
):
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    pt = np.array(pt)
    new_pt = np.concatenate([pt, np.ones([pt.shape[0], 1])], axis=1).T
    new_pt = np.dot(t, new_pt).T
    return new_pt[:, :2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def image_transform(img, input_wh, keep_res, shift=None):
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)

    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
    if keep_res:
        size = np.array(input_wh, dtype=np.int32)
    else:
        size = np.array([width, height], dtype=np.int32)

    trans_input = get_affine_transform(
        center, size, 0, input_wh, shift=shift
    )  # noqa

    inp = cv2.warpAffine(
        img, trans_input, tuple(input_wh), flags=cv2.INTER_LINEAR
    )

    trans_matrix = {"center": center, "size": size, "trans_input": trans_input}
    return inp, trans_matrix


class Object3d(object):
    """3d object label."""

    def __init__(self, dic):
        assert isinstance(dic, dict)
        # extract label, truncation, occlusion
        self.type = dic["category_id"]  # 'Car', 'Pedestrian', ...
        # self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            0
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = dic["alpha"]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = dic["bbox"][0]  # left
        self.ymin = dic["bbox"][1]  # top
        self.xmax = dic["bbox"][0] + dic["bbox"][2]  # right
        self.ymax = dic["bbox"][1] + dic["bbox"][3]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = dic["dim"][0]  # box height
        self.w = dic["dim"][1]  # box width
        self.length = dic["dim"][2]  # box length (in meters)
        self.t = dic["location"]  # location (x,y,z) in camera coord.
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.ry = dic["rotation_y"]

        self.is_ignore = None

    def estimate_diffculty(self):
        """Estimate difficulty to detect the object as defined in kitti website."""  # noqa
        # height of the bounding box
        bb_height = np.abs(self.xmax - self.xmin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif (
            bb_height >= 25
            and self.occlusion in [0, 1]
            and self.truncation <= 0.30
        ):  # noqa
            return "Moderate"
        elif (
            bb_height >= 25
            and self.occlusion in [0, 1, 2]
            and self.truncation <= 0.50  # noqa
        ):
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project_to_image(pts_3d, P, dist_coeff=None):
    """Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    if dist_coeff is not None:
        rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
        tvec = np.zeros(shape=(3, 1), dtype=np.float32)
        dist_coeff = np.array(dist_coeff)
        image_pts = cv2.projectPoints(
            pts_3d[:, :3], np.array(rvec), tvec, P[:, :3], dist_coeff
        )[0]
        pts_2d = np.squeeze(image_pts)
    else:
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P, distCoeffs=None):
    """Take an object and a projection matrix (P).

    And projects the 3d bounding box into the image plane.

    Returns:
        corners_2d: (8,2) array in left image coord.
        corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    length = obj.length
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [
        length / 2,
        length / 2,
        -length / 2,
        -length / 2,
        length / 2,
        length / 2,
        -length / 2,
        -length / 2,
    ]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P, distCoeffs)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def draw_projected_box3d(image, points, color=(0, 255, 0), thickness=2):
    """Draw 3d bounding box in image.

    points: (8,3) array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    points = points.astype(np.int32)
    for k in range(0, 4):
        # Ref:
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html  # noqa
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(
            image,
            (points[i, 0], points[i, 1]),
            (points[j, 0], points[j, 1]),
            color,
            thickness,
        )
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(
            image,
            (points[i, 0], points[i, 1]),
            (points[j, 0], points[j, 1]),
            color,
            thickness,
        )

        i, j = k, k + 4
        cv2.line(
            image,
            (points[i, 0], points[i, 1]),
            (points[j, 0], points[j, 1]),
            color,
            thickness,
        )
    return image
