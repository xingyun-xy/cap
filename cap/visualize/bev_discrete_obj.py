import cv2
import numpy as np


def vcs2bev_coord(pt, vcs_range, m_perpixel):
    pt = np.array(pt).reshape((-1, 2))
    u = (vcs_range[3] - pt[:, 1]) / m_perpixel[1]
    v = (vcs_range[2] - pt[:, 0]) / m_perpixel[0]
    u = u.reshape((-1, 1))
    v = v.reshape((-1, 1))
    return np.hstack([u, v]).astype(np.int32)


def get_vertices_from_vcs_box(wh, ct, yaw):
    w, h = wh
    ctx, cty = ct

    p0 = [0.5 * w, 0.5 * h, 1]
    p1 = [0.5 * w, -0.5 * h, 1]
    p2 = [-0.5 * w, -0.5 * h, 1]
    p3 = [-0.5 * w, 0.5 * h, 1]
    points = np.array([p0, p1, p2, p3])
    R = np.array(
        [
            [np.cos(yaw), np.sin(yaw), ctx],
            [-np.sin(yaw), np.cos(yaw), cty],
            [0, 0, 1],
        ]
    )
    points = R.dot(points.T).T
    return points[:, :2]


def vis_disc_obj(
    img,
    vcs_loc,
    vcs_dim,
    yaw,
    vcs_range,
    m_perpixel,
    color=(0, 255, 0),
    line_thicks=3,
):
    bev_loc = vcs2bev_coord(vcs_loc, vcs_range, m_perpixel).flatten()
    bev_dim = [vcs_dim[0] / m_perpixel[0], vcs_dim[1] / m_perpixel[1]]

    box = get_vertices_from_vcs_box(bev_dim, bev_loc, yaw + np.pi / 2)
    box = np.int0(box)
    cv2.line(img, tuple(box[0]), tuple(box[1]), color, line_thicks)
    cv2.line(img, tuple(box[1]), tuple(box[2]), color, line_thicks)
    cv2.line(img, tuple(box[2]), tuple(box[3]), color, line_thicks)
    cv2.line(img, tuple(box[3]), tuple(box[0]), color, line_thicks)
