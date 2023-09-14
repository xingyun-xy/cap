from collections import defaultdict
from math import atan, pi, sqrt

import cv2
import numpy as np

GlobalDistortMap = defaultdict()


def getUnDistortPoints(points, calib, distCoeffs, img_wh):
    # points's shape must be (n, 2)
    assert points.shape[1] == 2, "points shape must be (n, 2)"
    calib = np.array(calib)[:3, :3]
    distCoeffs = np.array(distCoeffs)
    hash_distcoeffs = distCoeffs.tobytes()
    if hash_distcoeffs not in GlobalDistortMap.keys():
        pt_undistort_lut_ = InitPinHoleUndistortLUT(
            img_wh[0], img_wh[1], calib, distCoeffs
        )
        GlobalDistortMap[hash_distcoeffs] = pt_undistort_lut_
    undist_pts = []
    for i in range(points.shape[0]):
        undist_pt = UndistortPt_bi(
            GlobalDistortMap[hash_distcoeffs], points[i]
        )
        undist_pts.append(undist_pt)
    undist_pts = np.array(undist_pts)
    return undist_pts


# The author of the following code is menghan.pan. undistort points by map.
# we do not only use cv2.undistortPoints to undistort. because
# cv2.undistortPoints may cause outlier. In addition to this,
# the following code is aligned with the software.
def main_func():
    # compare_cpp_py_csv()
    w, h, k_, d_ = init_param()
    pt_undistort_lut_ = InitPinHoleUndistortLUT(w, h, k_, d_)
    # undistort_lut_to_csv(pt_undistort_lut_)
    pts = [
        [50.0, 50.0],
        [500.0, 300.0],
        [150.0, 450.0],
        [2000.0, 1800.0],
        [0.0, 0.0],
        [5.0, 5.0],
        [2047.0, 1279.0],  # noqa
        [2040.0, 1270.0],
        [1.0, 1270.0],
        [50.0, 1260.0],
        [2047.0, 1.0],
        [2040.0, 5.0],
    ]  # noqa

    for pt in pts:
        undist_pt = UndistortPt_bi(pt_undistort_lut_, pt)


def init_param():
    # ar0233-WISSEN
    w = 2048
    h = 1280
    fu = 1095.7528076171875
    fv = 1095.7528076171875
    cu = 1024.2835693359375
    cv = 639.7689208984375
    k = np.array([fu, 0.0, cu, 0.0, fv, cv, 0.0, 0.0, 1.0]).reshape([3, 3])
    distort = np.array(
        [
            9.616952896118164,
            -0.4286363422870636,
            0.00043669503065757453,
            -6.74657931085676e-05,
            -0.6490375995635986,  # noqa
            9.931828498840332,
            2.334935426712036,
            -1.4143507480621338,
        ]
    )
    return w, h, k, distort


# copy from camera.cpp


def InitPinHoleUndistortLUT(w, h, k_, d_):
    fu = k_[0][0]
    # fv = k_[1][1]
    cu = k_[0][2]
    cv = k_[1][2]

    wid = int(w / 2)
    hei = int(h / 2)

    size = hei + 1, wid + 1, 2
    lut = np.zeros(size, dtype=np.float32)

    mat_pt = []
    for y in range(hei):
        for x in range(wid):
            mat_pt.append([x * 2, y * 2])

    # mat_pt = np.array(mat_pt, dtype=np.float32) # 4.5
    mat_pt = np.array(mat_pt, dtype=np.float32).reshape(
        -1, 1, 2
    )  # 3.4.5  # noqa
    mat_ptd = cv2.undistortPoints(mat_pt, k_, d_, P=k_)
    mat_pt = mat_pt.reshape(hei, wid, 2)
    mat_ptd = mat_ptd.reshape(hei, wid, 2)

    angle_thr = 80.0 * pi / 180.0
    offset_pixel = 10
    for y in range(hei):
        pt_lut = lut[y]
        pt_prev = mat_pt[y]
        pt_post = mat_ptd[y]
        # right
        flag = True
        mean_ddx = 0.0
        mean_ddy = 0.0
        start_x = int(cu / 2.0)

        for x in range(start_x, wid):
            x0 = pt_prev[x][0]
            y0 = pt_prev[x][1]
            x1 = pt_post[x][0]
            y1 = pt_post[x][1]

            x_c = abs(x1 - cu)
            y_c = abs(y1 - cv)

            angle = 2.0 * atan(sqrt(x_c * x_c + y_c * y_c) / fu)

            if flag and (angle < angle_thr):
                pt_lut[x][0] = x1 - x0
                pt_lut[x][1] = y1 - y0
            else:
                if flag:
                    for i in range(1, offset_pixel + 1):
                        dx = pt_post[(x - i)][0] - pt_post[(x - i - 1)][0]
                        dy = pt_post[(x - i)][1] - pt_post[(x - i - 1)][1]
                        dx_pre = (
                            pt_post[(x - i - 1)][0] - pt_post[(x - i - 2)][0]
                        )
                        dy_pre = (
                            pt_post[(x - i - 1)][1] - pt_post[(x - i - 2)][1]
                        )
                        mean_ddx += dx - dx_pre
                        mean_ddy += dy - dy_pre
                    mean_ddx = mean_ddx / offset_pixel
                    mean_ddy = mean_ddy / offset_pixel
                flag = False

                x_pre = pt_post[(x - 1)][0]
                y_pre = pt_post[(x - 1)][1]
                x_prep = pt_post[(x - 2)][0]
                y_prep = pt_post[(x - 2)][1]
                dx_pre = x_pre - x_prep
                dy_pre = y_pre - y_prep

                pt_post[x][0] = x_pre + dx_pre + mean_ddx
                pt_post[x][1] = y_pre + dy_pre + mean_ddy
                pt_lut[x][0] = pt_post[x][0] - x0
                pt_lut[x][1] = pt_post[x][1] - y0

        # left
        flag = True
        mean_ddx = 0.0
        mean_ddy = 0.0
        start_x = int(cu / 2.0) - 1
        x = start_x
        while True:
            x0 = pt_prev[x][0]
            y0 = pt_prev[x][1]
            x1 = pt_post[x][0]
            y1 = pt_post[x][1]
            x_c = abs(x1 - cu)
            y_c = abs(y1 - cv)
            angle = 2.0 * atan(sqrt(x_c * x_c + y_c * y_c) / fu)
            if flag and (angle < angle_thr):
                pt_lut[x][0] = x1 - x0
                pt_lut[x][1] = y1 - y0
            else:
                if flag:
                    for i in range(1, offset_pixel + 1):
                        dx = pt_post[x + i][0] - pt_post[x + i + 1][0]
                        dy = pt_post[x + i][1] - pt_post[x + i + 1][1]
                        dx_pre = pt_post[x + i + 1][0] - pt_post[x + i + 2][0]
                        dy_pre = pt_post[x + i + 1][1] - pt_post[x + i + 2][1]
                        mean_ddx += dx - dx_pre
                        mean_ddy += dy - dy_pre
                    mean_ddx = mean_ddx / offset_pixel
                    mean_ddy = mean_ddy / offset_pixel

                flag = False
                x_pre = pt_post[x + 1][0]
                y_pre = pt_post[x + 1][1]
                x_prep = pt_post[x + 2][0]
                y_prep = pt_post[x + 2][1]
                dx_pre = x_pre - x_prep
                dy_pre = y_pre - y_prep
                pt_post[x][0] = x_pre + dx_pre + mean_ddx
                pt_post[x][1] = y_pre + dy_pre + mean_ddy
                pt_lut[x][0] = pt_post[x][0] - x0
                pt_lut[x][1] = pt_post[x][1] - y0
            x = x - 1
            if x < 0:
                break

    lut[hei] = np.copy(lut[hei - 1])
    lut[:, wid, :] = np.copy(lut[:, wid - 1, :])

    return lut


def UndistortPt_bi(pt_undistort_lut_, dist_pt):
    ix = int(dist_pt[0] / 2.0)
    iy = int(dist_pt[1] / 2.0)

    fx = dist_pt[0] / 2.0 - ix
    fy = dist_pt[1] / 2.0 - iy

    if ix < 0:
        ix = 0
        fx = 0.0

    # cols
    if ix > (pt_undistort_lut_.shape[1] - 2):
        ix = pt_undistort_lut_.shape[1] - 2
        fx = 0.0

    if iy < 0:
        iy = 0
        fy = 0.0

    # rows
    if iy > (pt_undistort_lut_.shape[0] - 2):
        iy = pt_undistort_lut_.shape[0] - 2
        fy = 0.0

    ix_1 = ix + 1
    iy_1 = iy + 1

    ptr = pt_undistort_lut_[iy][ix]
    ptr_x1 = pt_undistort_lut_[iy][ix_1]
    ptr_y1 = pt_undistort_lut_[iy_1][ix]
    ptr_x1_y1 = pt_undistort_lut_[iy_1][ix_1]

    # bilinear interpolation
    t1 = 1.0 - fx
    t2 = 1.0 - fy
    x = (
        dist_pt[0]
        + t1 * t2 * ptr[0]
        + t1 * fy * ptr_x1[0]
        + fx * t2 * ptr_y1[0]
        + fx * fy * ptr_x1_y1[0]
    )
    y = (
        dist_pt[1]
        + t1 * t2 * ptr[1]
        + t1 * fy * ptr_x1[1]
        + fx * t2 * ptr_y1[1]
        + fx * fy * ptr_x1_y1[1]
    )

    return [x, y]


def undistort_lut_to_csv(pt_undistort_lut_):
    pt_undistort_lut_ = pt_undistort_lut_.reshape(-1, 1)
    np.savetxt(
        "undistort_lut_py.csv", pt_undistort_lut_, fmt="%.7f", delimiter=","
    )


def compare_cpp_py_csv():
    cpp_csv = np.loadtxt("undistort_lut_cpp.csv")
    py_csv = np.loadtxt("undistort_lut_py.csv")
    diff_csv = cpp_csv - py_csv


if __name__ == "__main__":
    main_func()
