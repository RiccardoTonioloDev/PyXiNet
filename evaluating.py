from typing import Literal

import numpy as np
from PIL import Image
import os
from collections import Counter


from Config import Config


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


###################### EVAL DISPARITIES FILE ######################


def eval_disparities_file(env: Literal["HomeLab", "Cluster"]):
    config = Config(env).get_configuration()
    pred_disparities: np.ndarray = np.load(config.disparities_to_use)

    num_samples = 697
    test_files = read_text_lines(config.filenames_file_testing)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(
        test_files, config.data_path
    )

    num_test = len(im_files)
    gt_depths = []
    pred_depths = []
    for t_id in range(num_samples):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        depth = generate_depth_map(
            gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True
        )
        gt_depths.append(depth.astype(np.float32))

        disp_pred = Image.fromarray(pred_disparities[t_id])
        disp_pred = disp_pred.resize(
            (im_sizes[t_id][1], im_sizes[t_id][0]), resample=Image.BILINEAR
        )
        disp_pred = np.array(disp_pred)
        disp_pred = disp_pred * disp_pred.shape[1]

        # need to convert from disparity to depth
        focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
        depth_pred = (baseline * focal_length) / disp_pred
        depth_pred[np.isinf(depth_pred)] = 0

        pred_depths.append(depth_pred)

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    min_depth = 1e-3
    max_depth = 80
    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        garg_crop = True
        eigen_crop = False
        if garg_crop or eigen_crop:
            gt_height, gt_width = gt_depth.shape

            # crop used by Garg ECCV16
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            if garg_crop:
                crop = np.array(
                    [
                        0.40810811 * gt_height,
                        0.99189189 * gt_height,
                        0.03594771 * gt_width,
                        0.96405229 * gt_width,
                    ]
                ).astype(np.int32)
            # crop we found by trial and error to reproduce Eigen NIPS14 results
            elif eigen_crop:
                crop = np.array(
                    [
                        0.3324324 * gt_height,
                        0.91351351 * gt_height,
                        0.0359477 * gt_width,
                        0.96405229 * gt_width,
                    ]
                ).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(
            gt_depth[mask], pred_depth[mask]
        )

    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            "abs_rel", "sq_rel", "rms", "log_rms", "d1_all", "a1", "a2", "a3"
        )
    )
    print(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
            abs_rel.mean(),
            sq_rel.mean(),
            rms.mean(),
            log_rms.mean(),
            d1_all.mean(),
            a1.mean(),
            a2.mean(),
            a3.mean(),
        )
    )


###################### EIGEN EVALUATION UTILS ######################


def read_text_lines(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return lines


def read_file_data(files, data_root):
    gt_files = []
    gt_calib = []
    im_sizes = []
    im_files = []
    cams = []
    num_probs = 0
    for filename in files:
        filename = filename.split()[0]
        splits = filename.split("/")
        camera_id = np.int32(splits[2][-1:])  # 2 is left, 3 is right
        date = splits[0]
        im_id = splits[4][:10]
        file_root = "{}/{}"

        im = filename
        vel = "{}/{}/velodyne_points/data/{}.bin".format(splits[0], splits[1], im_id)

        if os.path.isfile(data_root + im):
            gt_files.append(data_root + vel)
            gt_calib.append(data_root + date + "/")
            im_sizes.append(np.array(Image.open(data_root + im)).shape[:2])
            im_files.append(data_root + im)
            cams.append(2)
        else:
            num_probs += 1
            print("{} missing".format(data_root + im))
    print(num_probs, "files missing")

    return gt_files, gt_calib, im_sizes, im_files, cams


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    arr = list(map(float, value.split(" ")))
                    data[key] = np.array(arr)
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    print("Casting error")

    return data


def get_focal_length_baseline(calib_dir, cam):
    cam2cam = read_calib_file(calib_dir + "calib_cam_to_cam.txt")
    P2_rect = cam2cam["P_rect_02"].reshape(3, 4)
    P3_rect = cam2cam["P_rect_03"].reshape(3, 4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0, 3] / -P2_rect[0, 0]
    b3 = P3_rect[0, 3] / -P3_rect[0, 0]
    baseline = b3 - b2

    if cam == 2:
        focal_length = P2_rect[0, 0]
    elif cam == 3:
        focal_length = P3_rect[0, 0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(
    calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False
):
    # load calibration files
    cam2cam = read_calib_file(calib_dir + "calib_cam_to_cam.txt")
    velo2cam = read_calib_file(calib_dir + "calib_velo_to_cam.txt")
    R = velo2cam["R"]
    T = velo2cam["T"]
    velo2cam = np.hstack((R.reshape(3, 3), T.reshape(3, 1)))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam["R_rect_00"].reshape(3, 3)
    P_rect = cam2cam["P_rect_0" + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = (
        val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    )
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[
        :, 2
    ]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth
