import numpy as np
import torch

# import pyvista as pv
import numpy as np
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt

# from sklearn.cluster import HDBSCAN, SpectralClustering
from collections import Counter
from autolab_core import RigidTransform, PointCloud, DepthImage, CameraIntrinsics

import json
import pdb
import os
from pathlib import Path


# threshold to filter in a circle for pts in the gaussian splat or the pointcloud
def extract_xyz_rgb(data_path, num_samples=2000, save_grasp=False, plotter=None):
    with open(os.path.join(data_path, "transforms.json")) as r:
        transform_json = json.load(r)
        print(transform_json)

    cam_to_nerfcam = RigidTransform(
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        np.zeros(3),
        from_frame="zed",
        to_frame="zed",
    )
    breakpoint()
    cam_intrinsics = CameraIntrinsics.load(os.path.join(data_path, "zed.intr"))

    # p = pv.Plotter()
    # p.show_axes()
    if save_grasp:
        cloud = np.zeros((0, 7))
    else:
        cloud = np.zeros((0, 6))
    for frame_dict in tqdm(transform_json["frames"]):
        path = os.path.join(
            data_path, "depth", frame_dict["file_path"].split("/")[-1][:-3] + "npy"
        )
        depth = np.load(path)
        depth_im = DepthImage(depth, frame="zed")

        pnc = depth_im.point_normal_cloud(cam_intrinsics)
        point_cloud = pnc.point_cloud
        robot_transform = RigidTransform(
            *RigidTransform.rotation_and_translation_from_matrix(
                np.array(frame_dict["transform_matrix"])
            )
        )

        robot_transform.from_frame = "zed"
        robot_transform.to_frame = "base"
        point_cloud = (robot_transform * cam_to_nerfcam.inverse()) * point_cloud
        point = point_cloud.data.T

        img_path = data_path + "/" + frame_dict["file_path"]
        rgb = plt.imread(img_path).reshape(-1, 3)
        if save_grasp:
            grasps = np.load(
                f"{data_path}/grasps/{Path(frame_dict['file_path']).stem}.npy"
            ).flatten()[..., None]
            point = np.concatenate((point, rgb, grasps), axis=1)
        else:
            point = np.concatenate((point, rgb), axis=1)

        # Handling single image case
        if num_samples == None:
            return point

        randoms_anywhere = np.random.choice(
            np.arange(point.shape[0]),
            int(num_samples),
            replace=False,
        )
        # tabletop_pts = point.copy()
        # tabletop_pts = np.delete(tabletop_pts, np.where(tabletop_pts[:, 2] < -0.1), axis=0)#-0.2), axis=0)
        # dist = np.linalg.norm(tabletop_pts - np.array([0, 0.65, 0.05]), axis=-1)
        # Karim: deletes any pts not on the table and stuff
        # RADIUS_THRESH = 0.35
        # tabletop_pts = np.delete(tabletop_pts, np.where(dist > RADIUS_THRESH), axis = 0)

        cloud = np.vstack(
            (
                cloud,
                point[randoms_anywhere],
            )
        )

    if plotter is not None:
        plotter.add_points(cloud[:, :3])

    return cloud


# p = pv.Plotter()
# p.show_axes()
def point_cloud_generator_main(data_path):
    point_cloud = extract_xyz_rgb(data_path)
    print(point_cloud.shape)
    np.save(os.path.join(data_path, "pointcloud.npy"), point_cloud)
    return point_cloud


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene", type=str)
    args = argparser.parse_args()
    scene_name = args.scene
    data_path = f"/home/lifelong/sms/sms/ur5_interface/ur5_interface/data/{scene_name}"
    point_cloud = point_cloud_generator_main(data_path)

    import viser
    import time

    server = viser.ViserServer()
    colors = point_cloud[:, 3:].astype(np.uint8)
    server.add_point_cloud(
        f"cloud_{1}",
        10
        * (point_cloud[:, :3] - point_cloud[:, :3].mean())
        / point_cloud[:, :3].std(),
        colors=colors,
        position=(0, 0, 0),
        point_size=0.05,
        point_shape="circle",
    )

    while True:
        time.sleep(10)
