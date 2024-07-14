img_dir = "img"

import glob
import os
import numpy as np
import json
import math
from autolab_core import RigidTransform

def get_rot_matrix(angle, axis):
    cos, sin = math.cos, math.sin
    if axis == 'Z':
        rot_matrix = np.array([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
            ])
    elif axis == 'Y':
        rot_matrix = np.array([
            [cos(angle), 0, sin(angle)],
            [0, 1, 0],
            [-sin(angle), 0, -cos(angle)]
            ])
    elif axis == 'X':
        rot_matrix = np.array([
            [1, 0, 0],
            [0, cos(angle), -sin(angle)],
            [0, sin(angle), cos(angle)]
            ])
    else:
        raise AssertionError("axis must be 'X', 'Y', or 'Z'")

    return rot_matrix

def save_json(data, filename):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def save_poses(poses_dir, intrinsics_dict, cam_to_wrist):
    extrinsics_dicts = []
    print(os.listdir(poses_dir))
    num_files = len(os.listdir(poses_dir))
    print(num_files)
    # cam_to_wrist = RigidTransform.load("/home/gogs/Desktop/gogs/T_webcam_wrist.tf")
    cam_to_wrist._from_frame = "world"
    cam_to_wrist._to_frame = "world"
    for i in range(num_files):
        transform_mat = np.loadtxt(os.path.join(poses_dir, f"{i:03d}.txt"))
        extrinsics_dicts.append({
            "file_path": os.path.join(img_dir, f"{i:03d}.png"),
            "transform_matrix": transform_mat.tolist()
            })
        
    intrinsics_dict["frames"] = extrinsics_dicts
    intrinsics_dict["ply_file_path"] = "sparse_pc.ply"
    save_json(intrinsics_dict, os.path.join(poses_dir, "..", "transforms.json"))
