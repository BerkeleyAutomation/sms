import numpy as np
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import matplotlib.pyplot as plt
import os
import sys
import tty
import termios

wrist_to_cam = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_zed_mini.tf")

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def transform_pose(base_to_wrist, i=0, save_dir=None):
    base_to_wrist.from_frame = "wrist"
    base_to_wrist.to_frame = "base"

    # stupid Kush convention
    wrist_to_cam.from_frame = "cam"
    wrist_to_cam.to_frame = "wrist"
    # wrist_to_cam_flipped = cam_to_wrist
    # wrist_to_cam_flipped.from_frame ='cam_flipped'

    # cam_flipped_to_cam = RigidTransform(np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.zeros(3),from_frame='cam',to_frame='cam_flipped')

    cam_to_nerfcam = RigidTransform(
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        np.zeros(3),
        from_frame="nerf_cam",
        to_frame="cam",
    )

    # cam_pose = (base_to_wrist * wrist_to_cam_flipped * cam_flipped_to_cam) * cam_to_nerfcam
    cam_pose = base_to_wrist * wrist_to_cam * cam_to_nerfcam

    return cam_pose.matrix

def get_key():
    """
    Captures a single keypress from the terminal without requiring Enter.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def visualize_poses(poses, radius=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for pose in poses:
        translation = pose.translation
        rotation = pose.rotation
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]

        # Plot the X, Y, Z axes of the frame
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            length=0.01,
            normalize=True,
        )
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            length=0.01,
            normalize=True,
        )
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            length=0.01,
            normalize=True,
        )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()

capture = True
append = True
save_path = "/home/lifelong/sms/sms/ur5_interface/ur5_interface/outputs/ellipsoid_new.npy"

if capture:
    robot = UR5Robot(gripper=1)
    clear_tcp(robot)

    # home_joints = np.array([-1.459527317677633, -1.832590405141012, -0.7605069319354456, 2.585705280303955, -1.4630921522723597, 0.04261291027069092])
    # robot.move_joint(home_joints,vel=1.0,acc=0.1)
    # world_to_wrist = robot.get_pose()
    # world_to_wrist.from_frame = "wrist"
    # world_to_cam = world_to_wrist * wrist_to_cam
    # proper_world_to_wrist = world_to_cam * wrist_to_cam.inverse()

    # robot.move_pose(proper_world_to_wrist,vel=1.0,acc=0.1)
    robot.gripper.open()

    if append and os.path.exists(save_path):
        poses = np.load(save_path, allow_pickle=True).tolist()
        poses = poses[:len(poses)-1]
        breakpoint()
    else:
        poses = []
        
    input("enter to enter freedrive")
    robot.start_teach()

    while True:
        key = get_key()

        if key == ' ':  # Detect spacebar press
            rpose = robot.get_pose()
            print(rpose)
            input("confirm...")
            poses.append(rpose)

        elif key == '\x1b':  # Detect 'Esc' keypress (Escape sequence)
            print("Exiting...")
            break

    input("enter to exit freedrive")
    robot.stop_teach()
else:
    poses = np.load(save_path, allow_pickle=True)

poses_np = np.array(poses)
visualize_poses(poses_np, radius=0.5)

if capture:
    np.save(save_path, poses_np, allow_pickle=True)