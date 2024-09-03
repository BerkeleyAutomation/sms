import numpy as np
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import matplotlib.pyplot as plt
import time

FILE = "ellipsoid_new"

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)

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
    
robot = UR5Robot(gripper=1)
clear_tcp(robot)

filepath = "/home/lifelong/sms/sms/ur5_interface/ur5_interface/outputs/" + FILE + ".npy"
trajectory = np.load(filepath, allow_pickle=True)
visualize_poses(trajectory)
input("Execute?")


for pose in trajectory:
    print(pose)
    input("Move robot?")
    
    robot.move_pose(pose,vel=0.4,acc=0.1)
    time.sleep(0.5)
    
robot.kill()