import numpy as np
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import sys
import tty
import termios
import time
wrist_to_cam = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_zed_mini.tf")
trajectory_filepath = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/prime_centered_trajectory.npy'
K_RIGHT = b'\x1b[C'
K_LEFT  = b'\x1b[D'

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
robot = UR5Robot(gripper=1)
clear_tcp(robot)
joints_trajectory = np.load(str(trajectory_filepath))
i = 0
for joints in joints_trajectory:
    import pdb
    pdb.set_trace()
    robot.move_joint(joints,vel=1.0,acc=0.1)
    print(i)
    i += 1
    time.sleep(0.1)
    