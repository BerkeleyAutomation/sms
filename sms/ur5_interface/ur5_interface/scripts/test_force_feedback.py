from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import numpy as np

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
robot = UR5Robot(gripper=1)
clear_tcp(robot)
robot.gripper.close()


