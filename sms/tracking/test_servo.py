from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
import numpy as np
import time
def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)

def circle_transforms(end_effector_transform, radius, N):
    """
    Generate N transformation matrices representing positions along a circle on the XY plane.
    
    Parameters:
    - end_effector_transform: Initial 4x4 transform of the end effector (numpy array).
    - radius: Radius of the desired circle.
    - N: Number of points along the circle.
    
    Returns:
    - List of 4x4 transformation matrices for each point along the circle.
    """
    # Extract the current translation of the end-effector from the initial transform
    current_translation = end_effector_transform.translation
    
    # Initialize an empty list to store the transformation matrices
    transforms = []
    
    # Define the angular positions for the N points along the circle (full circle = 2 * pi radians)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    for angle in angles:
        # Compute the new position on the circle
        x = current_translation[0] + radius * np.cos(angle)
        y = current_translation[1] + radius * np.sin(angle)
        z = current_translation[2]  # z stays the same since it's on the XY-plane
        
        # Create a new transformation matrix (keeping the rotation from the initial transform)
        new_transform = end_effector_transform.copy()
        new_transform.translation = np.array([x,y,z])
        
        # Append the new transform to the list
        transforms.append(new_transform)
    
    return transforms

robot = UR5Robot(gripper=1)
clear_tcp(robot)
pose1 = RigidTransform(rotation=np.array([[ 0.99999042, -0.00215535,  0.00380905],[-0.00201319, -0.99931541, -0.03694121],[ 0.00388606,  0.03693319, -0.99931018]]), translation=np.array([-0.20660272, -0.5226325 ,  0.41952989]), from_frame='unassigned', to_frame='world')
pose2 = RigidTransform(rotation=np.array([[ 0.99999077, -0.00207156,  0.00376287],[-0.0019313 , -0.99931699, -0.03690289],[ 0.00383675,  0.03689528, -0.99931177]]), translation=np.array([ -0.10141581, -0.49689814,  0.4194795 ]), from_frame='unassigned', to_frame='world')
pose3 = RigidTransform(rotation=np.array([[ 0.99999077, -0.00207156,  0.00376287],[-0.0019313 , -0.99931699, -0.03690289],[ 0.00383675,  0.03689528, -0.99931177]]), translation=np.array([ 0, -0.49689814,  0.4194795 ]), from_frame='unassigned', to_frame='world')
pose4 = RigidTransform(rotation=np.array([[ 0.99999077, -0.00207156,  0.00376287],[-0.0019313 , -0.99931699, -0.03690289],[ 0.00383675,  0.03689528, -0.99931177]]), translation=np.array([ 0.05, -0.49689814,  0.4194795 ]), from_frame='unassigned', to_frame='world')
pose5 = RigidTransform(rotation=np.array([[ 0.99999077, -0.00207156,  0.00376287],[-0.0019313 , -0.99931699, -0.03690289],[ 0.00383675,  0.03689528, -0.99931177]]), translation=np.array([ 0.10141581, -0.49689814,  0.4194795 ]), from_frame='unassigned', to_frame='world')
i = 0
robot.move_pose(pose1,vel=0.3,acc=0.75)
time.sleep(1)
while True:
    if(i % 5 == 0):
        robot.move_pose(pose1,asyn=True,vel=1.0,acc=0.1)
        time.sleep(0.2)
    elif(i % 5 == 1):
        robot.move_pose(pose2,asyn=True,vel=1.0,acc=0.1)
        time.sleep(0.2)
    elif(i % 5 == 2):
        robot.move_pose(pose3,asyn=True,vel=1.0,acc=0.1)
        time.sleep(0.2)
    elif(i % 5 == 3):
        robot.move_pose(pose4,asyn=True,vel=1.0,acc=0.1)
        time.sleep(0.2)
    elif(i % 5 == 4):
        while True:
            robot.move_pose(pose5,asyn=True,vel=1.0,acc=0.1)
            time.sleep(0.2)
        print("DONE")
        exit()
    i += 1
import pdb
pdb.set_trace()
current_pose = robot.get_pose()
tf_list = circle_transforms(current_pose, 0.1, 4)
for pose_tf in tf_list:
    robot.servo_pose(target=pose_tf)
    exit()

