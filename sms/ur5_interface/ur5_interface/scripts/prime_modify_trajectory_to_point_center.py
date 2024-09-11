import numpy as np
from autolab_core import RigidTransform
from tracikpy import TracIKSolver
import pathlib
import viser
import matplotlib.pyplot as plt

def point_at(cam_t, obstacle_t, extra_R=np.eye(3)):
    """
    cam_t: numpy array of 3D position of gripper
    obstacle_t: numpy array of 3D position of location to point camera at
    """
    direction = obstacle_t - cam_t
    z_axis = direction / np.linalg.norm(direction)
    x_axis_dir = -np.cross(np.array((0, 0, 1)), z_axis)
    if np.linalg.norm(x_axis_dir) < 1e-10:
        x_axis_dir = np.array((0, 1, 0))
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)

    # postmultiply the extra rotation to rotate the camera WRT itself
    R = RigidTransform.rotation_from_axes(x_axis, y_axis, z_axis)
    return R

def visualize_poses(server,poses, prefix):
    i = 0
    for pose in poses:
        server.add_frame(prefix + '/frame_' + str(i),axes_length=0.05,axes_radius=0.0025,wxyz=viser.transforms.SO3.from_matrix(pose.rotation).wxyz,position=pose.translation)
        i += 1


# Hardcoded table center
table_center = np.array([0.02315526, 0.52568765, -0.19102996])
wrist_to_cam = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_zed_mini.tf")
calibration_save_path = "/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs"

#Note: URDF path and collision/visual paths are also hardcoded
urdf_path = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/urdf/ur5_robot.urdf'
ur5_solver = TracIKSolver(urdf_path,"base_link","tool0")

trajectory_path = pathlib.Path(calibration_save_path + "/prime_trajectory.npy")
joints = np.load(str(trajectory_path))

og_poses = []
new_poses = []
new_joints = []
i = 0
for joint in joints:
    base_to_wrist_matrix = ur5_solver.fk(joint)
    base_to_wrist = RigidTransform(rotation=base_to_wrist_matrix[:3,:3],translation=base_to_wrist_matrix[:3,3],from_frame="wrist",to_frame="base")
    base_to_cam = base_to_wrist * wrist_to_cam
    og_poses.append(base_to_cam)
    base_to_cam_translation = base_to_cam.translation
    new_base_to_cam_rotation = point_at(cam_t = base_to_cam_translation,obstacle_t=table_center)
    new_base_to_cam = RigidTransform(rotation=new_base_to_cam_rotation,translation=base_to_cam_translation,from_frame="zed_mini",to_frame="base")
    # Keep top down pose the same (which is the first and last pose)
    if(i == 0 or i == len(joints) - 1):
        new_base_to_cam = base_to_cam
    
    new_base_to_wrist = new_base_to_cam * wrist_to_cam.inverse()
    new_joint = ur5_solver.ik(new_base_to_wrist.matrix,qinit = joint,brx=1e-2,bry=1e-2,brz=1e-2)
    if(new_joint is not None):
        new_joints.append(new_joint)
        ik_base_to_wrist = ur5_solver.fk(new_joint)
        ik_base_to_wrist = RigidTransform(rotation=ik_base_to_wrist[:3,:3],translation=ik_base_to_wrist[:3,3],from_frame="wrist",to_frame="base")
        ik_base_to_cam = ik_base_to_wrist * wrist_to_cam
        new_poses.append(ik_base_to_cam)
    i += 1
server = viser.ViserServer()
visualize_poses(server,og_poses,prefix='og_poses')
visualize_poses(server,new_poses,prefix='new_poses')
server.add_point_cloud(name='table_center',points=table_center.reshape(-1,3),colors=np.array([0,0,0]).reshape(-1,3),point_size=0.05,point_shape='rounded')
remove_indices = input("Which ones do you want to remove? Separate with space")
remove_indices_list = [int(x) for x in remove_indices.split(' ')]
indices = sorted(remove_indices_list, reverse=True)
for index in indices:
  if 0 <= index < len(new_joints):
    new_joints.pop(index)
np.save(calibration_save_path + "/prime_centered_trajectory.npy",np.array(new_joints))

