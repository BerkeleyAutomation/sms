from ur5py import UR5Robot
import cv2
from ur5_interface.cameras.zed import ZedImageCapture
from ur5_interface.capture.capture_utils import estimate_cam2rob
import time
import numpy as np
from autolab_core import CameraIntrinsics, PointCloud, RigidTransform, Point
import matplotlib.pyplot as plt
from ur5_interface.capture.capture_utils import _generate_hemi
import subprocess
import pyzed.sl as sl
from tqdm import tqdm
import pdb
import os
from scipy.spatial.transform import Rotation as R
import pathlib

# script_directory = pathlib.Path(__file__).parent.resolve()
# calibration_save_path = str(script_directory) + '/../calibration_outputs'
calibration_save_path = "/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs"
wrist_to_zed_mini_path = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_zed_mini.tf'

if not os.path.exists(calibration_save_path):
    os.makedirs(calibration_save_path)

def find_corners(img, sx, sy, SB=True):
    """
    sx and sy are the number of internal corners in the chessboard
    """
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((sx * sy, 3), np.float32)
    objp[:, :2] = np.mgrid[0:sx, 0:sy].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # create images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    if SB:
        ret, corners = cv2.findChessboardCornersSB(gray, (sx, sy), None)
    else:
        ret, corners = cv2.findChessboardCorners(gray, (sx, sy), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        if corners is not None:
            return corners.squeeze()
    return None


def rvec_tvec_to_transform(rvec, tvec,to_frame):
    """
    convert translation and rotation to pose
    """
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame="tag", to_frame=to_frame)

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)

def pose_estimation(
    frame,
    aruco_dict_type,
    matrix_coefficients,
    distortion_coefficients,
    tag_length,
    visualize=False,
):
    """
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict,parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    if len(corners) == 0 or len(ids) == 0:
        print("No markers found")
        return None

    # If markers are detected
    rvec, tvec = None, None
    if len(corners) > 0:
        obj_points = np.array([[-tag_length / 2, tag_length / 2, 0],
                              [tag_length / 2, tag_length / 2, 0],
                              [tag_length / 2, -tag_length / 2, 0],
                              [-tag_length / 2, -tag_length / 2, 0]], dtype=np.float32)
        for i in range(0, len(ids)):
            img_points = corners[i].reshape((4, 2))
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, matrix_coefficients, distortion_coefficients)
            if success:
                frame_3 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Draw Axis
                frame_3 = cv2.drawFrameAxes(
                    frame_3, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1
                )
                if visualize:
                    cv2.imshow("img", frame_3)
                    cv2.waitKey(0)
                return frame, rvec, tvec
    return None

def get_hemi_translations(
    phi_min, phi_max, theta_min, theta_max, table_center, phi_div, theta_div, R
):
    sin, cos = lambda x: np.sin(np.deg2rad(x)), lambda x: np.cos(np.deg2rad(x))
    rel_pos = np.zeros((phi_div * theta_div, 3))
    for i, phi in enumerate(np.linspace(phi_min, phi_max, phi_div)):
        tmp_pose = []
        for j, theta in enumerate(np.linspace(theta_min, theta_max, theta_div)):
            tmp_pose.append(
                np.array(
                    [R * sin(phi) * cos(theta), R * sin(phi) * sin(theta), R * cos(phi)]
                )
            )
        if i % 2 == 1:
            tmp_pose.reverse()
        for k, pose in enumerate(tmp_pose):
            rel_pos[i * theta_div + k] = pose

    return rel_pos + table_center

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


def register_webcam():
    port_num = 0
    ur = UR5Robot(gripper=1)
    clear_tcp(ur)

    home_joints = np.array([0.30870315432548523, -1.2771266142474573, -1.5955479780780237, -1.754920784627096, 1.5260951519012451, 0.2983420491218567])
    ur.move_joint(home_joints,vel=1.0,acc=0.1)
    from ur5_interface.RAFT_Stereo.raftstereo.zed_stereo import Zed
    
    wrist_zed_id = 16347230
    extrinsic_zed_id = 22008760
    
    zed_mini = Zed(wrist_zed_id)
    extrinsic_zed = Zed(extrinsic_zed_id, is_res_1080=True)

    teach_mode = False
    saved_joints = []

    H_WRIST = RigidTransform(translation=[0, 0, 0]).as_frames("rob", "rob")
    ur.set_tcp(H_WRIST)
    H_chess_cams = []
    H_rob_worlds = []
    world_to_zed_extrinsic_rvecs = []
    world_to_zed_extrinsic_tvecs = []
    
    world_to_wrists = []
    zed_mini_to_arucos = []
    zed_extrinsic_to_arucos = []
        
    center = np.array((0, -0.5, 0))
    trajectory_path = pathlib.Path(calibration_save_path + "/calibrate_extrinsics_trajectory.npy")
    traj = None
    automatic_path = False
    if trajectory_path.exists() and not teach_mode:
        traj = np.load(trajectory_path)
        automatic_path = True
    else:
        num_poses = input("How many poses do you want to save?")
        num_poses = int(num_poses)
        traj = [0] * num_poses
    # angle range from top of sphere
    # phi_min, phi_max = 65, 20
    # theta_min, theta_max = 110, -110
    # phi_div, theta_div = 3, 10
    # table_center = np.array([0.48666, -0.0104, -0.170])
    # radius = 0.3
    # translations = get_hemi_translations(
    #     phi_min, phi_max, theta_min, theta_max, table_center, phi_div, theta_div, radius
    # )
    # rotations = [point_at(translation, table_center) for translation in translations]
    # dummy_cam_to_wrist = RigidTransform(rotation=np.array([[-1,0,0],
    #                                                        [0,-1,0],
    #                                                        [0,0,1]]),translation=np.array([0,-0.1365,-0.0137]),from_frame='wrist',to_frame='cam')
    # poses = [
    #     RigidTransform(rotations[i], translations[i], from_frame="cam")
    #     * dummy_cam_to_wrist
    #     for i in range(len(translations))
    # ]
    # for i, pose in enumerate(tqdm(poses)):
    for p in tqdm(traj):
        if not automatic_path:
            ur.start_teach()  
            input("Enter to take picture")
        else:
            ur.move_joint(p,vel=1.0,acc=0.1)
            time.sleep(0.5)
        img_zed_mini = zed_mini.get_frame()[0]
        img_zed_mini = img_zed_mini.detach().cpu().numpy()
        
        img_zed_extrinsic = extrinsic_zed.get_frame()[0]
        img_zed_extrinsic = img_zed_extrinsic.detach().cpu().numpy()
        H_rob_world = ur.get_pose()
        print("Robot joints: " + str(ur.get_joints()))
        k_zed_mini = zed_mini.get_K()
        k_zed_extrinsic = extrinsic_zed.get_K()
        # k = np.array(
        # [[1129.551243094171, 0., 966.9812584534886],
        # [0., 1124.5757372398643, 556.5882496966005],
        # [0., 0., 1.]]
        # )
        d = np.array([0.0, 0, 0, 0, 0])
        # tag dimensions
        l = 0.170  # 0.1558

        out_zed_mini = None
        out_zed_extrinsic = None
        if automatic_path:
            visualize_zed_mini = teach_mode
            out_zed_mini = pose_estimation(img_zed_mini, cv2.aruco.DICT_6X6_50, k_zed_mini, d, l, visualize_zed_mini)
            out_zed_extrinsic = pose_estimation(img_zed_extrinsic, cv2.aruco.DICT_6X6_50, k_zed_extrinsic, d, l, False)
            if(out_zed_mini is not None):
                output_zed_mini, rvec_zed_mini, tvec_zed_mini = out_zed_mini
                zed_mini_to_aruco = rvec_tvec_to_transform(rvec_zed_mini, tvec_zed_mini,to_frame="zed_mini")
                world_to_wrist = H_rob_world.as_frames("wrist","world")
                world_to_wrists.append(world_to_wrist)
                zed_mini_to_arucos.append(zed_mini_to_aruco)
                H_chess_cams.append(zed_mini_to_aruco.as_frames("cb", "cam"))
                H_rob_worlds.append(H_rob_world.as_frames("rob", "world"))
            
            if(out_zed_extrinsic is not None):
                output_zed_extrinsic, rvec_zed_extrinsic, tvec_zed_extrinsic = out_zed_extrinsic
                zed_extrinsic_to_aruco = rvec_tvec_to_transform(rvec_zed_extrinsic,tvec_zed_extrinsic,to_frame="zed_extrinsic")
                zed_extrinsic_to_arucos.append(zed_extrinsic_to_aruco)
        else:
            while out_zed_mini is None:
                out_zed_mini = pose_estimation(img_zed_mini, cv2.aruco.DICT_6X6_50, k_zed_mini, d, l, True)
                out_zed_extrinsic = pose_estimation(img_zed_extrinsic, cv2.aruco.DICT_6X6_50, k_zed_extrinsic, d, l, False)
                if out_zed_mini is None:
                    input("Enter to take picture")
                    img_zed_mini = zed_mini.get_frame()[0]
                    img_zed_mini = img_zed_mini.detach().cpu().numpy()
                    
                    img_zed_extrinsic = extrinsic_zed.get_frame()[0]
                    img_zed_extrinsic = img_zed_extrinsic.detach().cpu().numpy()
                    H_rob_world = ur.get_pose()
                    print("Robot joints: " + str(ur.get_joints()))
                    k_zed_mini = zed_mini.get_K()
                    k_zed_extrinsic = extrinsic_zed.get_K()
                    # k = np.array(
                    # [[1129.551243094171, 0., 966.9812584534886],
                    # [0., 1124.5757372398643, 556.5882496966005],
                    # [0., 0., 1.]]
                    # )
                    d = np.array([0.0, 0, 0, 0, 0])
                    # tag dimensions
                    l = 0.170  # 0.1558
                    
            if(out_zed_mini is not None):
                output_zed_mini, rvec_zed_mini, tvec_zed_mini = out_zed_mini
                zed_mini_to_aruco = rvec_tvec_to_transform(rvec_zed_mini, tvec_zed_mini,to_frame="zed_mini")
                world_to_wrist = H_rob_world.as_frames("wrist","world")
                world_to_wrists.append(world_to_wrist)
                zed_mini_to_arucos.append(zed_mini_to_aruco)
                H_chess_cams.append(zed_mini_to_aruco.as_frames("cb", "cam"))
                H_rob_worlds.append(H_rob_world.as_frames("rob", "world"))
                saved_joints.append(ur.get_joints())
            
            if(out_zed_extrinsic is not None):
                output_zed_extrinsic, rvec_zed_extrinsic, tvec_zed_extrinsic = out_zed_extrinsic
                zed_extrinsic_to_aruco = rvec_tvec_to_transform(rvec_zed_extrinsic,tvec_zed_extrinsic,to_frame="zed_extrinsic")
                zed_extrinsic_to_arucos.append(zed_extrinsic_to_aruco)
            
    if(teach_mode):
        np.save(calibration_save_path + "/calibrate_extrinsics_trajectory.npy",np.array(saved_joints))
      
    H_cam_rob, H_chess_world = estimate_cam2rob(H_chess_cams, H_rob_worlds)
    # remove the pre-specified wrist transform
    H_cam_rob = H_WRIST * H_cam_rob
    print("Estimated cam2rob:")
    print(H_cam_rob)
    print()
    print(H_chess_world)
    wrist_to_zed_mini = H_cam_rob
    if "n" not in input("Save? [y]/n"):
        H_cam_rob.to_frame = 'wrist'
        H_cam_rob.from_frame = 'zed_mini'
        H_cam_rob.save(calibration_save_path + "/wrist_to_zed_mini.tf")
 
    zed_extrinsic_to_aruco_translations = []
    zed_extrinsic_to_aruco_eulers = []
    for zed_extrinsic_to_aruco in zed_extrinsic_to_arucos:
        zed_extrinsic_to_aruco_translations.append(zed_extrinsic_to_aruco.translation)
        zed_extrinsic_to_aruco_eulers.append(R.from_matrix(zed_extrinsic_to_arucos[0].rotation).as_euler("xyz"))
    mean_zed_extrinsic_to_aruco_translation = np.mean(zed_extrinsic_to_aruco_translations,axis=0)
    mean_zed_extrinsic_to_aruco_euler = np.mean(zed_extrinsic_to_aruco_eulers,axis=0)
    mean_zed_extrinsic_to_aruco_rotation = R.from_euler("xyz",mean_zed_extrinsic_to_aruco_euler).as_matrix()
    zed_extrinsic_to_aruco = RigidTransform(rotation=mean_zed_extrinsic_to_aruco_rotation,translation=mean_zed_extrinsic_to_aruco_translation,from_frame="tag",to_frame='zed_extrinsic')
    for(world_to_wrist,zed_mini_to_aruco) in zip(world_to_wrists,zed_mini_to_arucos):  
        world_to_zed_extrinsic = world_to_wrist * wrist_to_zed_mini * zed_mini_to_aruco * zed_extrinsic_to_aruco.inverse()
        world_to_zed_extrinsic_rvec,_ = cv2.Rodrigues(world_to_zed_extrinsic.rotation)
        world_to_zed_extrinsic_tvec = world_to_zed_extrinsic.translation
        world_to_zed_extrinsic_rvecs.append(world_to_zed_extrinsic_rvec)
        world_to_zed_extrinsic_tvecs.append(world_to_zed_extrinsic_tvec)
    world_to_zed_extrinsic_translation = np.mean(np.array(world_to_zed_extrinsic_tvecs),axis=0)
    world_to_zed_extrinsic_rotation,_ = cv2.Rodrigues(np.mean(np.array(world_to_zed_extrinsic_rvecs),axis=0))
    world_to_zed_extrinsic_rigid_tf = RigidTransform(rotation=world_to_zed_extrinsic_rotation,translation=world_to_zed_extrinsic_translation,from_frame="zed_extrinsic",to_frame="world")
    print("Estimated cam2rob:")
    print(world_to_zed_extrinsic_rigid_tf)
    if "n" not in input("Save? [y]/n"):
        world_to_zed_extrinsic_rigid_tf.save(calibration_save_path + "/world_to_extrinsic_zed.tf")


if __name__ == "__main__":
    register_webcam()
