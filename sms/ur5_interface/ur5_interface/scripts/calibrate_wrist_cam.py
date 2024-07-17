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
import pathlib

# script_directory = pathlib.Path(__file__).parent.resolve()
# calibration_save_path = str(script_directory) + '/../calibration_outputs'
calibration_save_path = "/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs"
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


def rvec_tvec_to_transform(rvec, tvec):
    """
    convert translation and rotation to pose
    """
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame="tag", to_frame="cam")


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


def register_webcam():
    MANUAL = True
    port_num = 0
    ur = UR5Robot(gripper=1)
    from ur5_interface.RAFT_Stereo.raftstereo.zed_stereo import Zed

    zed = Zed()

    H_WRIST = RigidTransform(translation=[0, 0, 0]).as_frames("rob", "rob")
    ur.set_tcp(H_WRIST)
    H_chess_cams = []
    H_rob_worlds = []
    center = np.array((0, -0.5, 0))
    trajectory_path = pathlib.Path(calibration_save_path + "/calibrate_extrinsics_trajectory.npy")
    traj = None
    automatic_path = False
    save_joints  = False
    saved_joints = []
    if trajectory_path.exists():
        traj = np.load(trajectory_path)
        automatic_path = True
    else:
        traj = _generate_hemi(
            0.5,
            2,
            5,
            (np.deg2rad(-90), np.deg2rad(90)),
            (np.deg2rad(0), np.deg2rad(20)),
            center,
            center,
            False,
        )
    for p in tqdm(traj):
        if not automatic_path:
            ur.start_teach()
            input("Enter to take picture")
        else:
            ur.move_joint(p,vel=1.5,acc=0.1)
            time.sleep(0.5)
        img = zed.get_frame()[0]
        img = img.detach().cpu().numpy()
        H_rob_world = ur.get_pose()
        k = zed.get_K()
        # k = np.array(
        # [[1129.551243094171, 0., 966.9812584534886],
        # [0., 1124.5757372398643, 556.5882496966005],
        # [0., 0., 1.]]
        # )
        d = np.array([0.0, 0, 0, 0, 0])
        # tag dimensions
        l = 0.105  # 0.1558

        out = None
        # cv2.imwrite("img.png", img)
        while out is None:
            out = pose_estimation(img, cv2.aruco.DICT_ARUCO_ORIGINAL, k, d, l, True)
            if out is None:
                input("Enter to take picture")
                img = zed.get_frame()[0]
                img = img.detach().cpu().numpy()
                H_rob_world = ur.get_pose()
                k = zed.get_K()
                # k = np.array(
                # [[1129.551243094171, 0., 966.9812584534886],
                # [0., 1124.5757372398643, 556.5882496966005],
                # [0., 0., 1.]]
                # )
                d = np.array([0.0, 0, 0, 0, 0])
                # tag dimensions
                l = 0.105  # 0.1558

        output, rvec, tvec = out

        T_cb_camera = rvec_tvec_to_transform(rvec, tvec)
        print("T_cb_camera", T_cb_camera)

        H_chess_cams.append(T_cb_camera.as_frames("cb", "cam"))
        H_rob_worlds.append(H_rob_world.as_frames("rob", "world"))
        if(save_joints):
            saved_joints.append(ur.get_joints())
    if(save_joints):
        np.save(calibration_save_path + "/calibrate_extrinsics_trajectory.npy",np.array(saved_joints))
    H_cam_rob, H_chess_world = estimate_cam2rob(H_chess_cams, H_rob_worlds)
    # remove the pre-specified wrist transform
    H_cam_rob = H_WRIST * H_cam_rob
    print("Estimated cam2rob:")
    print(H_cam_rob)
    print()
    print(H_chess_world)
    if "n" not in input("Save? [y]/n"):
        H_cam_rob.to_frame = 'wrist'
        H_cam_rob.from_frame = 'zed_mini'
        H_cam_rob.save(calibration_save_path + "/wrist_to_zed_mini.tf")


if __name__ == "__main__":
    register_webcam()
