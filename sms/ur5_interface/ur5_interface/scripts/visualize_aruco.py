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
        return None,None,None

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
    return None,None,None

if __name__ == "__main__":
    port_num = 0
    ur = UR5Robot(gripper=1)
    from ur5_interface.RAFT_Stereo.raftstereo.zed_stereo import Zed
    
    wrist_zed_id = 16347230
    extrinsic_zed_id = 22008760
    
    zed_mini = Zed(wrist_zed_id)
    extrinsic_zed = Zed(extrinsic_zed_id)
    
    H_WRIST = RigidTransform(translation=[0, 0, 0]).as_frames("rob", "rob")
    ur.set_tcp(H_WRIST)
    
    ur.start_teach()
    input("Ready to take Aruco pictures")
    rvecs_zed_mini = []
    tvecs_zed_mini = []
    rvecs_zed_extrinsic = []
    tvecs_zed_extrinsic = []
    while True:
        img_zed_mini = zed_mini.get_frame()[0]
        img_zed_mini = img_zed_mini.detach().cpu().numpy()
        
        img_zed_extrinsic = extrinsic_zed.get_frame()[0]
        img_zed_extrinsic = img_zed_extrinsic.detach().cpu().numpy()
        
        k_zed_mini = zed_mini.get_K()
        k_zed_extrinsic = extrinsic_zed.get_K()
        # k = np.array(
        # [[1129.551243094171, 0., 966.9812584534886],
        # [0., 1124.5757372398643, 556.5882496966005],
        # [0., 0., 1.]]
        # )
        d = np.array([0.0, 0, 0, 0, 0])
        # tag dimensions
        l = 0.105
    
        output_zed_mini, rvec_zed_mini, tvec_zed_mini = pose_estimation(img_zed_mini, cv2.aruco.DICT_ARUCO_ORIGINAL, k_zed_mini, d, l, True)
        output_zed_extrinsic, rvec_zed_extrinsic, tvec_zed_extrinsic = pose_estimation(img_zed_extrinsic, cv2.aruco.DICT_ARUCO_ORIGINAL, k_zed_extrinsic, d, l, False)
        if(output_zed_mini is not None and output_zed_extrinsic is not None):
            rvecs_zed_mini.append(rvec_zed_mini.reshape(-1,))
            tvecs_zed_mini.append(tvec_zed_mini.reshape(-1,))
            rvecs_zed_extrinsic.append(rvec_zed_extrinsic.reshape(-1,))
            tvecs_zed_extrinsic.append(tvec_zed_extrinsic.reshape(-1,))
            print("ZED Mini Rvec: " + str(np.var(np.array(rvecs_zed_mini),axis=0)))
            print("ZED Mini Tvec: " + str(np.var(np.array(tvecs_zed_mini),axis=0)))
            print("ZED Extrinsic Rvec: " + str(np.var(np.array(rvecs_zed_extrinsic),axis=0)))
            print("ZED Extrinsic Tvec: " + str(np.var(np.array(tvecs_zed_extrinsic),axis=0)))
