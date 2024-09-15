import numpy as np
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
from sms.tracking.tri_zed import Zed
import pyzed.sl as sl
import time
import cv2
from scipy.spatial.transform import Rotation as R

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

def rvec_tvec_to_transform(rvec, tvec,to_frame):
    """
    convert translation and rotation to pose
    """
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame="tag", to_frame=to_frame)

def get_servo_pose(base_to_ee,base_to_frame_a,base_to_frame_b):
    start_time = time.time()
    thetas = np.linspace(-np.pi,np.pi,60)
    base_to_frame_b_variations = []
    for theta in thetas:
        rotation_tf = RigidTransform(rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]),translation=np.zeros(3),to_frame='object',from_frame='object')
        base_to_frame_b_rotation = base_to_frame_b * rotation_tf
        base_to_frame_b_variations.append(base_to_frame_b_rotation)
    min_angle = 1000
    min_base_to_frame_b = None
    for base_to_frame_b in base_to_frame_b_variations:
        rotation_error_matrix = base_to_frame_b.rotation @ base_to_frame_a.rotation.T
        angle = np.arccos((np.matrix.trace(rotation_error_matrix) - 1) / 2)
        if(angle < min_angle):
            min_angle = angle
            min_base_to_frame_b = base_to_frame_b
    print("Min angle: " + str(min_angle))
    base_to_frame_b = min_base_to_frame_b
    ee_to_object = base_to_ee.inverse() * base_to_frame_a
    new_base_to_ee = base_to_frame_b * ee_to_object.inverse()
    end_time = time.time()
    return new_base_to_ee

robot = UR5Robot(gripper=1)
clear_tcp(robot)
time.sleep(1)
home_joints = np.array([-1.6099513212787073, -1.8723843733416956, -1.9820297400103968, -0.6524918715106409, 1.2875401973724365, 1.21976637840271])
import pdb
pdb.set_trace()
robot.move_joint(home_joints,vel=0.5,acc=0.1)
camera_tf = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")

tool_handle_wxyz = np.array([-0.03003284,  0.5806886 ,  0.81356361,  0.00108548])
tool_handle_position = np.array([-0.29395993, -0.54714222, -0.07673486])

extrinsic_zed_id = 22008760
zed = Zed(cam_id=extrinsic_zed_id, is_res_1080=True) # Initialize ZED
zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 32)
zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 65)
time.sleep(1.0)
print("Extrinsic Zed Exposure is set to: ",
    zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
)
print("Extrinsic Zed Gain is set to: ",
    zed.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
)
print("Extrinsic Zed fps set to: ",
        zed.cam.get_camera_information().camera_configuration.fps)
    
zed_intrinsics = zed.get_K()
zed_distortion_coefficients = np.array([0.0, 0, 0, 0, 0])
aruco_tag_length = 0.170
tag_to_tool_distance = 0.07
import pdb
pdb.set_trace()
left, right, depth = zed.get_frame()
rgb_img = left.cpu().numpy()
cv2.imwrite('/home/lifelong/image2.png',cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR))
aruco_pose_output = pose_estimation(rgb_img, cv2.aruco.DICT_6X6_50, zed_intrinsics, zed_distortion_coefficients, aruco_tag_length, visualize=False)
aruco_to_desired_servo_frame_matrix = np.array([[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,-1.0,tag_to_tool_distance],[0.0,0.0,0.0,1.0]])
aruco_to_desired_servo_frame = RigidTransform(rotation=aruco_to_desired_servo_frame_matrix[:3,:3],translation=aruco_to_desired_servo_frame_matrix[:3,3],to_frame="tag",from_frame="object")
    
if(aruco_pose_output is not None):
    _, rvec_aruco, tvec_aruco = aruco_pose_output
    zed_extrinsic_to_aruco = rvec_tvec_to_transform(rvec_aruco,tvec_aruco,to_frame="zed_extrinsic")
    world_to_aruco = camera_tf * zed_extrinsic_to_aruco
    world_to_desired_servo_frame = world_to_aruco * aruco_to_desired_servo_frame
    
    world_to_ee = robot.get_pose()
    world_to_ee.from_frame = "ee"
    # world_to_ee.translation[2] = world_to_ee.translation[2] + tag_to_tool_distance + 0.01
    # robot.move_pose(world_to_ee,vel=0.1,acc=0.1)
    # time.sleep(1)
    world_to_tool = RigidTransform(rotation=R.from_quat(tool_handle_wxyz,scalar_first=True).as_matrix(),translation=tool_handle_position,to_frame="world",from_frame="object")
    if(world_to_desired_servo_frame is None):
        import pdb
        pdb.set_trace()
        print("No Aruco marker track")
    place_pose = get_servo_pose(world_to_ee,world_to_tool,world_to_desired_servo_frame)
    robot.move_pose(place_pose,vel=0.07,acc=0.1)
    time.sleep(1)
    import pdb
    pdb.set_trace()
    left, right, depth = zed.get_frame()
    rgb_img = left.cpu().numpy()
    cv2.imwrite('/home/lifelong/image3.png',cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR))