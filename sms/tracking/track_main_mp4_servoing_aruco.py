import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import tyro
from pathlib import Path
from autolab_core import RigidTransform
# from sms.tracking.zed import Zed
from sms.tracking.tri_zed import Zed
from sms.tracking.optim import Optimizer
from nerfstudio.cameras.cameras import Cameras
import warp as wp
from ur5py.ur5 import UR5Robot
from sms.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from sms.tracking.utils2 import overlay
from sms.tracking.toad_object import ToadObject
from sms.tracking.grasp_vis_utils import visualize_grasps
# import traceback 
import open3d as o3d
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R
import json
import cv2
import traceback
import os
from PIL import Image
from tracikpy import TracIKSolver

WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
WORLD_TO_ZED2 = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")

ur5_urdf_filepath = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/urdf/ur5_robot.urdf'
ur5_ik_solver = TracIKSolver(ur5_urdf_filepath, 'base_link', 'tool0')
tag_to_tool_distance = 0.07
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

def rvec_tvec_to_transform(rvec, tvec,to_frame):
    """
    convert translation and rotation to pose
    """
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame="tag", to_frame=to_frame)

def estimate_theta(robot):
    cos_theta = np.dot(robot.get_pose().matrix[:3,2],np.array([0,0,-1]))/(np.linalg.norm(robot.get_pose()))
    theta = np.arccos(cos_theta)
    # normalizes the angle to be between -pi and pi
    return np.arctan2(np.sin(theta), np.cos(theta))

def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/20240914_drill_solo/sms-data/2024-09-14_033305/config.yml")
):

    robot = UR5Robot(gripper=1)
    clear_tcp(robot)
    robot.set_playload(1.1)
    time.sleep(1)
    home_joints = np.array([-1.363786522542135, -1.8143838087665003, -0.9117425123797815, -1.9958069960223597, 1.5864784717559814, 0.22764822840690613])
    robot.move_joint(home_joints,vel=0.5,acc=0.1)
    server = viser.ViserServer()
    wp.init()
    # Set up the camera.
    opt_init_handle = server.add_gui_button("Set initial frame", disabled=True) # Button for initializing tracking optimization
    
    clip_encoder = OpenCLIPNetworkConfig(
            clip_model_type="ViT-B-16", 
            clip_model_pretrained="laion2b_s34b_b88k", 
            clip_n_dims=512, 
            device='cuda:0'
                ).setup() # OpenCLIP encoder for language querying utils
    assert isinstance(clip_encoder, OpenCLIPNetwork)
    
    text_handle = server.add_gui_text("Positives", "", disabled=True) # Text box for query input from user
    
    pick_query_handle = server.add_gui_button("Pick Query", disabled=True) # Button for querying the object once the user has inputted the query
    generate_grasps_handle = server.add_gui_button("Generate & Execute Grasps on Pick Query", disabled=True) # Button for generating the grasps once the user has queried the object
    # execute_grasp_handle = server.add_gui_button("Execute Grasp for Pick Query", disabled=True) # Button for executing the grasp once the user has generated all suitable grasps
    go_to_aruco_handle = server.add_gui_button("Go To Aruco", disabled=True)
    servo_aruco_handle = server.add_gui_button("Servo Aruco",disabled=False) # Button for querying the object once the user has inputted the query
    
    wrist_zed_id = 16347230
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
    
    world_to_wrist = robot.get_pose()
    world_to_wrist.from_frame = "wrist"
    world_to_cam = world_to_wrist * WRIST_TO_CAM
    proper_world_to_cam = world_to_cam
    
    zed_mini_focal_length = 730 
    if(abs(zed.f_ - zed_mini_focal_length) > 10): # Check if the ZED connected is ZED mini or ZED2
        print("Connected to Zed2")
        zed.zed_mesh = zed.zed2_mesh
        camera_tf = WORLD_TO_ZED2
    else:
        print("Connected to ZedMini")
        zed.zed_mesh = zed.zedM_mesh
        camera_tf = proper_world_to_cam
            
    # Visualize the camera.
    camera_frame = server.add_frame(
        "camera",
        position=camera_tf.translation,  # rough alignment.
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    server.add_mesh_trimesh(
        "camera/mesh",
        mesh=zed.zed_mesh,
        scale=0.001,
        position=zed.cam_to_zed.translation,
        wxyz=zed.cam_to_zed.quaternion,
    )

    l, _, depth = zed.get_frame(depth=True)  # Grab a frame from the camera.

    opt = Optimizer( # Initialize the optimizer
        config_path,
        zed.get_K(),
        l.shape[1],
        l.shape[0], 
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )

    @opt_init_handle.on_click # Btn callback -- initializes tracking optimization
    def _(_):
        print("Clicked set initial frame. If no reaction, try scrolling on eval viser window")
        assert (zed is not None) and (opt is not None)
        opt_init_handle.disabled = True
        l, _, depth = zed.get_frame(depth=True)
        opt.set_frame(l,opt.cam2world_ns,depth)
        with zed.raft_lock:
            opt.init_obj_pose()
        # then have the zed_optimizer be allowed to run the optimizer steps.
    opt_init_handle.disabled = False
    text_handle.disabled = False
    pick_query_handle.disabled = False
    go_to_aruco_handle.disabled = False
    

    @pick_query_handle.on_click
    def _(_):
        # TODO: Query for most relevant object
        text_positives = text_handle.value
        queries = text_positives.split(";")
        if len(queries) <= 0:
            print("Enter something in the text box and if you want multiple words, separate with ;")
        object_query = queries[0]
        clip_encoder.set_positives([object_query])
        relevancy = opt.get_clip_relevancy(clip_encoder)
        group_masks = opt.optimizer.group_masks

        relevancy_avg = []
        for mask in group_masks:
            relevancy_avg.append(torch.mean(relevancy[:,0:1][mask]))
        relevancy_avg = torch.tensor(relevancy_avg)
        opt.max_relevancy_label = torch.argmax(relevancy_avg).item()
        opt.max_relevancy_text = text_positives
        generate_grasps_handle.disabled = False
        # execute_grasp_handle.disabled = False

    @servo_aruco_handle.on_click
    def _(_):
        opt.is_servoing = True
        servo_aruco_handle.disabled = True
        pick_query_handle.disabled = True
        generate_grasps_handle.disabled = True
        
    @go_to_aruco_handle.on_click
    def _(_):
        tool_handle = manual_tf[opt.max_relevancy_label]
        world_to_ee = robot.get_pose()
        world_to_ee.from_frame = "ee"
        # world_to_ee.translation[2] = world_to_ee.translation[2] + tag_to_tool_distance + 0.01
        # robot.move_pose(world_to_ee,vel=0.1,acc=0.1)
        # time.sleep(1)
        world_to_tool = RigidTransform(rotation=R.from_quat(tool_handle.wxyz,scalar_first=True).as_matrix(),translation=tool_handle.position,to_frame="world",from_frame="object")
        if(world_to_desired_servo_frame is None):
            import pdb
            pdb.set_trace()
            print("No Aruco marker track")
        place_pose = get_servo_pose(world_to_ee,world_to_tool,world_to_desired_servo_frame)
        robot.move_pose(place_pose,vel=0.07,acc=0.1)
        time.sleep(1)
        import pdb
        pdb.set_trace()
        servo_aruco_handle.disabled = False
        pick_query_handle.disabled = True
        generate_grasps_handle.disabled = True
        
    # Pick in frame a and Place in frame b 
    def get_place_pose(base_to_ee,base_to_frame_a,base_to_frame_b):
        theta = 0
        rotation_0_tf = RigidTransform(rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]),translation=np.zeros(3),to_frame='object',from_frame='object')
        base_to_frame_b_0_rotation = base_to_frame_b * rotation_0_tf
        theta = np.pi / 2
        rotation_90_tf = RigidTransform(rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]),translation=np.zeros(3),to_frame='object',from_frame='object')
        base_to_frame_b_90_rotation = base_to_frame_b * rotation_90_tf
        theta = np.pi
        rotation_180_tf = RigidTransform(rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]),translation=np.zeros(3),to_frame='object',from_frame='object')
        base_to_frame_b_180_rotation = base_to_frame_b * rotation_180_tf
        theta = -np.pi / 2
        rotation_270_tf = RigidTransform(rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]),translation=np.zeros(3),to_frame='object',from_frame='object')
        base_to_frame_b_270_rotation = base_to_frame_b * rotation_270_tf

        base_to_frame_b_variations = [base_to_frame_b_0_rotation,base_to_frame_b_90_rotation,base_to_frame_b_180_rotation,base_to_frame_b_270_rotation]
        min_angle = 1000
        min_base_to_frame_b = None
        for base_to_frame_b_variation in base_to_frame_b_variations:
            frame_a_to_frame_b = base_to_frame_a.inverse() * base_to_frame_b_variation
            angle = np.linalg.norm(R.from_matrix(frame_a_to_frame_b.rotation).as_rotvec())
            if(angle < min_angle):
                min_angle = angle
                min_base_to_frame_b = base_to_frame_b_variation
        base_to_frame_b = min_base_to_frame_b
        ee_to_object = base_to_ee.inverse() * base_to_frame_a
        new_base_to_ee = base_to_frame_b * ee_to_object.inverse()
        return new_base_to_ee
        
    @generate_grasps_handle.on_click
    def _(_):
        # generate_grasps_handle.disabled = True
        opt.state_to_ply(opt.max_relevancy_label)
        tool_tip = manual_tf[opt.max_relevancy_label]
        world_to_tool_tip = np.eye(4)
        world_to_tool_tip[:3,:3] = R.from_quat(tool_tip.wxyz,scalar_first=True).as_matrix()
        world_to_tool_tip[:3,3] = tool_tip.position
        tool_tip_to_grasp = np.eye(4)
        tool_tip_to_grasp[:3,3] = np.array([0.005,0.0975,-0.27])
        best_grasp = world_to_tool_tip @ tool_tip_to_grasp
        if(best_grasp[0,1] < 0):
            rotate_180_z = np.array([[-1,0,0,0],
                                     [0,-1,0,0],
                                     [0,0,1,0],
                                     [0,0,0,1]])
            best_grasp = best_grasp @ rotate_180_z
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        grasp_point_world.transform(best_grasp)
        pre_grasp_tf = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,-0.1],
                                [0,0,0,1]])
        pre_grasp_world_frame = best_grasp @ pre_grasp_tf
        pre_grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        pre_grasp_point_world.transform(pre_grasp_world_frame)
        
        post_grasp_tf = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,-0.1],
                                [0,0,0,1]])
        post_grasp_world_frame = best_grasp @ post_grasp_tf
        post_grasp_rigid_tf = RigidTransform(rotation=post_grasp_world_frame[:3,:3],translation=post_grasp_world_frame[:3,3])

        pre_grasp_rigid_tf = RigidTransform(rotation=pre_grasp_world_frame[:3,:3],translation=pre_grasp_world_frame[:3,3])
        robot.gripper.open()
        time.sleep(1)
        robot.move_pose(pre_grasp_rigid_tf,vel=1.0,acc=0.1)
        time.sleep(1)
        final_grasp_rigid_tf = RigidTransform(rotation=best_grasp[:3,:3],translation=best_grasp[:3,3])
        robot.move_pose(final_grasp_rigid_tf,vel=1.0,acc=0.1)
        time.sleep(1)
        robot.gripper.close()
        time.sleep(1)
        robot.set_playload(1.1 + 1.179) # Mass of the drill
        time.sleep(1)
        robot.move_pose(post_grasp_rigid_tf,vel=0.3,acc=0.1)
        time.sleep(1)
        go_to_aruco_handle.disabled = False
        pick_query_handle.disabled = True
        generate_grasps_handle.disabled = True
        
    part_deltas = []
    save_videos = True
    obj_label_list = [None for _ in range(opt.num_groups)]
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_dir = config_path.parent.joinpath(timestr)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_frames_video_writer = None
    rendered_rgb_video_writer = None
    fps = 5  # or use zed camera fps
    print("Starting main tracking loop")
    zed_intrinsics = zed.get_K()
    zed_distortion_coefficients = np.array([0.0, 0, 0, 0, 0])
    aruco_tag_length = 0.170
    
    aruco_to_desired_servo_frame_matrix = np.array([[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,-1.0,tag_to_tool_distance],[0.0,0.0,0.0,1.0]])
    aruco_to_desired_servo_frame = RigidTransform(rotation=aruco_to_desired_servo_frame_matrix[:3,:3],translation=aruco_to_desired_servo_frame_matrix[:3,3],to_frame="tag",from_frame="object")
    world_to_desired_servo_frame = None
    while True: # Main tracking loop
        try:
            if zed is not None:
                left, right, depth = zed.get_frame()
                assert isinstance(opt, Optimizer)
                if opt.initialized:
                    opt.set_observation(left,opt.cam2world_ns,depth)
                    n_opt_iters = 9
                    with zed.raft_lock:
                        outputs = opt.step_opt(niter=n_opt_iters)

                    # Add ZED img and GS render to viser
                    rgb_img = left.cpu().numpy()
                    aruco_pose_output = pose_estimation(rgb_img, cv2.aruco.DICT_6X6_50, zed_intrinsics, zed_distortion_coefficients, aruco_tag_length, visualize=False)
                    if(aruco_pose_output is not None):
                        _, rvec_aruco, tvec_aruco = aruco_pose_output
                        zed_extrinsic_to_aruco = rvec_tvec_to_transform(rvec_aruco,tvec_aruco,to_frame="zed_extrinsic")
                        world_to_aruco = camera_tf * zed_extrinsic_to_aruco
                        world_to_desired_servo_frame = world_to_aruco * aruco_to_desired_servo_frame
                        server.add_frame("aruco",position=world_to_aruco.translation,wxyz=world_to_aruco.quaternion,show_axes=True,axes_length=tag_to_tool_distance,axes_radius=0.005)
                        server.add_frame("desired_servo_frame",position=world_to_desired_servo_frame.translation,wxyz=world_to_desired_servo_frame.quaternion,show_axes=True,axes_length=tag_to_tool_distance,axes_radius=0.005)
                    
                    for i in range(len(opt.group_masks)):
                        frame = opt.optimizer.frame.roi_frames[i]
                        xmin = frame.xmin
                        xmax = frame.xmax
                        ymin = frame.ymin
                        ymax = frame.ymax
                        rgb_img = cv2.rectangle(rgb_img, (xmin, ymin), (xmax, ymax),(255,0,0), 2)
                        if opt.optimizer.frame._obj_masks is not None:
                            rgb_img = overlay(rgb_img, mask = opt.optimizer.frame._obj_masks[i].detach().cpu(), color=((i+1)*100, 0, 255-i*100), alpha=0.3)
                        
                    server.add_image(
                        "cam/zed_left",
                        rgb_img,
                        render_width=rgb_img.shape[1]/3000,
                        render_height=rgb_img.shape[0]/3000,
                        position = (-0.5, -0.5, 0.5),
                        wxyz=(0, -0.7071068, -0.7071068, 0),
                        visible=True
                    )

                    server.add_image(
                        "cam/gs_render",
                        outputs["rgb"].cpu().detach().numpy(),
                        render_width=outputs["rgb"].shape[1]/3000,
                        render_height=outputs["rgb"].shape[0]/3000,
                        position = (0.5, -0.5, 0.5),
                        wxyz=(0, -0.7071068, -0.7071068, 0),
                        visible=True
                    )
                    if save_videos:
                        if real_frames_video_writer is None:
                            # Initialize the video writers
                            H_left, W_left = left.shape[:2]
                            H_rendered, W_rendered = outputs["rgb"].shape[:2]

                            # Define the codec and create VideoWriter object
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                            real_frames_video_writer = cv2.VideoWriter(str(output_dir.joinpath("real_frames.mp4")), fourcc, fps, (W_left, H_left))
                            rendered_rgb_video_writer = cv2.VideoWriter(str(output_dir.joinpath("rendered_rgb.mp4")), fourcc, fps, (W_rendered, H_rendered))

                        # Write frames to video files
                        # For real_frames
                        # import pdb; pdb.set_trace()
                        real_frame = left.cpu().detach().numpy()  # shape (H, W, 3)
                        real_frame = (real_frame).astype(np.uint8)
                        real_frame = cv2.cvtColor(real_frame, cv2.COLOR_RGB2BGR)  # Convert to BGR if needed
                        real_frames_video_writer.write(real_frame)

                        # For rendered_rgb_frames
                        rendered_frame = outputs["rgb"].cpu().detach().numpy()
                        rendered_frame = (rendered_frame * 255).astype(np.uint8)
                        rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
                        rendered_rgb_video_writer.write(rendered_frame)
                    
                    tf_list = opt.get_parts2world()
                    part_deltas.append(tf_list)
                    manual_tf = [None, None]
                    for idx, tf in enumerate(tf_list):
                        server.add_frame(
                            f"object/group_{idx}",
                            position=tf.translation(),
                            wxyz=tf.rotation().wxyz,
                            show_axes=True,
                            axes_length=0.05,
                            axes_radius=.001
                        )
                        
                        p2manual_tf_SE3 = opt.optimizer.p2manual_tf_SE3[idx]
                        manual_tf2w_SE3 = tf @ p2manual_tf_SE3 
                        manual_tf[idx] = server.add_frame(
                            f"object/manual_tf{idx}",
                            position=manual_tf2w_SE3.wxyz_xyz[4:],
                            wxyz= manual_tf2w_SE3.wxyz_xyz[:4],
                            show_axes=True,
                            axes_length=0.09,
                            axes_radius=.0025
                        )
                        
                        mesh = opt.toad_object.meshes[idx]
                        server.add_mesh_trimesh(
                            f"object/group_{idx}/mesh",
                            mesh=mesh,
                        )
                        if idx == opt.max_relevancy_label:
                            obj_label_list[idx] = server.add_label(
                            f"object/group_{idx}/label",
                            text=opt.max_relevancy_text,
                            position = (0,0,0.05),
                            )
                        elif idx == opt.place_max_relevancy_label:
                            obj_label_list[idx] = server.add_label(
                            f"object/group_{idx}/label",
                            text=opt.place_max_relevancy_text,
                            position = (0,0,0.05),
                            )
                        else:
                            if obj_label_list[idx] is not None:
                                obj_label_list[idx].remove()
                    if(opt.is_servoing):
                        tool_handle = manual_tf[opt.max_relevancy_label]
                        world_to_ee = robot.get_pose()
                        world_to_ee.from_frame = "ee"
                        world_to_tool = RigidTransform(rotation=R.from_quat(tool_handle.wxyz,scalar_first=True).as_matrix(),translation=tool_handle.position,to_frame="world",from_frame="object")
                        if(world_to_desired_servo_frame is None):
                            import pdb
                            pdb.set_trace()
                            print("No Aruco marker track")
                        place_pose = get_servo_pose(world_to_ee,world_to_tool,world_to_desired_servo_frame)
                        current_to_desired = world_to_ee.inverse() * place_pose
                        error_rotation_matrix = place_pose.matrix[:3,:3] @ world_to_ee.matrix[:3,:3].T
                        
                        rotation_angle = np.arccos((np.matrix.trace(error_rotation_matrix) - 1 ) / 2)
                        rotation_scaling_factor = 0.18
                        distance_scaling_factor = 1.5
                        distance = np.linalg.norm(current_to_desired.translation)
                        
                        robot_vel = min((distance_scaling_factor * distance) + (rotation_angle * rotation_scaling_factor),0.07)
                        robot.move_pose(place_pose,vel=robot_vel,acc=1.0,asyn=True)

                # Visualize pointcloud.
                K = torch.from_numpy(zed.get_K()).float().cuda()
                assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
                points, colors = Zed.project_depth(left, depth, K, depth_threshold=1.0, subsample=6)
                server.add_point_cloud(
                    "camera/points",
                    points=points,
                    colors=colors,
                    point_size=0.001,
                )

            else:
                time.sleep(1)
                
        except KeyboardInterrupt:
            # Release the video writers
            if real_frames_video_writer is not None:
                real_frames_video_writer.release()
            if rendered_rgb_video_writer is not None:
                rendered_rgb_video_writer.release()

            # Save background image
            background = opt.background_snapshot() # (H, W, 3)
            # Save background image to npy and png files
            np.save(output_dir.joinpath("background.npy"), background)
            # Save as PNG
            im = Image.fromarray((background*255).astype(np.uint8))
            im.save(str(output_dir.joinpath("background.png")))

            # Save part_deltas to npy file
            np.save(output_dir.joinpath("part_deltas_traj.npy"), np.array(part_deltas))

            # Save clusters
            if opt.cluster_from_file is not None:
                clusters = opt.cluster_from_file
                np.save(output_dir.joinpath("clusters.npy"), clusters)
            else:
                # copy the cluster file from the config path
                import shutil
                shutil.copy(opt.cluster_file, output_dir.joinpath("clusters.npy"))
            exit()
        except Exception as e:
            print("An exception occurred: ", e)
            traceback.print_exc()
            # Release the video writers if an exception occurs
            if real_frames_video_writer is not None:
                real_frames_video_writer.release()
            if rendered_rgb_video_writer is not None:
                rendered_rgb_video_writer.release()
            exit()
            
if __name__ == "__main__":
    tyro.cli(main)
