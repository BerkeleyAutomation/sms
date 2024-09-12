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
from sms.tracking.utils2 import generate_videos, overlay
from sms.tracking.toad_object import ToadObject
from sms.tracking.grasp_vis_utils import visualize_grasps
# import traceback 
import open3d as o3d
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R
import json
import cv2
import traceback

WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
WORLD_TO_ZED2 = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/20240912_iron_shelf/sms-data/2024-09-12_012956/config.yml")

    # config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/20240910_1350_shoe_solo/sms-data/2024-09-10_135106/config.yml")
    # config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/20240910_shoe_and_shoebox/sms-data/2024-09-10_053819/config.yml"),
    # config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/20240907_shoe_drill_tools/sms-data/2024-09-07_214248/config.yml"),
):
    """Quick interactive demo for object tracking.

    Args:
        config_path: Path to the nerfstudio config file.
    """
    robot = UR5Robot(gripper=1)
    clear_tcp(robot)
    home_joints = np.array([-1.433847729359762, -1.6635258833514612, -0.8512895742999476, -3.7683952490436, -1.4371045271502894, 3.1419787406921387])
    robot.move_joint(home_joints,vel=1.0,acc=0.1)
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
    generate_grasps_handle = server.add_gui_button("Generate Grasps on Pick Query", disabled=True) # Button for generating the grasps once the user has queried the object
    execute_grasp_handle = server.add_gui_button("Execute Grasp for Pick Query", disabled=True) # Button for executing the grasp once the user has generated all suitable grasps
    place_query_handle = server.add_gui_button("Place Query", disabled=True)
    execute_place_handle = server.add_gui_button("Execute Placement", disabled=True)
    
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
        execute_grasp_handle.disabled = False

    @place_query_handle.on_click
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
        opt.place_max_relevancy_label = torch.argmax(relevancy_avg).item()
        opt.place_max_relevancy_text = text_positives
        execute_place_handle.disabled = False
        # if len(queries) == 2: # Object and part query
        #     part_query = queries[1]
        #     max_mask_label = opt.max_relevancy_label
        #     clip_encoder.set_positives(part_query)
        #     relevancy = opt.get_clip_relevancy(clip_encoder)
        #     part_relevancies = relevancy[:,0:1][group_masks[max_mask_label]]
        #     dino_features_for_object = opt.pipeline.model.gauss_params['dino_feats'][group_masks[max_mask_label]]
        #     part_relevancies_filename = str(opt.config_path.parent.joinpath("part_relevancies.npy"))
        #     dino_features_for_object_filename = str(opt.config_path.parent.joinpath("dino_features_for_object.npy"))
        #     np.save(part_relevancies_filename,part_relevancies.detach().cpu().numpy())
        #     np.save(dino_features_for_object_filename,dino_features_for_object.detach().cpu().numpy())
        #     generate_grasps_handle.disabled = False
        #     execute_grasp_handle.disabled = False
        #     # Part Oriented Grasping here
        # else:
        #     print("No language query provided")
    
    @execute_place_handle.on_click
    def _(_):
        max_relevancy_label = opt.max_relevancy_label
        place_max_relevancy_label = opt.place_max_relevancy_label
        assert max_relevancy_label != place_max_relevancy_label, "You have the pick and the place set to the same spot"
        print("HI")
        pick_handle = manual_tf[max_relevancy_label]
        place_handle = manual_tf[place_max_relevancy_label]
        world_to_ee = robot.get_pose()
        world_to_ee.from_frame = "ee"
        world_to_pick = RigidTransform(rotation=R.from_quat(pick_handle.wxyz,scalar_first=True).as_matrix(),translation=pick_handle.position,to_frame="world",from_frame="object")
        world_to_place = RigidTransform(rotation=R.from_quat(place_handle.wxyz,scalar_first=True).as_matrix(),translation=place_handle.position,to_frame="world",from_frame="object")
        place_pose = get_place_pose(world_to_ee,world_to_pick,world_to_place)
        place_pose_z = world_to_ee.copy()
        place_pose_z.translation = np.array([world_to_ee.translation[0],world_to_ee.translation[1],place_pose.translation[2] + 0.01])
        import pdb
        pdb.set_trace()
        robot.move_pose(place_pose_z,vel=0.5,acc=0.1)
        time.sleep(1)
        place_pose_rotation = place_pose_z
        place_pose_rotation.rotation = place_pose.rotation
        import pdb
        pdb.set_trace()
        robot.move_pose(place_pose_rotation,vel=0.5,acc=0.1)
        time.sleep(1)
        import pdb
        pdb.set_trace()
        robot.move_pose(place_pose,vel=0.5,acc=0.1)
        time.sleep(1)
        
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
        local_ply_filename = str(opt.config_path.parent.joinpath("local.ply"))
        global_ply_filename = str(opt.config_path.parent.joinpath("global.ply"))
        table_bounding_cube_filename = str(opt.pipeline.datamanager.get_datapath().joinpath("table_bounding_cube.json"))
        save_dir = str(opt.config_path.parent)
        ToadObject.generate_grasps(local_ply_filename, global_ply_filename, table_bounding_cube_filename, save_dir)
        # generate_grasps_handle.disabled = False
        execute_grasp_handle.disabled = False
        
    @execute_grasp_handle.on_click
    def _(_):
        local_ply_filename = str(opt.config_path.parent.joinpath("local.ply"))
        global_ply_filename = str(opt.config_path.parent.joinpath("global.ply"))
        table_bounding_cube_filename = str(opt.pipeline.datamanager.get_datapath().joinpath("table_bounding_cube.json"))
        pred_grasps_filename = str(opt.config_path.parent.joinpath("pred_grasps_world.npy"))
        scores_filename = str(opt.config_path.parent.joinpath("scores.npy"))
        seg_pc = o3d.io.read_point_cloud(local_ply_filename)
        full_pc_unfiltered = o3d.io.read_point_cloud(global_ply_filename)

        full_pc_points = np.asarray(full_pc_unfiltered.points)
        full_pc_colors = np.asarray(full_pc_unfiltered.colors)
        # Crop out noisy Gaussian means
        bounding_box_dict = None
        with open(table_bounding_cube_filename, 'r') as json_file:
            # Step 2: Load the contents of the file into a Python dictionary
            bounding_box_dict = json.load(json_file)
        cropped_indices = (full_pc_points[:, 0] >= bounding_box_dict['x_min']) & (full_pc_points[:, 0] <= bounding_box_dict['x_max']) & (full_pc_points[:, 1] >= bounding_box_dict['y_min']) & (full_pc_points[:, 1] <= bounding_box_dict['y_max']) & (full_pc_points[:, 2] >= bounding_box_dict['z_min']) & (full_pc_points[:, 2] <= bounding_box_dict['z_max'])
        filtered_pc_points = full_pc_points[cropped_indices]
        filtered_pc_colors = full_pc_colors[cropped_indices]
        
        full_pc = o3d.geometry.PointCloud()
        full_pc.points = o3d.utility.Vector3dVector(filtered_pc_points)
        full_pc.colors = o3d.utility.Vector3dVector(filtered_pc_colors)
        
        pred_grasps = np.load(pred_grasps_filename)
        scores = np.load(scores_filename)
        ordered_scores = scores[np.argsort(scores[0])[::-1]]
        # include viser visualization of the quality of the grasps
        best_grasp = pred_grasps[np.argmax(scores)]
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
                                [0,0,1,-0.05],
                                [0,0,0,1]])
        post_grasp_world_frame = best_grasp @ post_grasp_tf
        post_grasp_rigid_tf = RigidTransform(rotation=post_grasp_world_frame[:3,:3],translation=post_grasp_world_frame[:3,3])
        # replace with viser
        grasp_server = viser.ViserServer()
        visualize_grasps(local_ply_filename, global_ply_filename, table_bounding_cube_filename, pred_grasps_filename, scores_filename, grasp_server)
        
        o3d.visualization.draw_geometries([full_pc,coordinate_frame,grasp_point_world,pre_grasp_point_world])
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
        robot.move_pose(post_grasp_rigid_tf,vel=0.3,acc=0.1)
        time.sleep(1)
        place_query_handle.disabled = False
        
        # center_gripper_joints = np.array(0.016485050320625305, -1.8846338430987757, -2.4609714190112513, 0.05439639091491699, 1.6994218826293945, 4.563924312591553)
        # robot.move_joint(center_gripper_joints,vel=0.3,acc=0.1)
        # robot.move_pose(final_grasp_rigid_tf,vel=0.5,acc=0.1)
        # time.sleep(1)
        # robot.gripper.open()
        # time.sleep(3)

    real_frames = []
    rendered_rgb_frames = []
    # rendered_depth_frames = []
    # rendered_dino_frames = []
    part_deltas = []
    save_videos = True
    obj_label_list = [None for _ in range(opt.num_groups)]
    
    
    print("Starting main tracking loop")
    while True: # Main tracking loop
        try:
            if zed is not None:
                # start_time = time.time()
                left, right, depth = zed.get_frame()
                # print("Got frame in ", time.time()-start_time)
                # start_time2 = time.time()
                assert isinstance(opt, Optimizer)
                if opt.initialized:
                    # start_time3 = time.time()
                    # opt.set_frame(left,opt.cam2world_ns,depth)
                    opt.set_observation(left,opt.cam2world_ns,depth)
                    # print("Set frame in ", time.time()-start_time3)
                    # start_time5 = time.time()
                    n_opt_iters = 10
                    with zed.raft_lock:
                        outputs = opt.step_opt(niter=n_opt_iters)
                    # print(f"{n_opt_iters} opt steps in ", time.time()-start_time5)

                    # Add ZED img and GS render to viser
                    rgb_img = left.cpu().numpy()
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
                    # if save_videos:
                        # real_frames.append(rgb_img)
                        
                        # real_frames.append(left.cpu().detach().numpy()) # Switch to this for no ROI bbox
                        
                        # rendered_rgb_frames.append(outputs["rgb"].cpu().detach().numpy())
                    
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

                # Visualize pointcloud.
                start_time4 = time.time()
                K = torch.from_numpy(zed.get_K()).float().cuda()
                assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
                points, colors = Zed.project_depth(left, depth, K, depth_threshold=1.0, subsample=6)
                server.add_point_cloud(
                    "camera/points",
                    points=points,
                    colors=colors,
                    point_size=0.001,
                )
                # print("Visualized pointcloud in ", time.time()-start_time4)
                # print("Opt in ", time.time()-start_time2)

            else:
                time.sleep(1)
                
        except KeyboardInterrupt:
            # Generate videos from the frames if the user interrupts the loop with ctrl+c
            frames_dict = {"real_frames": real_frames, 
                           "rendered_rgb": rendered_rgb_frames}
            timestr = generate_videos(frames_dict, fps = 5, config_path=config_path.parent)
            
            # Save part deltas to npy file
            path = config_path.parent.joinpath(f"{timestr}")
            np.save(path.joinpath("part_deltas_traj.npy"), np.array(part_deltas))
            exit()
        except Exception as e:
            print("An exception occured: ", e)
            traceback.print_exc()
            exit()
            
if __name__ == "__main__":
    tyro.cli(main)
