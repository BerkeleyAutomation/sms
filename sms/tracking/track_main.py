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
from sms.tracking.utils2 import generate_videos
from sms.tracking.toad_object import ToadObject
# import traceback 
import open3d as o3d
import pyzed.sl as sl
import json

WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
WORLD_TO_ZED2 = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/20240730_drill_battery2/sms-data/2024-07-31_032305/config.yml"),
):
    """Quick interactive demo for object tracking.

    Args:
        config_path: Path to the nerfstudio config file.
    """

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
    query_handle = server.add_gui_button("Query", disabled=True) # Button for querying the object once the user has inputted the query
    generate_grasps_handle = server.add_gui_button("Generate Grasps on Query", disabled=True) # Button for generating the grasps once the user has queried the object
    execute_grasp_handle = server.add_gui_button("Execute Grasp for Query", disabled=True) # Button for executing the grasp once the user has generated all suitable grasps
    
    wrist_zed_id = 16347230
    extrinsic_zed_id = 22008760
    zed = Zed(cam_id=extrinsic_zed_id,is_res_1080=True) # Initialize ZED
    
    robot = UR5Robot(gripper=1)
    clear_tcp(robot)
    home_joints = np.array([0.30947089195251465, -1.2793572584735315, -2.035713497792379, -1.388848606740133, 1.5713528394699097, 0.34230729937553406])
    robot.move_joint(home_joints,vel=1.0,acc=0.1)
    world_to_wrist = robot.get_pose()
    world_to_wrist.from_frame = "wrist"
    world_to_cam = world_to_wrist * WRIST_TO_CAM
    proper_world_to_cam_translation = world_to_cam.translation
    proper_world_to_cam_rotation = np.array([[0,1,0],[1,0,0],[0,0,-1]])
    proper_world_to_cam = RigidTransform(rotation=proper_world_to_cam_rotation,translation=proper_world_to_cam_translation,from_frame='cam',to_frame='world')
    proper_world_to_wrist = proper_world_to_cam * WRIST_TO_CAM.inverse()

    robot.move_pose(proper_world_to_wrist,vel=1.0,acc=0.1)
    
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
        assert (zed is not None) and (opt is not None)
        opt_init_handle.disabled = True
        l, _, depth = zed.get_frame(depth=True)
        opt.set_frame(l,opt.cam2world_ns,depth)
        with zed.raft_lock:
            opt.init_obj_pose()
        # then have the zed_optimizer be allowed to run the optimizer steps.
    opt_init_handle.disabled = False
    text_handle.disabled = False
    query_handle.disabled = False

    @query_handle.on_click
    def _(_):
        # TODO: Query for most relevant object
        text_positives = text_handle.value
        
        clip_encoder.set_positives(text_positives.split(";"))
        if len(clip_encoder.positives) > 0:
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
        else:
            print("No language query provided")
    
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
            import pdb
            pdb.set_trace()
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
                    start_time3 = time.time()
                    # opt.set_frame(left,opt.cam2world_ns,depth)
                    opt.set_observation(left,opt.cam2world_ns,depth)
                    print("Set frame in ", time.time()-start_time3)
                    start_time5 = time.time()
                    n_opt_iters = 20
                    with zed.raft_lock:
                        outputs = opt.step_opt(niter=n_opt_iters)
                    print(f"{n_opt_iters} opt steps in ", time.time()-start_time5)

                    # Add ZED img and GS render to viser
                    server.add_image(
                        "cam/zed_left",
                        left.cpu().detach().numpy(),
                        render_width=left.shape[1]/2500,
                        render_height=left.shape[0]/2500,
                        position = (0.5, 0.5, 0.5),
                        wxyz=(0, -0.7071068, -0.7071068, 0),
                        visible=True
                    )
                    if save_videos:
                        real_frames.append(left.cpu().detach().numpy())
                    
                    server.add_image(
                        "cam/gs_render",
                        outputs["rgb"].cpu().detach().numpy(),
                        render_width=left.shape[1]/2500,
                        render_height=left.shape[0]/2500,
                        position = (0.5, -0.5, 0.5),
                        wxyz=(0, -0.7071068, -0.7071068, 0),
                        visible=True
                    )
                    if save_videos:
                        rendered_rgb_frames.append(outputs["rgb"].cpu().detach().numpy())
                    
                    tf_list = opt.get_parts2world()
                    part_deltas.append(tf_list)
                    for idx, tf in enumerate(tf_list):
                        server.add_frame(
                            f"object/group_{idx}",
                            position=tf.translation(),
                            wxyz=tf.rotation().wxyz,
                            show_axes=True,
                            axes_length=0.05,
                            axes_radius=.001
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
            exit()


if __name__ == "__main__":
    tyro.cli(main)
