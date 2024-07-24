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
import traceback 
import open3d as o3d


WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
WORLD_TO_ZED2 = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/2024_07_22_green_tape_bounding_cube/sms-data/2024-07-22_193605/config.yml"),
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
    
    zed = Zed() # Initialize ZED
    
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
    
    toad_opt = Optimizer( # Initialize the optimizer
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
        assert (zed is not None) and (toad_opt is not None)
        opt_init_handle.disabled = True
        l, _, depth = zed.get_frame(depth=True)
        toad_opt.set_frame(l,toad_opt.cam2world_ns,depth)
        with zed.raft_lock:
            toad_opt.init_obj_pose()
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
            relevancy = toad_opt.get_clip_relevancy(clip_encoder)
            group_masks = toad_opt.optimizer.group_masks

            relevancy_avg = []
            for mask in group_masks:
                relevancy_avg.append(torch.mean(relevancy[:,0:1][mask]))
            relevancy_avg = torch.tensor(relevancy_avg)
            toad_opt.max_relevancy_label = torch.argmax(relevancy_avg).item()
            toad_opt.max_relevancy_text = text_positives
            generate_grasps_handle.disabled = False
            execute_grasp_handle.disabled = False
        else:
            print("No language query provided")
    
    @generate_grasps_handle.on_click
    def _(_):
        # generate_grasps_handle.disabled = True
        toad_opt.state_to_ply(toad_opt.max_relevancy_label)
        local_ply_filename = str(toad_opt.config_path.parent.joinpath("local.ply"))
        global_ply_filename = str(toad_opt.config_path.parent.joinpath("global.ply"))
        table_bounding_cube_filename = str(toad_opt.pipeline.datamanager.get_datapath().joinpath("table_bounding_cube.json"))
        save_dir = str(toad_opt.config_path.parent)
        ToadObject.generate_grasps(local_ply_filename, global_ply_filename, table_bounding_cube_filename, save_dir)
        # generate_grasps_handle.disabled = False
        execute_grasp_handle.disabled = False
        
    @execute_grasp_handle.on_click
    def _(_):
        local_ply_filename = str(toad_opt.config_path.parent.joinpath("local.ply"))
        global_ply_filename = str(toad_opt.config_path.parent.joinpath("global.ply"))
        pred_grasps_filename = str(toad_opt.config_path.parent.joinpath("pred_grasps_world.npy"))
        scores_filename = str(toad_opt.config_path.parent.joinpath("scores.npy"))
        seg_pc = o3d.io.read_point_cloud(local_ply_filename)
        full_pc = o3d.io.read_point_cloud(global_ply_filename)
        pred_grasps = np.load(pred_grasps_filename)
        scores = np.load(scores_filename)
        ordered_scores = scores[np.argsort(scores[0])[::-1]]
        # include viser visualization of the quality of the grasps
        best_grasp = pred_grasps[np.argmax(scores)]
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
        # replace with viser
        o3d.visualization.draw_geometries([full_pc,coordinate_frame,grasp_point_world,pre_grasp_point_world])
        pre_grasp_rigid_tf = RigidTransform(rotation=pre_grasp_world_frame[:3,:3],translation=pre_grasp_world_frame[:3,3])
        robot.move_pose(pre_grasp_rigid_tf,vel=1.0,acc=0.1)
        final_grasp_rigid_tf = RigidTransform(rotation=best_grasp[:3,:3],translation=best_grasp[:3,3])
        robot.move_pose(final_grasp_rigid_tf,vel=1.0,acc=0.1)
        robot.gripper.close()
        robot.move_pose(pre_grasp_rigid_tf,vel=1.0,acc=0.1)

    real_frames = []
    rendered_rgb_frames = []
    # rendered_depth_frames = []
    # rendered_dino_frames = []
    
    obj_label_list = [None for _ in range(toad_opt.num_groups)]

    while True: # Main tracking loop
        try:
            if zed is not None:
                start_time = time.time()
                left, right, depth = zed.get_frame()
                print("Got frame in ", time.time()-start_time)
                start_time2 = time.time()
                assert isinstance(toad_opt, Optimizer)
                if toad_opt.initialized:
                    start_time3 = time.time()
                    toad_opt.set_frame(left,toad_opt.cam2world_ns,depth)
                    print("Set frame in ", time.time()-start_time3)
                    start_time5 = time.time()
                    n_opt_iters = 25
                    with zed.raft_lock:
                        outputs = toad_opt.step_opt(niter=n_opt_iters)
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
                    rendered_rgb_frames.append(outputs["rgb"].cpu().detach().numpy())
                    
                    tf_list = toad_opt.get_parts2world()
                    for idx, tf in enumerate(tf_list):
                        server.add_frame(
                            f"object/group_{idx}",
                            position=tf.translation(),
                            wxyz=tf.rotation().wxyz,
                            show_axes=True,
                            axes_length=0.05,
                            axes_radius=.001
                        )
                        mesh = toad_opt.toad_object.meshes[idx]
                        server.add_mesh_trimesh(
                            f"object/group_{idx}/mesh",
                            mesh=mesh,
                        )
                        if idx == toad_opt.max_relevancy_label:
                            obj_label_list[idx] = server.add_label(
                            f"object/group_{idx}/label",
                            text=toad_opt.max_relevancy_text,
                            position = (0,0,0.05),
                            )
                        else:
                            if obj_label_list[idx] is not None:
                                obj_label_list[idx].remove()
                            
                        
                    
                        # grasps = toad_opt.toad_object.grasps[idx] # [N_grasps, 7]
                        # grasp_mesh = toad_opt.toad_object.grasp_axis_mesh()
                        # for j, grasp in enumerate(grasps):
                        #     server.add_mesh_trimesh(
                        #         f"camera/object/group_{idx}/grasp_{j}",
                        #         grasp_mesh,
                        #         position=grasp[:3].cpu().numpy(),
                        #         wxyz=grasp[3:].cpu().numpy(),
                        #     )

                # Visualize pointcloud.
                start_time4 = time.time()
                K = torch.from_numpy(zed.get_K()).float().cuda()
                assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
                points, colors = Zed.project_depth(left, depth, K, depth_threshold=0.5, subsample=6)
                server.add_point_cloud(
                    "camera/points",
                    points=points,
                    colors=colors,
                    point_size=0.001,
                )
                print("Visualized pointcloud in ", time.time()-start_time4)
                print("Opt in ", time.time()-start_time2)

            else:
                time.sleep(1)
                
        except:
            # traceback.print_exc() 
            # Generate videos from the frames if the user interrupts the loop with ctrl+c
            frames_dict = {"real_frames": real_frames, 
                           "rendered_rgb": rendered_rgb_frames}
            generate_videos(frames_dict, fps = 5, config_path=config_path.parent)
            exit()


if __name__ == "__main__":
    tyro.cli(main)
