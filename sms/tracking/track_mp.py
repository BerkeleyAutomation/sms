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
import pyzed.sl as sl
import json
from multiprocessing import Process, SimpleQueue, shared_memory
from torchvision.transforms.functional import resize
import os.path as osp
from sms.data.utils.dino_dataloader2 import DinoDataloader

WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
WORLD_TO_ZED2 = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def create_shm(shm_dict, create=True):
    shm_dict["depth_shm"] = shared_memory.SharedMemory(name="depth_shm", create=create, size=704*1280*32)
    shm_dict["dino_shm"] = shared_memory.SharedMemory(name="dino_shm", create=create, size=41*75*64*32)
    shm_dict["rgb_shm"] = shared_memory.SharedMemory(name="rgb_shm", create=create, size=704*1280*3*8)
    return shm_dict

def create_processes(process_dict, shm_dict, m_dict):
    process_dict["capture_dino"] = Process(target=cap_dino_process, args=(shm_dict,m_dict,))
    process_dict["track_opt"] = Process(target=track_opt_process, args=(shm_dict,m_dict,))
    return process_dict

def cap_dino_process(shm_dict, m_dict):
    zed = Zed(cam_id=16347230)
    zed_shape = m_dict["zed_shape"]
    print("ZED shape: ", zed_shape)
    img_buffer = np.ndarray((zed_shape[0], zed_shape[1], 3), dtype=np.uint8, buffer=shm_dict['rgb_shm'].buf)
    depth_buffer = np.ndarray((zed_shape[0], zed_shape[1]), dtype=np.float32, buffer=shm_dict['depth_shm'].buf)
    dino_buffer = np.ndarray((41, 75, 64), dtype=np.float32, buffer=shm_dict['dino_shm'].buf)
    cache_dir = m_dict["config_path"].parent.parent.parent
    dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
    
    dino_dataloader = DinoDataloader(
        image_list = None,
        device = 'cuda',
        cfg={"image_shape": [719, 1279]}, #HARDCODED BAD
        cache_path=dino_cache_path,
        use_denoiser=False,
    )
    dino_fn = dino_dataloader.get_pca_feats
    
    while True:
        left, right, depth = zed.get_frame()
        rgb = left/255.0
        dino_feats = dino_fn(
                rgb.permute(2, 0, 1).unsqueeze(0)
            ).squeeze()
        img_buffer[:,:,:] = left.cpu().detach().numpy()
        dino_buffer[:,:] = dino_feats.cpu().detach().numpy()
        depth_buffer[:,:] = depth.cpu().detach().numpy()
        
    
def track_opt_process(shm_dict, m_dict):
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
    
    config_path = m_dict["config_path"]
    zed_K = m_dict["zed_K"]
    zed_shape = m_dict["zed_shape"]
    camera_tf = m_dict["camera_tf"]
    zed_mesh = m_dict["zed_mesh"]
    c2z = m_dict["c2z"]
    
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
        mesh=zed_mesh,
        scale=0.001,
        position=c2z.translation,
        wxyz=c2z.quaternion,
    )
    toad_opt = Optimizer( # Initialize the optimizer
        config_path,
        zed_K,
        zed_shape[1],
        zed_shape[0],
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )
    img_buffer = np.ndarray((zed_shape[0], zed_shape[1], 3), dtype=np.uint8, buffer=shm_dict['rgb_shm'].buf)
    depth_buffer = np.ndarray((zed_shape[0], zed_shape[1]), dtype=np.float32, buffer=shm_dict['depth_shm'].buf)
    dino_buffer = np.ndarray((41, 75, 64), dtype=np.float32, buffer=shm_dict['dino_shm'].buf)
    
    print("Waiting for capture process to populate buffers")
    while (img_buffer[:,:,:] == 0).all(): # Wait for the capture process to populate the buffer
        time.sleep(0.1)
    print("Buffers populated")
    
    opt_init_handle.disabled = False
    @opt_init_handle.on_click # Btn callback -- initializes tracking optimization
    def _(_):
        assert toad_opt is not None
        opt_init_handle.disabled = True
        l = torch.from_numpy(img_buffer).cuda().clone()
        depth = torch.from_numpy(depth_buffer).cuda().clone()
        dino = torch.from_numpy(dino_buffer).cuda().clone()
        toad_opt.set_frame(l,toad_opt.cam2world_ns,depth,dino)
        toad_opt.init_obj_pose()
        # then have the zed_optimizer be allowed to run the optimizer steps.
    opt_init_handle.disabled = False
    text_handle.disabled = False
    query_handle.disabled = False
    
    @query_handle.on_click
    def _(_):
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
        execute_grasp_handle.disabled = True
        toad_opt.state_to_ply(toad_opt.max_relevancy_label)
        local_ply_filename = str(toad_opt.config_path.parent.joinpath("local.ply"))
        global_ply_filename = str(toad_opt.config_path.parent.joinpath("global.ply"))
        table_bounding_cube_filename = str(toad_opt.pipeline.datamanager.get_datapath().joinpath("table_bounding_cube.json"))
        save_dir = str(toad_opt.config_path.parent)
        ToadObject.generate_grasps(local_ply_filename, global_ply_filename, table_bounding_cube_filename, save_dir)
        execute_grasp_handle.disabled = False
            
    # Tracking process
    print("Starting tracking process")
    while True:
        left = torch.from_numpy(img_buffer).cuda().clone()
        depth = torch.from_numpy(depth_buffer).cuda().clone()
        dino = torch.from_numpy(dino_buffer).cuda().clone()
        if toad_opt.initialized:
            toad_opt.set_frame(left,toad_opt.cam2world_ns,depth,dino)
            n_opt_iters = 25
            outputs = toad_opt.step_opt(niter=n_opt_iters)
        
        # Visualize the pointcloud
        K = torch.from_numpy(zed_K).float().cuda()
        assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
        points, colors = Zed.project_depth(left, depth, K, depth_threshold=1.0, subsample=6)
        server.add_point_cloud(
            "camera/points",
            points=points,
            colors=colors,
            point_size=0.001,
        )
    
def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/2024_07_25_panda_gripper_demo4/sms-data/2024-07-25_231454/config.yml"),
):
    """
    Args:
        config_path: Path to the nerfstudio config file.
    """
    shm_dict = {}
    m_dict = {}
    process_dict = {}
    
    torch.multiprocessing.set_start_method('spawn', force = True)
    try:
        shm_dict = create_shm(shm_dict)
    except FileExistsError:
        print("Shared memory already exists, attempting to connect to existing shared memory objects")
        shm_dict = create_shm(shm_dict, create=False)
    except Exception as e:
        print(e)
    
    wrist_zed_id = 16347230
    extrinsic_zed_id = 22008760
    cam_id = extrinsic_zed_id
    zed = Zed(cam_id=cam_id) # Initialize ZED
    
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

    l, _, depth = zed.get_frame(depth=True)  # Grab a frame from the camera.
    m_dict["camera_tf"] = camera_tf
    m_dict["zed_mesh"] = zed.zed_mesh
    m_dict["config_path"] = config_path
    m_dict["zed_shape"] = l.shape
    m_dict["zed_K"] = zed.get_K()

    m_dict["c2z"] = zed.cam_to_zed
    
    # @execute_grasp_handle.on_click
    # def _(_):
    #     local_ply_filename = str(toad_opt.config_path.parent.joinpath("local.ply"))
    #     global_ply_filename = str(toad_opt.config_path.parent.joinpath("global.ply"))
    #     table_bounding_cube_filename = str(toad_opt.pipeline.datamanager.get_datapath().joinpath("table_bounding_cube.json"))
    #     pred_grasps_filename = str(toad_opt.config_path.parent.joinpath("pred_grasps_world.npy"))
    #     scores_filename = str(toad_opt.config_path.parent.joinpath("scores.npy"))
    #     seg_pc = o3d.io.read_point_cloud(local_ply_filename)
    #     full_pc_unfiltered = o3d.io.read_point_cloud(global_ply_filename)

    #     full_pc_points = np.asarray(full_pc_unfiltered.points)
    #     full_pc_colors = np.asarray(full_pc_unfiltered.colors)
    #     # Crop out noisy Gaussian means
    #     bounding_box_dict = None
    #     with open(table_bounding_cube_filename, 'r') as json_file:
    #         # Step 2: Load the contents of the file into a Python dictionary
    #         bounding_box_dict = json.load(json_file)
    #     cropped_indices = (full_pc_points[:, 0] >= bounding_box_dict['x_min']) & (full_pc_points[:, 0] <= bounding_box_dict['x_max']) & (full_pc_points[:, 1] >= bounding_box_dict['y_min']) & (full_pc_points[:, 1] <= bounding_box_dict['y_max']) & (full_pc_points[:, 2] >= bounding_box_dict['z_min']) & (full_pc_points[:, 2] <= bounding_box_dict['z_max'])
    #     filtered_pc_points = full_pc_points[cropped_indices]
    #     filtered_pc_colors = full_pc_colors[cropped_indices]
        
    #     full_pc = o3d.geometry.PointCloud()
    #     full_pc.points = o3d.utility.Vector3dVector(filtered_pc_points)
    #     full_pc.colors = o3d.utility.Vector3dVector(filtered_pc_colors)
        
    #     pred_grasps = np.load(pred_grasps_filename)
    #     scores = np.load(scores_filename)
    #     ordered_scores = scores[np.argsort(scores[0])[::-1]]
    #     # include viser visualization of the quality of the grasps
    #     best_grasp = pred_grasps[np.argmax(scores)]
    #     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #     grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #     grasp_point_world.transform(best_grasp)
    #     pre_grasp_tf = np.array([[1,0,0,0],
    #                             [0,1,0,0],
    #                             [0,0,1,-0.1],
    #                             [0,0,0,1]])
    #     pre_grasp_world_frame = best_grasp @ pre_grasp_tf
    #     pre_grasp_point_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #     pre_grasp_point_world.transform(pre_grasp_world_frame)
    #     # replace with viser
    #     o3d.visualization.draw_geometries([full_pc,coordinate_frame,grasp_point_world,pre_grasp_point_world])
    #     pre_grasp_rigid_tf = RigidTransform(rotation=pre_grasp_world_frame[:3,:3],translation=pre_grasp_world_frame[:3,3])
    #     robot.gripper.open()
    #     time.sleep(1)
    #     robot.move_pose(pre_grasp_rigid_tf,vel=0.5,acc=0.1)
    #     time.sleep(1)
    #     final_grasp_rigid_tf = RigidTransform(rotation=best_grasp[:3,:3],translation=best_grasp[:3,3])
    #     robot.move_pose(final_grasp_rigid_tf,vel=0.5,acc=0.1)
    #     time.sleep(1)
    #     robot.gripper.close()
    #     time.sleep(1)
    #     robot.move_pose(pre_grasp_rigid_tf,vel=0.5,acc=0.1)
    #     time.sleep(5)
    #     robot.move_pose(final_grasp_rigid_tf,vel=0.5,acc=0.1)
    #     time.sleep(1)
    #     robot.gripper.open()
    #     time.sleep(3)

    # real_frames = []
    # rendered_rgb_frames = []
    # # rendered_depth_frames = []
    # # rendered_dino_frames = []
    # save_videos = False
    # obj_label_list = [None for _ in range(toad_opt.num_groups)]
    
    try:
        process_dict = create_processes(process_dict=process_dict, shm_dict=shm_dict, m_dict=m_dict)

        for key in process_dict:
            process_dict[key].start()
    except KeyboardInterrupt:
        print("Exiting...")
        for key in process_dict:
            process_dict[key].join()
            print(f"Joined [{key}] process")
        for key in shm_dict:
            shm_dict[key].close()
            shm_dict[key].unlink()
            print(f"Closed and unlinked [/{key}] shared memory object")
    except Exception as e:
        print(e)
        exit()

if __name__ == "__main__":
    tyro.cli(main)
