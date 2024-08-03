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
# from ur5py.ur5 import UR5Robot
from sms.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from sms.tracking.utils2 import generate_videos
from sms.tracking.toad_object import ToadObject
# import traceback 
# import open3d as o3d
# import pyzed.sl as sl
# import json
import os
import cv2

WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
WORLD_TO_ZED2 = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")
DEVICE = 'cuda:0'
set_initial_frame = False

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/20240728_panda_gripper_light_blue_jaw_4/sms-data/2024-07-28_231908/config.yml"),
    offline_folder: Path = Path('/home/lifelong/sms/sms/data/utils/Detic/outputs/20240728_panda_gripper_light_blue_jaw_4/offline_video')
):
    """Quick interactive demo for object tracking.

    Args:
        config_path: Path to the nerfstudio config file.
        offline_folder: Path to the offline folder with images and depth
    """
    image_folder = os.path.join(offline_folder,"img")
    depth_folder = os.path.join(offline_folder,"depth")
    image_paths = sorted(os.listdir(image_folder))
    depth_paths = sorted(os.listdir(depth_folder))
    
    server = viser.ViserServer()
    wp.init()
    # Set up the camera.
    opt_init_handle = server.add_gui_button("Set initial frame", disabled=True) # Button for initializing tracking optimization
    
    clip_encoder = OpenCLIPNetworkConfig(
            clip_model_type="ViT-B-16", 
            clip_model_pretrained="laion2b_s34b_b88k", 
            clip_n_dims=512, 
            device=DEVICE
                ).setup() # OpenCLIP encoder for language querying utils
    assert isinstance(clip_encoder, OpenCLIPNetwork)
    
    text_handle = server.add_gui_text("Positives", "", disabled=True) # Text box for query input from user
    query_handle = server.add_gui_button("Query", disabled=True) # Button for querying the object once the user has inputted the query
    generate_grasps_handle = server.add_gui_button("Generate Grasps on Query", disabled=True) # Button for generating the grasps once the user has queried the object
    execute_grasp_handle = server.add_gui_button("Execute Grasp for Query", disabled=True) # Button for executing the grasp once the user has generated all suitable grasps
    
    camera_tf = WORLD_TO_ZED2
            
    # Visualize the camera.
    camera_frame = server.add_frame(
        "camera",
        position=camera_tf.translation,  # rough alignment.
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )
    # server.add_mesh_trimesh(
    #     "camera/mesh",
    #     mesh=zed.zed_mesh,
    #     scale=0.001,
    #     position=zed.cam_to_zed.translation,
    #     wxyz=zed.cam_to_zed.quaternion,
    # )
    
    initial_image_path = os.path.join(image_folder,image_paths[0])
    initial_depth_path = os.path.join(depth_folder,depth_paths[0])
    img_numpy = cv2.imread(initial_image_path)
    depth_numpy = np.load(initial_depth_path)
    l = torch.from_numpy(cv2.cvtColor(img_numpy,cv2.COLOR_RGB2BGR)).to(DEVICE)
    depth = torch.from_numpy(depth_numpy).to(DEVICE)
    
    zedK = np.array([[1.05576221e+03, 0.00000000e+00, 9.62041199e+02],
       [0.00000000e+00, 1.05576221e+03, 5.61746765e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    toad_opt = Optimizer( # Initialize the optimizer
        config_path,
        zedK,
        l.shape[1],
        l.shape[0], 
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )
    real_frames = []
    rendered_rgb_frames = []
    # rendered_depth_frames = []
    # rendered_dino_frames = []
    part_deltas = []
    save_videos = True
    obj_label_list = [None for _ in range(toad_opt.num_groups)]
    initial_image_path = os.path.join(image_folder,image_paths[0])
    initial_depth_path = os.path.join(depth_folder,depth_paths[0])
    img_numpy = cv2.imread(initial_image_path)
    depth_numpy = np.load(initial_depth_path)
    l = torch.from_numpy(cv2.cvtColor(img_numpy,cv2.COLOR_RGB2BGR)).to(DEVICE)
    depth = torch.from_numpy(depth_numpy).to(DEVICE)
    toad_opt.set_frame(l,toad_opt.cam2world_ns_ds,depth)
    # with zed.raft_lock:
    toad_opt.init_obj_pose()
    print("Starting main tracking loop")
    import pdb
    pdb.set_trace()
    try:
        # if zed is not None:
        start_time = time.time()
        start_time2 = time.time()
        assert isinstance(toad_opt, Optimizer)
        while not toad_opt.initialized:
            time.sleep(0.1)
        if toad_opt.initialized:
            start_time3 = time.time()
            for(img_path,depth_path) in zip(image_paths,depth_paths):
                full_image_path = os.path.join(image_folder,img_path)
                full_depth_path = os.path.join(depth_folder,depth_path)
                img_numpy = cv2.imread(full_image_path)
                depth_numpy = np.load(full_depth_path)
                left = torch.from_numpy(cv2.cvtColor(img_numpy,cv2.COLOR_RGB2BGR)).to(DEVICE)
                depth = torch.from_numpy(depth_numpy).to(DEVICE)
                toad_opt.set_observation(left,toad_opt.cam2world_ns,depth)
                print("Set frame in ", time.time()-start_time3)
                start_time5 = time.time()
                n_opt_iters = 25
                # with zed.raft_lock:
                outputs = toad_opt.step_opt(niter=n_opt_iters)
                print(f"{n_opt_iters} opt steps in ", time.time()-start_time5)

                # Add ZED img and GS render to viser
                server.add_image(
                    "cam/zed_left",
                    left.cpu().detach().numpy(),
                    render_width=left.shape[1]/2500,
                    render_height=left.shape[0]/2500,
                    position = (0.5, -0.5, 0.5),
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
                    position = (0.5, 0.5, 0.5),
                    wxyz=(0, -0.7071068, -0.7071068, 0),
                    visible=True
                )
                if save_videos:
                    rendered_rgb_frames.append(outputs["rgb"].cpu().detach().numpy())
                
                tf_list = toad_opt.get_parts2world()
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

                # Visualize pointcloud.
                start_time4 = time.time()
                K = torch.from_numpy(zedK).float().cuda()
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

        # else:
        #     time.sleep(1)
            
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
