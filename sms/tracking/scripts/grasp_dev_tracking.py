import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import tyro
from pathlib import Path
from autolab_core import RigidTransform
# from sms.tracking.zed import Zed
from sms.sms.tracking.tri_zed import Zed
from sms.tracking.optim import Optimizer
from nerfstudio.cameras.cameras import Cameras
import warp as wp
from ur5py.ur5 import UR5Robot
from sms.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from sms.model.sms_gaussian_splatting import smsGaussianSplattingModelConfig, SH2RGB
from toad_object_karim import ToadObject


WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")
WORLD_TO_ZED2 = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf")

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/drill_and_spool1/sms-data/2024-07-21_155718/config.yml"),
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
    generate_grasps_handle = server.add_gui_button("Generate Grasps on Query", disabled=True) # Button for querying the object once the user has inputted the query

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
            generate_grasps_handle.disabled = False
        else:
            print("No language query provided")
    
    @generate_grasps_handle.on_click
    def _(_):
        # Global gaussian means : toad_opt.pipeline.state_stack[0]["means"] [N, 3]
        # Global gaussian spherical harmonics : toad_opt.pipeline.state_stack[0]["features_dc"] [N, 3]
        # ^ need to convert from SH2RGB
        # Global object masks: toad_opt.group_masks_global  [num_objects, N]
        # Object index: toad_opt.max_relevancy_label (int) (only available after querying)

        breakpoint()
        # NOTE until both envs are merged suggest just saving this data to a file and loading it in the other env to run grasping
        # TODO need to apply the deltas onto the original gaussian means to get the new tracked gaussian means 
        points = toad_opt.pipeline.state_stack[0]["means"].detach().cpu().numpy()
        features_dc = toad_opt.pipeline.state_stack[0]["features_dc"].detach().cpu().numpy()
        # TODO need to convert from SH2RGB
        colors = SH2RGB(features_dc)
        group_masks = np.array([group_mask.detach().cpu().numpy() for group_mask in toad_opt.group_masks_global])
        obj_idx = toad_opt.max_relevancy_label #  For drill spool scene: 0 for drill, 1 for wire spool
        ToadObject.generate_grasps(points, colors, group_masks, obj_idx)
        return NotImplementedError

    obj_label_list = [None for _ in range(toad_opt.num_groups)]

    while True:
        time.sleep(1.0)
    


if __name__ == "__main__":
    tyro.cli(main)
