import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import tyro
from pathlib import Path
from autolab_core import RigidTransform
# from sms.tracking.zed import Zed
from sms.tracking.prime_tri_zed import Zed
from sms.tracking.optim import Optimizer
from nerfstudio.cameras.cameras import Cameras
import warp as wp
from ur5py.ur5 import UR5Robot

WRIST_TO_CAM = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_cam.tf")

def clear_tcp(robot):
    tool_to_wrist = RigidTransform()
    tool_to_wrist.translation = np.array([0, 0, 0])
    tool_to_wrist.from_frame = "tool"
    tool_to_wrist.to_frame = "wrist"
    robot.set_tcp(tool_to_wrist)
    
def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/bowl/sms-data/2024-07-18_163924/config.yml"),
):
    """Quick interactive demo for object tracking.

    Args:
        config_path: Path to the nerfstudio config file.
    """
    server = viser.ViserServer()
    wp.init()
    # Set up the camera.
    opt_init_handle = server.add_gui_button("Set initial frame", disabled=True)
    # try:
    zed = Zed()
    zed_mini_focal_length = 730
    if(abs(zed.f_ - zed_mini_focal_length) > 10):
        print("Accidentally connected to wrong Zed. Trying again")
        zed = Zed()
        if(abs(zed.f_ - zed_mini_focal_length) > 10):
            print("Make sure just Zed mini is plugged in")
            exit()

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
    
    # Visualize the camera.
    # camera_tf = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_ZM.tf")
    camera_tf = proper_world_to_cam
    
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
    # zed_intr = zed.get_K()
    # ns_camera = Cameras(fx=zed_intr[0][0],
    #                     fy=zed_intr[1][1],
    #                     cx=zed_intr[0][2],
    #                     cy=zed_intr[1][2],
    #                     width=zed.width,
    #                     height=zed.height,
    #                     camera_to_worlds=torch.from_numpy(camera_tf.matrix[:3,:4]).unsqueeze(0).float())
    l, _, depth = zed.get_frame(depth=True)  # type: ignore
    
    toad_opt = Optimizer(
        config_path,
        zed.get_K(),
        l.shape[1],
        l.shape[0], 
        # zed.width,
        # zed.height,
        init_cam_pose=torch.from_numpy(
            vtf.SE3(
                wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
            ).as_matrix()[None, :3, :]
        ).float(),
    )
    # import pdb; pdb.set_trace()
    

    @opt_init_handle.on_click
    def _(_):
        assert (zed is not None) and (toad_opt is not None)
        opt_init_handle.disabled = True
        l, _, depth = zed.get_frame(depth=True)
        toad_opt.set_frame(l,toad_opt.cam2world_ns,depth)
        with zed.raft_lock:
            toad_opt.init_obj_pose()
        # then have the zed_optimizer be allowed to run the optimizer steps.
    opt_init_handle.disabled = False

    # except Exception as e:
    #     print(e)
        # print("Zed not available -- won't show camera feed.")
        # print("Also won't run the optimizer.")
        # zed, toad_opt = None, None

    while True:
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
                
                server.add_image(
                    "cam/gs_render",
                    outputs["rgb"].cpu().detach().numpy(),
                    render_width=left.shape[1]/2500,
                    render_height=left.shape[0]/2500,
                    position = (0.5, -0.5, 0.5),
                    wxyz=(0, -0.7071068, -0.7071068, 0),
                    visible=True
                )
                
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
            points, colors = Zed.project_depth(left, depth, K)
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


if __name__ == "__main__":
    tyro.cli(main)
