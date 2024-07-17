import torch
import viser
import viser.transforms as vtf
import time
import numpy as np
import tyro
from pathlib import Path
from autolab_core import RigidTransform
from sms.tracking.zed import Zed
from sms.tracking.optim import Optimizer
from nerfstudio.cameras.cameras import Cameras


def main(
    config_path: Path = Path("/home/lifelong/sms/sms/data/utils/Detic/outputs/block_tape_screwdriver/sms-data/2024-07-16_220503/config.yml"),
):
    """Quick interactive demo for object tracking.

    Args:
        config_path: Path to the nerfstudio config file.
    """
    server = viser.ViserServer()

    # Set up the camera.
    opt_init_handle = server.add_gui_button("Set initial frame", disabled=True)
    # try:
    zed = Zed()

    # Visualize the camera.
    camera_tf = RigidTransform.load("/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_final_mini.tf")
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
    # import pdb; pdb.set_trace()
    zed_intr = zed.get_K()
    ns_camera = Cameras(fx=zed_intr[0][0],
                        fy=zed_intr[1][1],
                        cx=zed_intr[0][2],
                        cy=zed_intr[1][2],
                        width=zed.width,
                        height=zed.height,
                        camera_to_worlds=torch.from_numpy(camera_tf.matrix[:3,:4]).unsqueeze(0))

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
    

    @opt_init_handle.on_click
    def _(_):
        assert (zed is not None) and (toad_opt is not None)
        opt_init_handle.disabled = True
        l, _, depth = zed.get_frame(depth=True)
        toad_opt.set_frame(l,ns_camera,depth)
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
            left, right, depth = zed.get_frame()
            print("Got frame")
            assert isinstance(toad_opt, Optimizer)
            if toad_opt.initialized:
                toad_opt.set_frame(left,ns_camera,depth)
                with zed.raft_lock:
                    toad_opt.step_opt(niter=50)

                tf_list = toad_opt.get_parts2cam()
                print(tf_list)
                for idx, tf in enumerate(tf_list):
                    server.add_frame(
                        f"camera/object/group_{idx}",
                        position=tf.translation(),
                        wxyz=tf.rotation().wxyz,
                        show_axes=True,
                        axes_length=0.02,
                        axes_radius=.002
                    )
                    mesh = toad_opt.toad_object.meshes[idx]
                    server.add_mesh_trimesh(
                        f"camera/object/group_{idx}/mesh",
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
            K = torch.from_numpy(zed.get_K()).float().cuda()
            assert isinstance(left, torch.Tensor) and isinstance(depth, torch.Tensor)
            points, colors = Zed.project_depth(left, depth, K)
            server.add_point_cloud(
                "camera/points",
                points=points,
                colors=colors,
                point_size=0.001,
            )

        else:
            time.sleep(1)


if __name__ == "__main__":
    tyro.cli(main)
