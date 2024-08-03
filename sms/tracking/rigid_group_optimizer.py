import torch
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import numpy as np
from typing import List, Optional, Literal, Callable
import kornia
from sms.model.sms_gaussian_splatting import smsGaussianSplattingModel
from sms.data.utils.dino_dataloader2 import DinoDataloader
from contextlib import nullcontext
from nerfstudio.engine.schedulers import (
    ExponentialDecayScheduler,
    ExponentialDecaySchedulerConfig,
)
import warp as wp
from sms.tracking.atap_loss import ATAPLoss
from sms.tracking.utils import *
import viser.transforms as vtf
import trimesh
from typing import Tuple
from nerfstudio.model_components.losses import depth_ranking_loss
from sms.tracking.utils2 import *
# from sms.tracking.frame import Frame
from sms.tracking.observation import PosedObservation, Frame
import time
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
import wandb
from dataclasses import dataclass
from nerfstudio.cameras.cameras import Cameras

@dataclass
class RigidGroupOptimizerConfig:
    use_depth: bool = True
    rank_loss_mult: float = 0.1
    rank_loss_erode: int = 5
    depth_ignore_threshold: float = 0.1  # in meters
    use_atap: bool = False
    pose_lr: float = 0.005
    pose_lr_final: float = 0.001
    mask_hands: bool = False
    do_obj_optim: bool = False
    blur_kernel_size: int = 5
    clip_grad: float = 0.8
    use_roi = True
    roi_inflate: float = 0.25
    
class RigidGroupOptimizer:
    """From: part, To: object. in current world frame. Part frame is centered at part centroid, and object frame is centered at object centroid."""

    def __init__(
        self,
        config: RigidGroupOptimizerConfig,
        sms_model: smsGaussianSplattingModel,
        group_masks: List[torch.Tensor],
        group_labels: torch.Tensor,
        dataset_scale: float,
        render_lock = nullcontext(),
        use_wandb = False,
    ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        torch.autograd.set_detect_anomaly(True)
        self.config = config
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project="LEGS-TOGO", save_code=True)
        self.dataset_scale = dataset_scale
        self.tape = None
        self.is_initialized = False
        self.hand_lefts = [] #list of bools for each hand frame
        self.sms_model = sms_model
        # detach all the params to avoid retain_graph issue
        # self.sms_model.gauss_params["means"] = self.sms_model.gauss_params[
        #     "means"
        # ].detach().clone()
        # self.sms_model.gauss_params["quats"] = self.sms_model.gauss_params[
        #     "quats"
        # ].detach().clone()

        self.group_labels = group_labels
        self.group_masks = group_masks
        
        # store a 7-vec of trans, rotation for each group (x,y,z,qw,qx,qy,qz)
        self.part_deltas = torch.zeros(
            len(group_masks), 7, dtype=torch.float32, device="cuda"
        )
        self.part_deltas[:, 3:] = torch.tensor(
            [1, 0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.part_deltas = torch.nn.Parameter(self.part_deltas)
        self.part_deltas.requires_grad_(True)
        k = self.config.blur_kernel_size
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.config.pose_lr)
        self.keyframes = []
        #hand_frames stores a list of hand vertices and faces for each keyframe, stored in the OBJECT COORDINATE FRAME
        self.hand_frames = []
        # lock to prevent blocking the render thread if provided
        self.render_lock = render_lock
        if self.config.use_atap and len(group_masks) > 1:
            self.atap = ATAPLoss(sms_model, group_masks, group_labels, self.dataset_scale)
        else:
            self.config.use_atap = False
        self.init_means = self.sms_model.gauss_params["means"].detach().clone()
        self.init_quats = self.sms_model.gauss_params["quats"].detach().clone()
        self.init_opacities = self.sms_model.gauss_params["opacities"].detach().clone()
        # Save the initial object to world transform, and initial part to object transforms

        self.init_p2w = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        self.init_p2w_7vec = torch.zeros(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        self.init_p2w_7vec[:,3] = 1.0
        for i,g in enumerate(self.group_masks):
            gp_centroid = self.init_means[g].mean(dim=0)
            self.init_p2w_7vec[i,:3] = gp_centroid
            self.init_p2w[i,:,:] = torch.from_numpy(vtf.SE3.from_rotation_and_translation(
                vtf.SO3.identity(), (gp_centroid).cpu().numpy()
            ).as_matrix()).float().cuda()

    def initialize_obj_pose(self, niter=100, n_seeds=6, render=False):
        renders1 = []
        renders2 = []
        assert not self.is_initialized, "Can only initialize once"

        def try_opt(start_pose_adj, niter, use_depth, rndr = False, use_roi = False):
            "tries to optimize for the initial pose, returns loss and pose + GS render if requested"
            self.reset_transforms()
            whole_pose_adj = start_pose_adj.detach().clone()
            if start_pose_adj.shape[0] != len(self.group_masks):
                whole_pose_adj.repeat(len(self.group_masks), 1)
            else:
                assert start_pose_adj.shape[0] == len(self.group_masks), start_pose_adj.shape
            whole_pose_adj = torch.nn.Parameter(whole_pose_adj)
            optimizer = torch.optim.Adam([whole_pose_adj], lr=0.005)
            for i in range(niter):
                tape = wp.Tape()
                optimizer.zero_grad()
                with tape:
                    loss, outputs = self.get_optim_loss(self.frame, whole_pose_adj, use_depth, False, False, False, use_roi=use_roi)
                loss.backward()
                tape.backward()
                optimizer.step()
                if rndr:
                    with torch.no_grad():
                        if isinstance(self.frame, PosedObservation):
                            frame = self.frame.frame
                            outputs = self.sms_model.get_outputs(frame.camera, tracking=True)
                            renders2.append(outputs["rgb"].detach())
                        else:
                            frame = self.frame
                            outputs = self.sms_model.get_outputs(frame.camera, tracking=True)
                            renders1.append(outputs["rgb"].detach())
            self.is_initialized = True
            return loss, whole_pose_adj.data.detach()

        best_loss = float("inf")

        # obj_centroid = self.init_p2w_7vec[:,:3]

        # for z_rot in np.linspace(0, np.pi * 2, n_seeds):
        whole_pose_adj = torch.zeros(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        # x y z qw qx qy qz
        z_rot = 0
        quat = torch.from_numpy(vtf.SO3.from_z_radians(z_rot).wxyz).cuda()
        whole_pose_adj[:, :3] = torch.zeros(3, dtype=torch.float32, device="cuda")
        whole_pose_adj[:, 3:] = quat
        loss, final_poses = try_opt(whole_pose_adj, niter, False, render)
        # import pdb
        # pdb.set_trace()
        if loss is not None and loss < best_loss:
            best_loss = loss
            # best_outputs = outputs
            best_poses = final_poses
            
            self.set_observation(PosedObservation(rgb=self.frame.rgb, camera=self.frame.camera, dino_fn=self.frame._dino_fn, metric_depth_img=self.frame.depth))
        _, best_poses = try_opt(best_poses, 85, use_depth=True, rndr=render, use_roi=True)# do a few optimization steps with depth
        with self.render_lock:
            self.apply_to_model(
                best_poses,
                self.group_labels,
            )
        self.part_deltas = best_poses
        self.part_deltas = torch.nn.Parameter(self.part_deltas)
        self.part_deltas.requires_grad_(True)
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.config.pose_lr)

        self.prev_part_deltas = best_poses
        return renders1, renders2
    
    @property
    def objreg2objinit(self):
        return torch_posevec_to_mat(self.obj_delta).squeeze()
    
    def get_poses_relative_to_camera(self, c2w: torch.Tensor, keyframe: Optional[int] = None):
        """
        Returns the current group2cam transform as defined by the specified camera pose in world coords
        c2w: 3x4 tensor of camera to world transform

        Coordinate origin of the object aligns with world axes and centered at centroid

        returns:
        Nx4x4 tensor of obj2camera transform for each of the N groups, in the same ordering as the cluster labels
        """
        with torch.no_grad():
            assert c2w.shape == (3, 4)
            c2w = torch.cat(
                [
                    c2w,
                    torch.tensor([0, 0, 0, 1], dtype=torch.float32, device="cuda").view(
                        1, 4
                    ),
                ],
                dim=0,
            )
            obj2cam_physical_batch = torch.empty(
                len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda"
            )
            for i in range(len(self.group_masks)):
                # if keyframe is None:
                obj2world_physical = self.get_part2world_transform(i)
                # else:
                #     obj2world_physical = self.get_keyframe_part2world_transform(i, keyframe)
                obj2world_physical[:3,3] /= self.dataset_scale
                obj2cam_physical_batch[i, :, :] = c2w.inverse().matmul(obj2world_physical)
        return obj2cam_physical_batch
    
    def get_part_poses(self, keyframe: Optional[int] = None):
        """
        Returns the current group2world transform 

        Coordinate origin of the object aligns with world axes and centered at centroid
        returns:
        Nx4x4 tensor of obj2world transform for each of the N groups, in the same ordering as the cluster labels
        """
        with torch.no_grad():
            obj2cam_physical_batch = torch.empty(
                len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda"
            )
            for i in range(len(self.group_masks)):
                obj2world_physical = self.get_part2world_transform(i)
                obj2world_physical[:3,3] /= self.dataset_scale
                obj2cam_physical_batch[i, :, :] = obj2world_physical
        return obj2cam_physical_batch
    
    def get_partdelta_transform(self,i):
        """
        returns the transform from part_i to parti_i init at keyframe index given
        """
        return torch_posevec_to_mat(self.part_deltas[i].unsqueeze(0)).squeeze()
    

    def get_part2world_transform(self,i):
        """
        returns the transform from part_i to world
        """
        R_delta = torch.from_numpy(vtf.SO3(self.part_deltas[i, 3:].cpu().numpy()).as_matrix()).float().cuda()
        # we premultiply by rotation matrix to line up the 
        initial_part2world = self.get_initial_part2world(i)

        part2world = initial_part2world.clone()
        part2world[:3,:3] = R_delta[:3,:3].matmul(part2world[:3,:3])# rotate around world frame
        part2world[:3,3] += self.part_deltas[i,:3] # translate in world frame
        return part2world
    
    def get_initial_part2world(self,i):
        return self.init_p2w[i]
    
    # @profile
    def get_optim_loss(self, frame: Frame, part_deltas, use_depth, use_rgb, use_atap, use_hand_mask, use_roi = False):
        """
        Returns a backpropable loss for the given frame
        """
        feats_dict = {
            "real_rgb": [],
            "real_dino": [],
            "real_depth": [],
            "rendered_rgb": [],
            "rendered_dino": [],
            "rendered_depth": [],
            "object_mask": [],
                }
        with self.render_lock:
            self.sms_model.eval()
            self.apply_to_model(
                part_deltas, self.group_labels
            )
            if not use_roi:
                outputs = self.sms_model.get_outputs(frame.camera, tracking=True)
                feats_dict["real_rgb"] = frame.rgb
                feats_dict["real_dino"] = frame.dino_feats
                feats_dict["real_depth"] = frame.depth
                feats_dict["rendered_rgb"] = outputs['rgb']
                feats_dict["rendered_dino"] = outputs['dino']
                feats_dict["rendered_depth"] = outputs['depth']
                with torch.no_grad():
                    feats_dict["object_mask"] = outputs["accumulation"] > 0.8
                if not feats_dict["object_mask"].any():
                    return None
            else:
                # outputs = self.sms_model.get_outputs(frame.frame.camera, tracking=True)
                # feats_dict["rendered_depth"] = outputs["depth"]
                # feats_dict["real_depth"] = frame.frame.depth
                # feats_dict["object_mask"] = outputs["accumulation"] > 0.8
                for i in reversed(range(len(self.group_masks))):
                    camera = frame.roi_frames[i].camera
                    outputs = self.sms_model.get_outputs(camera, tracking=True)
                    feats_dict["real_rgb"].append(frame.roi_frames[i].rgb)
                    feats_dict["real_dino"].append(frame.roi_frames[i].dino_feats)
                    feats_dict["real_depth"].append(frame.roi_frames[i].depth)
                    feats_dict["rendered_rgb"].append(outputs['rgb'])
                    feats_dict["rendered_dino"].append(self.blur(outputs['dino'].permute(2,0,1)[None]).squeeze().permute(1,2,0))
                    feats_dict["rendered_depth"].append(outputs['depth'])
                    feats_dict["object_mask"].append(outputs['accumulation']>0.8)
                for key in feats_dict.keys():
                    # if key not in ["rendered_depth", "real_depth", "object_mask"]:
                    for i in range(len(self.group_masks)):
                        feats_dict[key][i] = feats_dict[key][i].view(-1, feats_dict[key][i].shape[-1])
                    feats_dict[key] = torch.cat(feats_dict[key])
                # import pdb; pdb.set_trace()
        # if "dino" not in outputs:
        #     self.reset_transforms()
        #     raise RuntimeError("Lost tracking")

        # dino_feats = (
        #     self.blur(outputs["dino"].permute(2, 0, 1)[None])
        #     .squeeze()
        #     .permute(1, 2, 0)
        # )
        # if use_hand_mask:
        #     loss = (frame.dino_feats - outputs["dino"])[frame.hand_mask].norm(dim=-1).mean()
        # else:
        loss = (feats_dict["real_dino"] - feats_dict["rendered_dino"]).norm(dim=-1).nanmean()
        
            # else:
            #     dino_mask = outputs["dino"].sum(dim=-1) > 1e-3
            #     loss = (frame.dino_feats[dino_mask] - outputs["dino"][dino_mask]).norm(dim=-1).nanmean()
        # THIS IS BAD WE NEED TO FIX THIS (because resizing makes the image very slightly misaligned)
        if self.use_wandb:
            wandb.log({"DINO mse_loss": loss.mean().item()})
        if use_depth:
            physical_depth = feats_dict["rendered_depth"] / self.dataset_scale
            valids = feats_dict["object_mask"] & (~feats_dict["real_depth"].isnan())
            if use_hand_mask:
                valids = valids & frame.hand_mask.unsqueeze(-1)
            # if full_img:
            physical_depth_clamped = torch.clamp(physical_depth, min=-1e8, max=2.0)
            real_depth_clamped = torch.clamp(feats_dict["real_depth"], min=-1e8, max=2.0)
            # else:
            # physical_depth_clamped = torch.clamp(physical_depth, min=-1e8, max=2.0)[valids]
            # real_depth_clamped = torch.clamp(feats_dict["real_depth"], min=-1e8, max=2.0)[valids]
            pix_loss = (physical_depth_clamped - real_depth_clamped) ** 2
            pix_loss = pix_loss[
                    valids & (pix_loss < self.config.depth_ignore_threshold**2)
                ]
            # import pdb; pdb.set_trace()
            if self.use_wandb:
                wandb.log({"depth_loss": pix_loss.mean().item()})
            if not torch.isnan(pix_loss.mean()).any():
                loss = loss + pix_loss.mean()
        if use_rgb:
            rgb_loss = 0.05 * (feats_dict["real_rgb"] - feats_dict["rendered_rgb"]).abs().mean()
            loss = loss + rgb_loss
            if self.use_wandb:
                wandb.log({"rgb_loss": rgb_loss.item()})
        if use_atap:
            weights = torch.ones(len(self.group_masks), len(self.group_masks),dtype=torch.float32,device='cuda')
            atap_loss = self.atap(weights)
            if self.use_wandb:
                wandb.log({"atap_loss": atap_loss.item()})
            loss = loss + atap_loss

        return loss, outputs
        
    # @profile
    def step(self, niter=1, use_depth=True, use_rgb=False):
        part_scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=self.config.pose_lr_final, max_steps=niter
            )
        ).get_scheduler(self.part_optimizer, self.config.pose_lr)

        for _ in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.part_deltas[:, 3:] = self.part_deltas[:, 3:] / self.part_deltas[:, 3:].norm(dim=1, keepdim=True)
                # self.prev_part_deltas = self.part_deltas.detach().clone()
            tape = wp.Tape()
            self.part_optimizer.zero_grad()

            # Compute loss
            with tape:
                if self.config.use_roi:
                    # loss = 0
                    # for obj_id in range(len(self.group_masks)):
                    #     use_depth = (obj_id==len(self.group_masks)-1)
                    loss, outputs = self.get_optim_loss(self.frame, self.part_deltas, 
                            use_depth, use_rgb, self.config.use_atap, self.config.mask_hands, True)
                # import pdb; pdb.set_trace()
            if loss is not None:
                loss.backward()
                #tape backward needs to be after loss backward since loss backward propagates gradients to the outputs of warp kernels
                tape.backward()
            # torch.nn.utils.clip_grad_norm_(self.part_deltas, self.config.clip_grad)
            self.part_optimizer.step()
            part_scheduler.step()
            part_grad_norms = self.part_deltas.grad.norm(dim=1)
            if self.use_wandb:
                for i in range(len(self.group_masks)):
                    wandb.log({f"part_delta{i} grad_norm": part_grad_norms[i].item()})
                wandb.log({"loss": loss.item()})
        # reset lr
        self.part_optimizer.param_groups[0]["lr"] = self.config.pose_lr
        
        with torch.no_grad():
            with self.render_lock:
                self.sms_model.eval()
                self.apply_to_model(
                        self.part_deltas, self.group_labels
                    )
                # if self.config.use_roi:
                #     outputs = self.sms_model.get_outputs(self.frame.frame.camera, tracking=True)
        return {k:i.detach() for k,i in outputs.items()}

    def apply_to_model(self, part_deltas, group_labels):
        """
        Takes the current part_deltas and applies them to each of the group masks
        """
        self.reset_transforms()
        new_quats = torch.empty_like(
            self.sms_model.gauss_params["quats"], requires_grad=False
        )
        new_means = torch.empty_like(
            self.sms_model.gauss_params["means"], requires_grad=True
        )
        wp.launch(
            kernel=apply_to_model,
            dim=self.sms_model.num_points,
            inputs = [
                wp.from_torch(self.init_p2w_7vec),
                wp.from_torch(part_deltas),
                wp.from_torch(group_labels),
                wp.from_torch(self.sms_model.gauss_params["means"], dtype=wp.vec3),
                wp.from_torch(self.sms_model.gauss_params["quats"]),
            ],
            outputs=[wp.from_torch(new_means, dtype=wp.vec3), wp.from_torch(new_quats)],
        )
        self.sms_model.gauss_params["quats"] = new_quats
        self.sms_model.gauss_params["means"] = new_means


    @torch.no_grad()
    def register_keyframe(self, lhands: List[trimesh.Trimesh], rhands: List[trimesh.Trimesh]):
        """
        Saves the current pose_deltas as a keyframe
        """
        # hand vertices are given in world coordinates
        w2o = self.get_registered_o2w().inverse().cpu().numpy()
        all_hands = lhands + rhands
        is_lefts = [True]*len(lhands) + [False]*len(rhands)
        if len(all_hands)>0:
            all_hands = [hand.apply_transform(w2o) for hand in all_hands]
            self.hand_frames.append(all_hands)
            self.hand_lefts.append(is_lefts)
        else:
            self.hand_frames.append([])
            self.hand_lefts.append([])

        partdeltas = torch.empty(len(self.group_masks), 4, 4, dtype=torch.float32, device="cuda")
        for i in range(len(self.group_masks)):
            partdeltas[i] = self.get_partdelta_transform(i)
        self.keyframes.append(partdeltas)
        
    @torch.no_grad()
    def apply_keyframe(self, i):
        """
        Applies the ith keyframe to the pose_deltas
        """
        deltas_to_apply = torch.empty(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
        for j in range(len(self.group_masks)):
            delta = self.keyframes[i][j]
            deltas_to_apply[j,:3] = delta[:3,3]
            deltas_to_apply[j,3:] = torch.from_numpy(vtf.SO3.from_matrix(delta[:3,:3].cpu().numpy()).wxyz).cuda()
        self.apply_to_model(deltas_to_apply, self.group_labels)
    
    def save_trajectory(self, path: Path):
        """
        Saves the trajectory to a file
        """
        torch.save({
            "keyframes": self.keyframes,
            "hand_frames": self.hand_frames,
            "hand_lefts": self.hand_lefts
        }, path)

    def load_trajectory(self, path: Path):
        """
        Loads the trajectory from a file. Sets keyframes and hand_frames.
        """
        data = torch.load(path)
        self.keyframes = [d.cuda() for d in data["keyframes"]]
        self.hand_frames = data['hand_frames']
        self.hand_lefts = data['hand_lefts']
    
    def compute_single_hand_assignment(self) -> List[int]:
        """
        returns the group index closest to the hand

        list of increasing distance. list[0], list[1] second best etc
        """
        sum_part_dists = [0]*len(self.group_masks)
        for frame_id,hands in enumerate(self.hand_frames):
            if len(hands) == 0:
                continue
            self.apply_keyframe(frame_id)
            for h_id,h in enumerate(hands):
                h = h.copy()
                h.apply_transform(self.get_registered_o2w().cpu().numpy())
                # compute distance to fingertips for each group
                for g in range(len(self.group_masks)):
                    group_mask = self.group_masks[g]
                    means = self.sms_model.gauss_params["means"][group_mask].detach()
                    # compute nearest neighbor distance from index finger to the gaussians
                    finger_position = h.vertices[349]
                    thumb_position = h.vertices[745]
                    finger_dist = (means - torch.from_numpy(np.array(finger_position)).cuda()).norm(dim=1).min().item()
                    thumb_dist = (means - torch.from_numpy(np.array(thumb_position)).cuda()).norm(dim=1).min().item()
                    closest_dist = (finger_dist + thumb_dist)/2
                    sum_part_dists[g] += closest_dist
        ids = list(range(len(self.group_masks)))
        zipped = list(zip(ids,sum_part_dists))
        zipped.sort(key=lambda x: x[1])
        return [z[0] for z in zipped]

    def compute_two_hand_assignment(self) -> List[Tuple[int,int]]:
        """
        tuple of left_group_id, right_group_id
        list of increasing distance. list[0], list[1] second best etc
        """
        # store KxG tensors storing the minimum distance to left, right hands at each frame
        left_part_dists = torch.zeros(len(self.hand_frames), len(self.group_masks),device='cuda')
        right_part_dists = torch.zeros(len(self.hand_frames), len(self.group_masks),device='cuda')
        for frame_id,hands in enumerate(self.hand_frames):
            if len(hands) == 0:
                continue
            self.apply_keyframe(frame_id)
            for h_id,h in enumerate(hands):
                h = h.copy()
                h.apply_transform(self.get_registered_o2w().cpu().numpy())
                # compute distance to fingertips for each group
                for g in range(len(self.group_masks)):
                    group_mask = self.group_masks[g]
                    means = self.sms_model.gauss_params["means"][group_mask].detach()
                    # compute nearest neighbor distance from index finger to the gaussians
                    finger_position = h.vertices[349]
                    thumb_position = h.vertices[745]
                    finger_dist = (means - torch.from_numpy(np.array(finger_position)).cuda()).norm(dim=1).min().item()
                    thumb_dist = (means - torch.from_numpy(np.array(thumb_position)).cuda()).norm(dim=1).min().item()
                    closest_dist = (finger_dist + thumb_dist)/2
                    if self.hand_lefts[frame_id][h_id]:
                        left_part_dists[frame_id,g] = closest_dist
                    else:
                        right_part_dists[frame_id,g] = closest_dist
        # Next brute force all hand-part assignments and pick the best one
        assignments = []
        for li in range(len(self.group_masks)):
            for ri in range(len(self.group_masks)):
                if li == ri:
                    continue
                dist = left_part_dists[:,li].sum() + right_part_dists[:,ri].sum()
                assignments.append((li,ri,dist))
        assignments.sort(key=lambda x: x[2])
        return [(a[0],a[1]) for a in assignments]

    def reset_transforms(self):
        with torch.no_grad():
            self.sms_model.gauss_params["means"] = self.init_means.detach().clone()
            self.sms_model.gauss_params["quats"] = self.init_quats.detach().clone()
    
    def calculate_roi(self, cam: Cameras, obj_id: int):
        """
        Calculate the ROI for the object given a certain camera pose and object index
        """
        with torch.no_grad():
            outputs = self.sms_model.get_outputs(cam,tracking=True,obj_id=obj_id)
            object_mask = outputs["accumulation"] > 0.8
            valids = torch.where(object_mask)
            valid_xs = valids[1]/object_mask.shape[1]
            valid_ys = valids[0]/object_mask.shape[0]#normalize to 0-1
            inflate_amnt = (self.config.roi_inflate*(valid_xs.max() - valid_xs.min()).item(),
                            self.config.roi_inflate*(valid_ys.max() - valid_ys.min()).item())# x, y
            xmin, xmax, ymin, ymax = max(0,valid_xs.min().item() - inflate_amnt[0]), min(1,valid_xs.max().item() + inflate_amnt[0]),\
                                max(0,valid_ys.min().item() - inflate_amnt[1]), min(1,valid_ys.max().item() + inflate_amnt[1])
            return xmin, xmax, ymin, ymax
        
    def set_frame(self, frame: Frame):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        self.frame = frame
        
    def set_observation(self, frame: PosedObservation, extrapolate_velocity = True):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        
        # assert self.is_initialized, "Must initialize first with the first frame"
        
        if self.is_initialized and self.config.use_roi:
            for obj_id in range(len(self.group_masks)):
                xmin, xmax, ymin, ymax = self.calculate_roi(frame.frame.camera, obj_id)
                frame.add_roi(xmin, xmax, ymin, ymax)
        self.frame = frame
        
        # self.add_frame(frame)
        # add another timestep of pose to the part and object poses
        # if extrapolate_velocity and self.obj_delta.shape[0] > 1:
        #     with torch.no_grad():
        #         new_parts = extrapolate_poses(self.part_deltas[-2], self.part_deltas[-1],.2)
        #         self.part_deltas = torch.nn.Parameter(torch.cat([new_parts.unsqueeze(0)], dim=0))
        # else:
        #     self.part_deltas = torch.nn.Parameter(torch.cat([self.part_deltas, self.part_deltas[-1].unsqueeze(0)], dim=0))
        # append_in_optim(self.part_optimizer, [self.part_deltas])
        # zero_optim_state(self.part_optimizer, [-2])
