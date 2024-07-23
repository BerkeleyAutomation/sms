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
from sms.tracking.frame import Frame
import time
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

class RigidGroupOptimizer:
    use_depth: bool = True
    rank_loss_mult: float = 0.1
    rank_loss_erode: int = 5
    depth_ignore_threshold: float = 0.1  # in meters
    use_atap: bool = True
    pose_lr: float = 0.003
    pose_lr_final: float = 0.0005
    mask_hands: bool = False
    do_obj_optim: bool = True

    init_p2w: torch.Tensor
    """From: part, To: object. in current world frame. Part frame is centered at part centroid, and object frame is centered at object centroid."""

    def __init__(
        self,
        sms_model: smsGaussianSplattingModel,
        # dino_loader: DinoDataloader,
        group_masks: List[torch.Tensor],
        group_labels: torch.Tensor,
        dataset_scale: float,
        render_lock = nullcontext(),
    ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
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
        k = 13
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.pose_lr)
        self.keyframes = []
        #hand_frames stores a list of hand vertices and faces for each keyframe, stored in the OBJECT COORDINATE FRAME
        self.hand_frames = []
        # lock to prevent blocking the render thread if provided
        self.render_lock = render_lock
        if self.use_atap:
            self.atap = ATAPLoss(sms_model, group_masks, group_labels, self.dataset_scale)
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
        renders = []
        assert not self.is_initialized, "Can only initialize once"

        def try_opt(start_pose_adj, niter, use_depth, rndr = False):
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
                    loss = self.get_optim_loss(self.frame, whole_pose_adj, use_depth, False, False, False, False)
                loss.backward()
                tape.backward()
                optimizer.step()
                if rndr:
                    with torch.no_grad():
                        dig_outputs = self.sms_model.get_outputs(self.frame.camera, tracking=True)
                    renders.append(dig_outputs["rgb"].detach())
            self.is_initialized = True
            return loss, whole_pose_adj.data.detach()

        best_loss = float("inf")

        # obj_centroid = self.init_p2w_7vec[:,:3]

        for z_rot in np.linspace(0, np.pi * 2, n_seeds):
            whole_pose_adj = torch.zeros(len(self.group_masks), 7, dtype=torch.float32, device="cuda")
            # x y z qw qx qy qz
            quat = torch.from_numpy(vtf.SO3.from_z_radians(z_rot).wxyz).cuda()
            whole_pose_adj[:, :3] = torch.zeros(3, dtype=torch.float32, device="cuda")
            whole_pose_adj[:, 3:] = quat
            loss, final_poses = try_opt(whole_pose_adj, niter, False, render)
            if loss is not None and loss < best_loss:
                best_loss = loss
                # best_outputs = dig_outputs
                best_poses = final_poses
        _, best_poses = try_opt(best_poses, 85, True, True)# do a few optimization steps with depth
        with self.render_lock:
            self.apply_to_model(
                best_poses,
                self.group_labels,
            )
        self.part_deltas = best_poses
        self.part_deltas = torch.nn.Parameter(self.part_deltas)
        self.part_deltas.requires_grad_(True)
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.pose_lr)

        self.prev_part_deltas = best_poses
        return renders
    
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
        part2world[:3,3] += self.part_deltas[i,:3] # * VISER_NERFSTUDIO_SCALE_RATIO # translate in world frame
        return part2world
    
    def get_initial_part2world(self,i):
        return self.init_p2w[i]
  
    def get_optim_loss(self, frame: Frame, part_deltas, use_depth, use_rgb, use_atap, use_hand_mask, do_obj_optim):
        """
        Returns a backpropable loss for the given frame
        """
        with self.render_lock:
            self.sms_model.eval()
            # print("apply_to_model")
            self.apply_to_model(
                part_deltas, self.group_labels
            )
            # print("Getting output")
            # start_time = time.time()
            dig_outputs = self.sms_model.get_outputs(frame.camera, tracking=True)
            # print("Get output time: ", time.time()-start_time)
        if "dino" not in dig_outputs:
            self.reset_transforms()
            raise RuntimeError("Lost tracking")
        with torch.no_grad():
            object_mask = dig_outputs["accumulation"] > 0.9
        dino_feats = (
            self.blur(dig_outputs["dino"].permute(2, 0, 1)[None])
            .squeeze()
            .permute(1, 2, 0)
        )
        if use_hand_mask:
            mse_loss = (frame.dino_feats - dino_feats)[frame.hand_mask].norm(dim=-1)
        else:
            mse_loss = (frame.dino_feats - dino_feats).norm(dim=-1)
        # THIS IS BAD WE NEED TO FIX THIS (because resizing makes the image very slightly misaligned)
        loss = mse_loss.mean()
        if use_depth:
            if frame.metric_depth:
                physical_depth = dig_outputs["depth"] / self.dataset_scale
                valids = object_mask & (~frame.depth.isnan())
                if use_hand_mask:
                    valids = valids & frame.hand_mask.unsqueeze(-1)
                pix_loss = (physical_depth - frame.depth) ** 2
                pix_loss = pix_loss[
                    valids & (pix_loss < self.depth_ignore_threshold**2)
                ]
                loss = loss + pix_loss.mean()
            else:
                # This is ranking loss for monodepth (which is disparity)
                frame_depth = 1 / frame.depth # convert disparity to depth
                N = 30000
                # erode the mask by like 10 pixels
                object_mask = object_mask & (~frame_depth.isnan())
                object_mask = object_mask & (dig_outputs['depth'] > .05)
                object_mask = kornia.morphology.erosion(
                    object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(), torch.ones((self.rank_loss_erode, self.rank_loss_erode), device='cuda')
                ).squeeze().bool()
                if use_hand_mask:
                    object_mask = object_mask & frame.hand_mask
                valid_ids = torch.where(object_mask)
                rand_samples = torch.randint(
                    0, valid_ids[0].shape[0], (N,), device="cuda"
                )
                rand_samples = (
                    valid_ids[0][rand_samples],
                    valid_ids[1][rand_samples],
                )
                rend_samples = dig_outputs["depth"][rand_samples]
                mono_samples = frame_depth[rand_samples]
                rank_loss = depth_ranking_loss(rend_samples, mono_samples)
                loss = loss + self.rank_loss_mult*rank_loss
        if use_rgb:
            loss = loss + 0.05 * (dig_outputs["rgb"] - frame.rgb).abs().mean()
        if use_atap:
            weights = torch.ones(len(self.group_masks), len(self.group_masks),dtype=torch.float32,device='cuda')
            atap_loss = self.atap(weights)
            loss = loss + atap_loss
        if do_obj_optim:
            # add regularizer on the poses to not move much
            loss = loss + 0.02 * self.part_deltas[:,:3].norm(dim=-1).mean() + .01 * 2 * torch.acos(0.99*self.part_deltas[:,3]).mean()
        return loss
        
    def step(self, niter=1, use_depth=True, use_rgb=False):
        part_scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=self.pose_lr_final, max_steps=niter
            )
        ).get_scheduler(self.part_optimizer, self.pose_lr)
        # if self.do_obj_optim:
        #     obj_scheduler = ExponentialDecayScheduler(
        #         ExponentialDecaySchedulerConfig(
        #             lr_final=self.pose_lr_final, max_steps=niter
        #         )
        #     ).get_scheduler(self.obj_optimizer, self.pose_lr)
        for _ in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.part_deltas[:, 3:] = self.part_deltas[:, 3:] / self.part_deltas[:, 3:].norm(dim=1, keepdim=True)
                # self.obj_delta[0, 3:] = self.obj_delta[0, 3:] / self.obj_delta[0, 3:].norm()
                self.prev_part_deltas = self.part_deltas.detach().clone()
            tape = wp.Tape()
            self.part_optimizer.zero_grad()
            # if self.do_obj_optim:
            #     self.obj_optimizer.zero_grad()
            # Compute loss
            with tape:
                # loss = self.get_optim_loss(self.frame, self.obj_delta, self.part_deltas, 
                #     use_depth, use_rgb, self.use_atap, self.mask_hands, self.do_obj_optim)
                loss = self.get_optim_loss(self.frame, self.part_deltas, 
                    use_depth, use_rgb, self.use_atap, self.mask_hands, self.do_obj_optim)
            loss.backward()
            #tape backward needs to be after loss backward since loss backward propagates gradients to the outputs of warp kernels
            tape.backward()
            self.part_optimizer.step()
            part_scheduler.step()
            # if self.do_obj_optim:
            #     self.obj_optimizer.step()
            #     obj_scheduler.step()
        # reset lr
        self.part_optimizer.param_groups[0]["lr"] = self.pose_lr
        
        # # If tracking results in nan, propogate previous pose
        # if torch.isnan(self.part_deltas).any().item():
        #     for i, p_deltas in enumerate(self.part_deltas):
        #         if torch.isnan(p_deltas).any().item():
        #             self.part_deltas[i] = self.prev_part_deltas[i]
        #     self.part_deltas = torch.nn.Parameter(self.part_deltas)
        #     self.part_deltas.requires_grad_(True)
        
        with torch.no_grad():
            with self.render_lock:
                self.sms_model.eval()
                self.apply_to_model(
                        self.part_deltas, self.group_labels
                    )
                # print("Getting output")
                # start_time = time.time()
                full_outputs = self.sms_model.get_outputs(self.frame.camera, tracking=True)
                # print("Get output time: ", time.time()-start_time)
        return {k:i.detach() for k,i in full_outputs.items()}

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
        # assert objdelta.shape == (1,7), objdelta.shape
        wp.launch(
            kernel=apply_to_model,
            dim=self.sms_model.num_points,
            inputs = [
                # wp.from_torch(self.init_o2w_7vec),
                wp.from_torch(self.init_p2w_7vec),
                # wp.from_torch(objdelta),
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
    
    def set_frame(self, frame: Frame):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        self.frame = frame