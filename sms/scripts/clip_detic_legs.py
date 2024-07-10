import os
import json
import numpy as np
import open3d as o3d
from autolab_core import RigidTransform
import cv2
from autolab_core.points import PointCloud, RgbCloud, Point
import torchvision
from pathlib import Path
import viser
import viser.transforms as vtf
from sms.data.utils.pyramid_embedding_dataloader2 import PyramidEmbeddingDataloader
from sms.data.utils.dino_dataloader2 import DinoDataloader
from sms.encoders.image_encoder import BaseImageEncoderConfig
from sms.encoders.openclip_encoder import OpenCLIPNetworkConfig
from sms.data.utils.detic_dataloader import DeticDataloader
import time
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from typing import Tuple

DEBUG_MODE = False

class RGBDClipImagePose():
    def __init__(self,rgb_image,depth_image,pose,intrinsics,width,height):
        self.width_ = width
        self.height_ = height
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.clip_relevancy_ = None
        self.pose = pose
        self.intrinsics_ = intrinsics
        self.fx_ = self.intrinsics_[0,0]
        self.cx_ = self.intrinsics_[0,2]
        self.fy_ = self.intrinsics_[1,1]
        self.cy_ = self.intrinsics_[1,2]
        self.pointcloud_cam_frame_,self.rgbcloud_cam_frame_,self.subsample_pointcloud_cam_frame_,self.subsample_rgbcloud_cam_frame_ = self.getPointcloud(depth_image)
        
    def getPointcloud(self, depth_image):
        rows, cols = depth_image.shape
        y, x = np.meshgrid(range(rows), range(cols), indexing="ij")

        # Convert needle depth image to x,y,z pointcloud
        Z = depth_image
        X = (x - self.cx_) * Z / self.fx_
        Y = (y - self.cy_) * Z / self.fy_
        points = np.stack((X,Y,Z),axis=-1)
        rgbs = self.rgb_image
        
        # Remove all 0's
        non_zero_indices = np.all(points != [0, 0, 0], axis=-1)
        points = points[non_zero_indices]
        rgbs = rgbs[non_zero_indices]
        
        points = points.reshape(-1,3)
        rgbs = rgbs.reshape(-1,3)
        points = PointCloud(points.T,self.pose.from_frame)
        rgbs = RgbCloud(rgbs.T,self.pose.from_frame)
        (subsample_pts,subsample_indices) = points.subsample(4,random=True)
        subsample_rgbs = rgbs._data[:,subsample_indices]
        return points,rgbs,subsample_pts,subsample_rgbs
    
class ClipDeticLegs():
    def __init__(self,rgbd_clip_image_poses):
        self.rgbd_clip_image_poses_ = rgbd_clip_image_poses
        if(len(self.rgbd_clip_image_poses_) <= 0):
            print("Please load in RGBD Clip Image Poses")
            exit()
        self.height_ = self.rgbd_clip_image_poses_[0].height_
        
        self.width_ = self.rgbd_clip_image_poses_[0].width_
        self.device = 'cuda:0'
        """The device to run on"""
        self.patch_tile_size_range: Tuple[int, int] = (0.08, 0.5)
        """The range of tile sizes to sample from for patch-based training"""
        self.patch_tile_size_res: int = 7
        """The number of tile sizes to sample from for patch-based training"""
        self.patch_stride_scaler: float = 0.5
        """The stride scaler for patch-based training"""
        self.network: BaseImageEncoderConfig = OpenCLIPNetworkConfig(device=self.device)
        """specifies the vision-language self.network config"""
        self.clip_downscale_factor: int = 1
        """The downscale factor for the clip pyramid"""

        self.dino_dataloader = DinoDataloader(
                    # image_list=images,
                    device=self.device,
                    cfg={"image_shape": [self.height_,self.width_]},
                    # cache_path=dino_cache_path,
                )
        torch.cuda.empty_cache()

        self.clip_interpolator = PyramidEmbeddingDataloader(
            device=self.device,
            cfg={
                "tile_size_range": list(self.patch_tile_size_range),
                "tile_size_res": self.patch_tile_size_res,
                "stride_scaler": self.patch_stride_scaler,
                "image_shape": [self.height_,self.width_],
                "model_name": 'Optimus'
            },
            model=self.network.setup(),
            #cache_path=Path('.')
        )
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.image_encoder = self.clip_interpolator.model
        self.detic_ = DeticDataloader()
        self.detic_.create()
        self.detic_.default_vocab()

    def tt2clipinterp(self,tt_frame, clip_downscale_factor=1):
        to_pil = torchvision.transforms.ToPILImage()
        image = self.transform(to_pil(tt_frame.permute(2, 0, 1).to(torch.uint8)))
        self.clip_interpolator.generate_clip_interp(image)
        H, W = image.shape[1:]
        # scale = torch.tensor(0.1).to(device)
        scaled_height = H//clip_downscale_factor
        scaled_width = W//clip_downscale_factor

        x = torch.arange(0, scaled_width*clip_downscale_factor, clip_downscale_factor).view(1, scaled_width, 1).expand(scaled_height, scaled_width, 1).to(self.device)
        y = torch.arange(0, scaled_height*clip_downscale_factor, clip_downscale_factor).view(scaled_height, 1, 1).expand(scaled_height, scaled_width, 1).to(self.device)
        image_idx_tensor = torch.zeros(scaled_height, scaled_width, 1).to(self.device)
        positions = torch.cat((image_idx_tensor, y, x), dim=-1).view(-1, 3).to(int)
        with torch.no_grad():
            # data["clip"], data["clip_scale"] = clip_interpolator(positions, scale)[0], clip_interpolator(positions, scale)[1]
            data = self.clip_interpolator(positions)[0].view(H, W, -1)
        return data
    
    def normalize_array(self,arr):
        # Find the minimum and maximum values in the array
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Normalize the array
        normalized_arr = (arr - min_val) / (max_val - min_val)
        
        return normalized_arr
    
    def relevancy(self,img,query):
        positive = []
        positive.append(query)
        clip_frame = self.tt2clipinterp(img)
        H = clip_frame.shape[0]
        W = clip_frame.shape[1]
        self.image_encoder.set_positives(positive)
        probs1 = self.image_encoder.get_relevancy(clip_frame.view(-1, self.image_encoder.embedding_dim), 0)
        color = probs1[...,0:1].reshape([H,W])
        color_np = color.cpu().numpy()
        max_score = np.max(color_np)
        color_norm = self.normalize_array(color_np)
        color_norm = color_norm.astype('float64')
        return color_norm,max_score
    
    def tt2dino(self,tt_frame, clip_downscale_factor=1):
        im = tt_frame.permute(2, 0, 1).to(torch.float32)
        self.dino_dataloader.generate_dino_embed(im)
        H, W = im.shape[1:]
        # scale = torch.tensor(0.1).to(device)
        scaled_height = H//clip_downscale_factor
        scaled_width = W//clip_downscale_factor

        x = torch.arange(0, scaled_width*clip_downscale_factor, clip_downscale_factor).view(1, scaled_width, 1).expand(scaled_height, scaled_width, 1).to(self.device)
        y = torch.arange(0, scaled_height*clip_downscale_factor, clip_downscale_factor).view(scaled_height, 1, 1).expand(scaled_height, scaled_width, 1).to(self.device)
        image_idx_tensor = torch.zeros(scaled_height, scaled_width, 1).to(self.device)
        positions = torch.cat((image_idx_tensor, y, x), dim=-1).view(-1, 3).to(int)
        # print(positions.device)
        with torch.no_grad():
            data = self.dino_dataloader(positions.cpu()).view(H, W, -1)
        return data
    
    def getMaskedPointcloud(self,depth_image,mask):
        rows, cols = depth_image.shape
        y, x = np.meshgrid(range(rows), range(cols), indexing="ij")
        depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        
        # Convert needle depth image to x,y,z pointcloud
        Z_needle = depth_image
        X_needle = (x - self.cx_) * Z_needle / self.fx_
        Y_needle = (y - self.cy_) * Z_needle / self.fy_
        points_3d = np.stack((X_needle,Y_needle,Z_needle),axis=-1)
        points_3d_image = np.stack((X_needle,Y_needle,Z_needle),axis=2)
        # Remove all 0's
        non_zero_indices_needle = np.all(points_3d != [0, 0, 0], axis=-1)
        points_3d = points_3d[non_zero_indices_needle]
        points_3d = points_3d.reshape(-1,3)
        return points_3d,points_3d_image
    
    def findQueryClipMax(self,query):
        global_max_score = 0
        global_max_index = -1
        i = 0
        query_time = time.time()
        for rgbd_clip_image_pose in self.rgbd_clip_image_poses_:
            rgb_image = rgbd_clip_image_pose.rgb_image
            clip_norm,max_score = self.relevancy(torch.from_numpy(rgb_image).cuda(),query)
            if(max_score > global_max_score):
                global_max_score = max_score
                global_max_index = i
            rgbd_clip_image_pose.clip_relevancy_ = clip_norm
            i += 1
        print("Global max score: " + str(global_max_score))
        print("Global max index: " + str(global_max_index))
        print(time.time() - query_time)
        max_rgbd_clip_image_pose = self.rgbd_clip_image_poses_[global_max_index]
        return max_rgbd_clip_image_pose
    
    def localizeQuery(self,query):
        max_rgbd_clip_image_pose = self.findQueryClipMax(query)
        
        if DEBUG_MODE:
            plt.figure(1)
            plt.imshow(max_rgbd_clip_image_pose.rgb_image)
            plt.figure(2)
            plt.imshow(max_rgbd_clip_image_pose.clip_relevancy_,cmap='jet')
        out = self.findDeticMask(max_rgbd_clip_image_pose.clip_relevancy_,max_rgbd_clip_image_pose.rgb_image)
        if DEBUG_MODE:
            output_im = out['vis'].get_image()
            plt.figure(3)
            plt.imshow(output_im)
        weighted_average_point_world_frame = self.findWeightedAveragePoint(out,max_rgbd_clip_image_pose)
        return weighted_average_point_world_frame
    
    def findDeticMask(self,color_norm,image):
        rgb_image = image.copy()
        # Detic takes in BGR image
        bgr_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR)
        out = self.detic_.predict(bgr_image)
        return out

    def findWeightedAveragePoint(self,out,max_rgbd_clip_image_pose):
        color_norm = max_rgbd_clip_image_pose.clip_relevancy_
        rgb_image = max_rgbd_clip_image_pose.rgb_image
        depth_image = max_rgbd_clip_image_pose.depth_image
        image_intrinsics = max_rgbd_clip_image_pose.intrinsics_
        image_pose = max_rgbd_clip_image_pose.pose
        global_max_relevancy_per_pixel = 0
        global_max_relevancy_per_pixel_index = -1
        global_clip_detic_mask = None
        detic_mask = None
        i = 0
        kernel = np.ones((5,5),np.uint8)
        if DEBUG_MODE:
            if not os.path.exists('detic_masks'):
                os.makedirs('detic_masks')
        for mask in out['masks']:
            np_mask = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            np_mask = cv2.erode(np_mask,kernel,iterations=2)
            plt.imsave('detic_masks/mask_' + str(i).zfill(3) + '.png',np_mask)
            num_mask_pixels = np.count_nonzero(np_mask)
            clip_detic_mask = np.where(np_mask,color_norm,0)
            total_relevancy = np.sum(clip_detic_mask)
            relevancy_per_pixel = total_relevancy / num_mask_pixels
            if(relevancy_per_pixel > global_max_relevancy_per_pixel):
                global_max_relevancy_per_pixel = relevancy_per_pixel
                global_max_relevancy_per_pixel_index = i
                global_clip_detic_mask = clip_detic_mask
                detic_mask = np_mask
            i += 1
        points_3d,points_3d_image,colors_3d = self.getMaskedPointcloud(rgb_image,depth_image,detic_mask,image_intrinsics)
        global_clip_detic_probability_mask = global_clip_detic_mask / np.sum(global_clip_detic_mask)
        if DEBUG_MODE:
            plt.figure(4)
            plt.imshow(global_clip_detic_mask,cmap='jet')
        weighted_average = np.array([np.sum(points_3d_image[:,:,0]*global_clip_detic_probability_mask),np.sum(points_3d_image[:,:,1]*global_clip_detic_probability_mask),np.sum(points_3d_image[:,:,2]*global_clip_detic_probability_mask)])
        if DEBUG_MODE:
            object_server = viser.ViserServer()
            object_server.add_point_cloud(name="query_pointcloud",points=points_3d,colors=colors_3d,point_size=0.001)
            object_server.add_frame(name="query_localization",axes_length = 0.3, axes_radius= 0.01,position=weighted_average)
        weighted_average_point_cam_frame = Point(weighted_average,frame=image_pose.from_frame)
        weighted_average_point_world_frame = image_pose.apply(weighted_average_point_cam_frame)
        return weighted_average_point_world_frame

    def getMaskedPointcloud(self,rgb_image,depth_image,mask,image_intrinsics):
        rows, cols = depth_image.shape
        y, x = np.meshgrid(range(rows), range(cols), indexing="ij")
        depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        
        fx = image_intrinsics[0,0]
        fy = image_intrinsics[1,1]
        cx = image_intrinsics[0,2]
        cy = image_intrinsics[1,2]
        # Convert needle depth image to x,y,z pointcloud
        Z_needle = depth_image
        X_needle = (x - cx) * Z_needle / fx
        Y_needle = (y - cy) * Z_needle / fy
        points_3d = np.stack((X_needle,Y_needle,Z_needle),axis=-1)
        points_3d_image = np.stack((X_needle,Y_needle,Z_needle),axis=2)
        # Remove all 0's
        non_zero_indices_needle = np.all(points_3d != [0, 0, 0], axis=-1)
        points_3d = points_3d[non_zero_indices_needle]
        points_3d = points_3d.reshape(-1,3)
        colors_3d = rgb_image[non_zero_indices_needle]
        colors_3d = colors_3d.reshape(-1,3)
        return points_3d,points_3d_image,colors_3d
    
def constructRGBDClipImagePoses(image_folder,depth_folder,pose_file_path):
    image_files = sorted(os.listdir(image_folder))
    depth_files = sorted(os.listdir(depth_folder))
    pose_data = None
    # Load JSON data as a dictionary
    with open(pose_file_path, 'r') as json_file:
        pose_data = json.load(json_file)
    intrinsics = np.array([[pose_data['fl_x'],0,pose_data['cx']],
                        [0,pose_data['fl_y'],pose_data['cy']],
                        [0,0,1]])
    width = pose_data['w']
    height = pose_data['h']
    i = 0
    open3d_coordinate_frames = []
    nerf_frame_to_image_frame = np.array([[1,0,0,0],
                                        [0,-1,0,0],
                                        [0,0,-1,0],
                                        [0,0,0,1]])
    rgbd_clip_image_poses = []
    for image_file, depth_file in zip(image_files, depth_files):
        # Construct full file paths
        image_path = os.path.join(image_folder, image_file)
        depth_path = os.path.join(depth_folder, depth_file)
        world_to_nerf_frame = np.array(pose_data['frames'][i]['transform_matrix'])
        world_to_image_frame = world_to_nerf_frame @ nerf_frame_to_image_frame
        world_to_image_rigid_tf = RigidTransform(rotation=world_to_image_frame[:3,:3],translation=world_to_image_frame[:3,3],from_frame="image_frame_"+str(i),to_frame="world")
        rgb_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
        depth_image = np.load(depth_path)
        rgbd_image_pose = RGBDClipImagePose(rgb_image,depth_image,world_to_image_rigid_tf,intrinsics,width,height)
        rgbd_clip_image_poses.append(rgbd_image_pose)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        coordinate_frame.transform(world_to_image_frame)
        open3d_coordinate_frames.append(coordinate_frame)
        i += 1
    return rgbd_clip_image_poses

def visualizePointcloudViser(rgbd_clip_image_poses):
    server = viser.ViserServer()
    i = 0
    global_pointcloud = None
    for rgbd_clip_image_pose in rgbd_clip_image_poses:
        rgb_image = rgbd_clip_image_pose.rgb_image
        pose = rgbd_clip_image_pose.pose
        pointcloud_cam_frame = rgbd_clip_image_pose.subsample_pointcloud_cam_frame_
        rgbcloud_cam_frame = rgbd_clip_image_pose.subsample_rgbcloud_cam_frame_
        server.add_camera_frustum(name="image_frames/"+pose.from_frame,fov=90,aspect=rgb_image.shape[1]/rgb_image.shape[0],scale=0.01,position=pose.translation,wxyz=vtf.SO3.from_matrix(pose.rotation).wxyz)
        #server.add_frame(name="coordinate_frames/"+pose.from_frame,axes_length = 0.05, axes_radius= 0.0025,position=pose.translation,wxyz=vtf.SO3.from_matrix(pose.rotation).wxyz)
        pointcloud_world_frame = pose.apply(pointcloud_cam_frame)
        if(global_pointcloud is None):
            global_pointcloud = pointcloud_world_frame.data.T
        else:
            global_pointcloud = np.vstack((global_pointcloud,pointcloud_world_frame.data.T))
        server.add_point_cloud(name="pointcloud/" + pose.from_frame,points=pointcloud_world_frame.data.T,colors=rgbcloud_cam_frame.T,point_size=0.005)
        i += 1
    # Create an Open3D PointCloud object
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(global_pointcloud)

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud("sparse_pc.ply", pointcloud_o3d)

def visualizePointcloudWithQueryViser(rgbd_clip_image_poses,query_point):
    object_server = viser.ViserServer()
    i = 0
    for rgbd_clip_image_pose in rgbd_clip_image_poses:
        rgb_image = rgbd_clip_image_pose.rgb_image
        pose = rgbd_clip_image_pose.pose
        pointcloud_cam_frame = rgbd_clip_image_pose.subsample_pointcloud_cam_frame_
        rgbcloud_cam_frame = rgbd_clip_image_pose.subsample_rgbcloud_cam_frame_
        object_server.add_camera_frustum(name="image_frames/"+pose.from_frame,fov=90,aspect=rgb_image.shape[1]/rgb_image.shape[0],scale=0.01,position=pose.translation,wxyz=vtf.SO3.from_matrix(pose.rotation).wxyz)
        #server.add_frame(name="coordinate_frames/"+pose.from_frame,axes_length = 0.05, axes_radius= 0.0025,position=pose.translation,wxyz=vtf.SO3.from_matrix(pose.rotation).wxyz)
        pointcloud_world_frame = pose.apply(pointcloud_cam_frame)
        object_server.add_point_cloud(name="pointcloud/" + pose.from_frame,points=pointcloud_world_frame.data.T,colors=rgbcloud_cam_frame.T,point_size=0.005)
        i += 1
    object_server.add_frame(name="query",axes_length = 0.3, axes_radius= 0.01,position=query_point.data)

if __name__ == "__main__":
    start_time = time.time()
    image_folder = '/home/lifelong/sms_w_2d/stapler_cup_apple_2_scissors/img'
    depth_folder = '/home/lifelong/sms_w_2d/stapler_cup_apple_2_scissors/depth'
    pose_file_path = '/home/lifelong/sms_w_2d/stapler_cup_apple_2_scissors/transforms.json'
    rgbd_clip_image_poses = constructRGBDClipImagePoses(image_folder,depth_folder,pose_file_path)
    visualizePointcloudViser(rgbd_clip_image_poses)
    clip_detic_legs = ClipDeticLegs(rgbd_clip_image_poses)
    another_query = True
    end_time = time.time()
    print("Non query time: " + str(end_time - start_time))
    while(another_query):
        query = input("Please type in a query\n")
        weighted_average_point_world_frame = clip_detic_legs.localizeQuery(query)
        if DEBUG_MODE:
            plt.show()
        visualizePointcloudWithQueryViser(rgbd_clip_image_poses,weighted_average_point_world_frame)
        response = input("Do you want to input another query? Type y/n")
        if(response == 'y'):
            another_query = True
        elif(response == 'n'):
            another_query = False
        else:
            print("Please type y or n next time\n")
            exit()
