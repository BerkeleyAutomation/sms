import pyzed.sl as sl
import numpy as np
from autolab_core import CameraIntrinsics, PointCloud, RgbCloud
from raftstereo.raft_stereo import *
from raftstereo.utils.utils import InputPadder
import argparse
class Zed:
    def __init__(self, recording_file=None, start_time=0.0):
        init = sl.InitParameters()
        self.height_ = None
        self.width_ = None
        if recording_file is not None:
            init.set_from_svo_file(recording_file)
            # disable depth
            init.camera_image_flip = sl.FLIP_MODE.OFF
            init.depth_mode = sl.DEPTH_MODE.NONE
            init.camera_resolution = sl.RESOLUTION.HD1080
            init.sdk_verbose = 1
            init.camera_fps = 30
        else:
            init.camera_resolution = sl.RESOLUTION.HD720  # sl.RESOLUTION.HD1080
            self.height_ = 720
            self.width_ = 1280
            init.sdk_verbose = 1
            init.camera_fps = 30
            # flip camera
            init.camera_image_flip = sl.FLIP_MODE.OFF
            init.depth_mode = sl.DEPTH_MODE.NONE
            init.depth_minimum_distance = 100  # millimeters
        self.cam = sl.Camera()
        init.camera_disable_self_calib = True
        status = self.cam.open(init)
        # manually sets exposure
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 30)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 60)
        self.recording_file = recording_file
        self.start_time = start_time
        if recording_file is not None:
            fps = self.cam.get_camera_information().camera_configuration.fps
            self.cam.set_svo_position(int(start_time * fps))
        if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
            print("Camera Open : " + repr(status) + ". Exit program.")
            exit()
        else:
            print("Opened camera")
            print("Current Exposure is set to: ",
                self.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
            )
            print("Current Gain is set to: ",
                self.cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN),
            )
        left_cx = self.get_K(cam="left")[0, 2]
        right_cx = self.get_K(cam="right")[0, 2]
        self.cx_diff = right_cx - left_cx  # /1920
        self.f_ = self.get_K(cam="left")[0,0]
        self.cx_ = left_cx
        self.cy_ = self.get_K(cam="left")[1,2]
        self.Tx_ = self.get_stereo_transform()[0,3]
        self.valid_iters_ = 32
        self.padder_ = InputPadder(torch.empty(1,3,self.height_,self.width_).shape,divis_by=32)
        self.model = create_raft()
        # cam.self.get_stereo_transform()[0,3]

    def load_image_raft(self,im):
        img = torch.from_numpy(im).permute(2,0,1).float()
        return img[None].to('cuda')
    
    def get_depth_image_and_pointcloud(self,left_img,right_img,from_frame):
        with torch.no_grad():
            image1 = self.load_image_raft(left_img)
            image2 = self.load_image_raft(right_img)
            image1, image2 = self.padder_.pad(image1, image2)
            _, flow_up = self.model(image1, image2, iters=self.valid_iters_, test_mode=True)
            flow_up = self.padder_.unpad(flow_up).squeeze()

            flow_up_np = -flow_up.detach().cpu().numpy().squeeze()
            
            depth_image = (self.f_ * self.Tx_) / abs(flow_up_np + self.cx_diff)  
            rows, cols = depth_image.shape
            y, x = np.meshgrid(range(rows), range(cols), indexing="ij")

            # Convert needle depth image to x,y,z pointcloud
            Z = depth_image
            X = (x - self.cx_) * Z / self.f_
            Y = (y - self.cy_) * Z / self.f_
            points = np.stack((X,Y,Z),axis=-1)
            rgbs = left_img
            
            # Remove all 0's
            non_zero_indices = np.all(points != [0, 0, 0], axis=-1)
            points = points[non_zero_indices]
            rgbs = rgbs[non_zero_indices]
            
            points = points.reshape(-1,3)
            rgbs = rgbs.reshape(-1,3)
            points = PointCloud(points.T,from_frame)
            rgbs = RgbCloud(rgbs.T,from_frame)
            return depth_image,points,rgbs 

    def get_frame(self, depth=True, cam="left"):
        res = sl.Resolution()
        res.width = 1280
        res.height = 720
        if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
            left_rgb = sl.Mat()
            right_rgb = sl.Mat()
            self.cam.retrieve_image(left_rgb, sl.VIEW.LEFT, sl.MEM.CPU, res)
            self.cam.retrieve_image(right_rgb, sl.VIEW.RIGHT, sl.MEM.CPU, res)
            left, right = (
                torch.from_numpy(
                    np.flip(left_rgb.get_data()[..., :3], axis=2).copy()
                ).cuda(),
                torch.from_numpy(
                    np.flip(right_rgb.get_data()[..., :3], axis=2).copy()
                ).cuda(),
            )
            if depth:
                left_torch, right_torch = left.permute(2, 0, 1), right.permute(2, 0, 1)

                if cam == "left":
                    flow = raft_inference(left_torch, right_torch, self.model)
                else:
                    right_torch = torch.flip(right_torch, dims=[2])
                    left_torch = torch.flip(left_torch, dims=[2])
                    flow = raft_inference(right_torch, left_torch, self.model)

                fx = self.get_K()[0, 0]  # *1280/1920
                depth = (
                    fx * self.get_stereo_transform()[0, 3] / (flow.abs() + self.cx_diff)
                )  # *1280)

                if cam != "left":
                    depth = torch.flip(depth, dims=[1])
            else:
                depth = None
            return left, right, depth
        elif self.cam.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of recording file")
            return None, None, None
        else:
            raise RuntimeError("Could not grab frame")

    def get_K(self, cam="left"):
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        if cam == "left":
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        K = np.array(
            [
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1],
            ]
        )
        return K

    def get_intr(self, cam="left"):
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        if cam == "left":
            intrinsics = calib.left_cam
        else:
            intrinsics = calib.right_cam
        return CameraIntrinsics(
            frame="zed",
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            width=1280,
            height=720,
        )

    def get_stereo_transform(self):
        transform = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.m
        )
        transform[:3, 3] /= 1000  # convert to meters
        return transform

    def start_record(self, out_path):
        recordingParameters = sl.RecordingParameters()
        recordingParameters.compression_mode = sl.SVO_COMPRESSION_MODE.H264
        recordingParameters.video_filename = out_path
        err = self.cam.enable_recording(recordingParameters)

    def stop_record(self):
        self.cam.disable_recording()

    def get_rgb_depth(self, cam="left"):
        """
        added function to meet current API, should be refactored
        """
        left, right, depth = self.get_frame(cam=cam)
        return left.cpu().numpy(), right.cpu().numpy(), depth.cpu().numpy()
    
    def get_rgb(self, cam="left"):
        """
        added function to meet current API, should be refactored
        """
        left, right,_ = self.get_frame(depth=False,cam=cam)
        return left.cpu().numpy(), right.cpu().numpy()

    def get_ns_intrinsics(self):
        calib = (
            self.cam.get_camera_information().camera_configuration.calibration_parameters
        )
        calibration_parameters_l = calib.left_cam
        return {
            # "w": self.cam.get_camera_information().camera_resolution.width,
            # "h": self.cam.get_camera_information().camera_resolution.height,
            "w": 1280,
            "h": 720,
            "fl_x": calibration_parameters_l.fx,
            "fl_y": calibration_parameters_l.fy,
            "cx": calibration_parameters_l.cx,
            "cy": calibration_parameters_l.cy,
            "k1": calibration_parameters_l.disto[0],
            "k2": calibration_parameters_l.disto[1],
            "p1": calibration_parameters_l.disto[3],
            "p2": calibration_parameters_l.disto[4],
            "camera_model": "OPENCV",
        }

    def get_zed_depth(self):
        if self.cam.grab() == sl.ERROR_CODE.SUCCESS:
            depth = sl.Mat()
            self.cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            return depth.get_data()
        else:
            raise RuntimeError("Could not grab frame")


    def close(self):
        self.cam.close()
        self.cam = None
        print("Closed camera")

    def reopen(self):
        if self.cam is None:
            init = sl.InitParameters()
            if self.recording_file is not None:
                init.set_from_svo_file(self.recording_file)
                # disable depth
                init.camera_image_flip = sl.FLIP_MODE.OFF
                init.depth_mode = sl.DEPTH_MODE.NONE
                init.camera_resolution = sl.RESOLUTION.HD1080
                init.sdk_verbose = 1
                init.camera_fps = 30
            else:
                init.camera_resolution = sl.RESOLUTION.HD720  # sl.RESOLUTION.HD1080
                init.sdk_verbose = 1
                init.camera_fps = 30
                # flip camera
                init.camera_image_flip = sl.FLIP_MODE.OFF
                init.depth_mode = sl.DEPTH_MODE.NONE
                init.depth_minimum_distance = 100  # millimeters
            self.cam = sl.Camera()
            # manually sets exposure
            self.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 30)
            self.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 60)
            init.camera_disable_self_calib = True
            status = self.cam.open(init)
            if self.recording_file is not None:
                fps = self.cam.get_camera_information().camera_configuration.fps
                self.cam.set_svo_position(int(self.start_time * fps))
            if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
                print("Camera Open : " + repr(status) + ". Exit program.")
                exit()
            else:
                print("Opened camera")
                print(
                    "Current Exposure is set to: ",
                    self.cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE),
                )
            self.model = create_raft()
            left_cx = self.get_K(cam="left")[0, 2]
            right_cx = self.get_K(cam="right")[0, 2]
            self.cx_diff = right_cx - left_cx  # /1920