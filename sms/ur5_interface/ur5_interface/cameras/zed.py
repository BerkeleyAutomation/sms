try:
    import pyzed.sl as sl
    from autolab_core import CameraIntrinsics
except:
    pass
import numpy as np



try:
    resolutions = {
        "720p": sl.RESOLUTION.HD720,
        "1080p": sl.RESOLUTION.HD1080,
        "2K": sl.RESOLUTION.HD2K,
        "VGA": sl.RESOLUTION.VGA,
    }
except:
    pass


class ZedImageCapture:
    def __init__(
        self,
        resolution="2K",
        brightness=-1,
        contrast=-1,
        hue=-1,
        saturation=-1,
        sharpness=-1,
        gain=20,
        exposure=100,
        whitebalance_temp=3300,
        fps=15,
    ):

        # Set sensing mode in FILL
        # runtime_parameters = sl.RuntimeParameters()
        # runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
        self.zed = sl.Camera()
        assert (
            resolution in resolutions
        ), "Choose a valid resolution from (`720p`, `1080p`, `2K`, `VGA`)"
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolutions[resolution]
        self.init_params.camera_fps = fps
        self.init_params.sdk_verbose = True
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        self.init_params.depth_stabilization = True
        self.init_params.depth_maximum_distance = 1
        self.init_params.depth_minimum_distance = 0.1
        self.init_params.camera_image_flip=sl.FLIP_MODE.ON

        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.sharpness = sharpness
        self.gain = gain
        self.exposure = exposure
        self.whitebalance_temp = whitebalance_temp

        # Open the camera
        self.open()
        self.runtime_parameters = sl.RuntimeParameters()

        zed_calibration_params = (
            self.zed.get_camera_information().calibration_parameters
        )
        self.fxl = zed_calibration_params.left_cam.fx
        self.fyl = zed_calibration_params.left_cam.fy
        self.fxr = zed_calibration_params.right_cam.fx
        self.fyr = zed_calibration_params.right_cam.fy
        self.hfovl = zed_calibration_params.left_cam.h_fov
        self.vfovl = zed_calibration_params.left_cam.v_fov

        self.hfovr = zed_calibration_params.right_cam.h_fov
        self.vfovr = zed_calibration_params.right_cam.v_fov
        self.cxl = zed_calibration_params.left_cam.cx
        self.cyl = zed_calibration_params.left_cam.cy
        self.cxr = zed_calibration_params.right_cam.cx
        self.cyr = zed_calibration_params.right_cam.cy
        self.stereo_translation = zed_calibration_params.T
        self.intrinsics = CameraIntrinsics(
            "zed", self.fxl, self.fyl, cx=self.cxl, cy=self.cyl
        )

    def capture_image(self, depth=False):
        img_left, img_right = sl.Mat(), sl.Mat()
        if depth:
            depth_map = sl.Mat()
            self.runtime_parameters.enable_depth = True
        else:
            self.runtime_parameters.enable_depth = False

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            self.zed.retrieve_image(img_left, sl.VIEW.LEFT)  # Retrieve the left image
            self.zed.retrieve_image(img_right, sl.VIEW.RIGHT)
            if depth:
                self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Retrieve depth

        if depth:
            return (
                self._mat_to_rgb(img_left),
                self._mat_to_rgb(img_right),
                depth_map.get_data(),
            )
        return self._mat_to_rgb(img_left), self._mat_to_rgb(img_right)

    def print_camera_settings(self):
        print("BRIGHTNESS", self.zed.get_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS))
        print("CONTRAST", self.zed.get_camera_settings(sl.VIDEO_SETTINGS.CONTRAST))
        print("HUE", self.zed.get_camera_settings(sl.VIDEO_SETTINGS.HUE))
        print("SATURATION", self.zed.get_camera_settings(sl.VIDEO_SETTINGS.SATURATION))
        print("SHARPNESS", self.zed.get_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS))
        print("GAIN", self.zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN))
        print("EXPOSURE", self.zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE))
        print(
            "WHITEBALANCE_TEMPERATURE",
            self.zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE),
        )
        print("--------------------------------------------")

    def open(self):
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            assert False, "Camera activation failed. Try unplugging and replugging"

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, self.brightness)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, self.contrast)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, self.hue)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, self.saturation)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, self.sharpness)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, self.gain)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.exposure)
        self.zed.set_camera_settings(
            sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, self.whitebalance_temp
        )

    def set_exposure(self, exp):
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exp)

    def set_gain(self, gain):
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

    def close(self):
        self.zed.close()

    @staticmethod
    def _mat_to_rgb(mat):
        img_bgr = mat.get_data()[:, :, :3]
        return np.flip(img_bgr, axis=2)
