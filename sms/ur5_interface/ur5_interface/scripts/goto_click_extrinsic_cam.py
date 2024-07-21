########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################
import sys
import pyzed.sl as sl
import math
import os
import numpy as np
import sys
import math
# from matplotlib import pyplot as plt
import cv2
import time
from autolab_core import RigidTransform, Point, PointCloud
from tqdm import tqdm
from ur5py import UR5Robot

from ur5_interface.RAFT_Stereo.raftstereo.zed_stereo import Zed
import pathlib

# script_directory = pathlib.Path(__file__).parent.resolve()
# wrist_to_zed_mini_path = str(script_directory) + '/../calibration_outputs/wrist_to_cam.tf'
wrist_to_zed_mini_path = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/wrist_to_zed_mini.tf'
world_to_extrinsic_zed_path = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/calibration_outputs/world_to_extrinsic_zed.tf'
def click_event(event, u, v, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = param["img"][v, u]
        print(f"Pixel coordinates: (u={u}, v={v}) - Pixel value: {pixel_value}")

        # Display the coordinates and pixel value on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            param["img"],
            f"({u},{v})",
            (u, v),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            param["img"],
            str(pixel_value),
            (u, v + 20),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        time.sleep(0.5)
        param["u"] = u
        param["v"] = v
        # cv2.imshow('Image', param['img'])


def main():
    robot = UR5Robot(gripper=1)
    robot.gripper.close()
    home_joints = None
    # robot.move_joint(home_joints)
    # accounts for the length of the tool
    robot.set_tcp(wrist_to_tool)

    # Create a Camera object
    # zed = sl.Camera()

    # # Create a InitParameters object and set configuration parameters
    # init_params = sl.InitParameters()
    # init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    # init_params.coordinate_units = (
    #     sl.UNIT.METER
    # )  # Use meter units (for depth measurements)

    # # Open the camera
    # status = zed.open(init_params)
    # if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
    #     print("Camera Open : " + repr(status) + ". Exit program.")
    #     exit()

    # # Create and set RuntimeParameters after opening the camera
    # runtime_parameters = sl.RuntimeParameters()

    # image = sl.Mat()
    # depth = sl.Mat()
    # point_cloud = sl.Mat()

    # mirror_ref = sl.Transform()
    # mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))

    zed = Zed()
    zed_focal_length = 520 # For 1280x720
    if(abs(zed.f_ - zed_focal_length) > 10):
        print("Trying other Zed camera")
        zed = Zed()
        if(abs(zed.f_ - zed_focal_length) > 10):
            print("Please make sure Zed2 is plugged in")
            exit()
    calibration_params = zed.cam.get_camera_information().camera_configuration.calibration_parameters

    f_x = calibration_params.left_cam.fx
    f_y = calibration_params.left_cam.fy
    c_x = calibration_params.left_cam.cx
    c_y = calibration_params.left_cam.cy

    for i in tqdm(range(100)):
        previous_pose = robot.get_pose()
        # A new image is available if grab() returns SUCCESS
        # if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        #     # Retrieve left image
        #     zed.retrieve_image(image, sl.VIEW.LEFT)
        #     # Retrieve depth map. Depth is aligned on the left image
        #     zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        #     # Retrieve colored point cloud. Point cloud is aligned on the left image.
        #     zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        #     point_cloud_data = point_cloud.get_data()

        #     img = image.get_data()

        img, _, depth = zed.get_rgb_depth()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # Create a dictionary to pass to the callback function
        params = {"img": img.copy(), "u": None, "v": None}

        # Display the image and set the mouse callback
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", click_event, param=params)

        # Wait until a key is pressed and the coordinates are set
        while params["u"] is None or params["v"] is None:
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

        # Retrieve the u, v pixel coordinates from the callback
        u, v = params["u"], params["v"]

        # def on_mouse_click(event):
        #     if event.button == 1:  # Left-click
        #         u, v = event.xdata, event.ydata
        # plt.imshow(img)
        # plt.axis('image')  # Set aspect ratio to be equal
        # # Connect the click event to the figure
        # fig = plt.gcf()
        # fig.canvas.mpl_connect('button_press_event', on_mouse_click)

        # # Show the plot and wait for user input
        # plt.show()

        print(f"Selected coordinates: (u={u}, v={v})")

        Z = depth[v,u]
        X = (( u - c_x) * Z) / (f_x)
        Y = (( v - c_y) * Z) / (f_y) 
        print(f"The projected point is supposed to be: X={X}, Y={Y}, Z={Z}")
        point = Point(np.array([X,Y,Z]), frame="zed_extrinsic")
        pose = robot.get_pose()
        pose.from_frame = "tool"
        pose.to_frame = "base"
        point_in_robot = world_to_extrinsic_zed * point

        print("The point in robot frame:", point_in_robot.data)

        target_pose = RigidTransform()
        target_pose.rotation = RigidTransform.x_axis_rotation(math.pi)
        target_pose.translation = point_in_robot.data
        target_pose.translation[2] += 0.2
        
        robot.move_pose(target_pose, vel=0.2)

        target_pose.translation[2] -= 0.2
        robot.move_pose(target_pose, vel=0.2)

        input("Press Enter to continue...")
        robot.move_pose(previous_pose, vel=0.2)
    # Close the camera
    zed.close()


wrist_to_tool = RigidTransform()
# 0.1651 was old measurement is the measure dist from suction to 
# 0.1857375 Parallel Jaw gripper
wrist_to_tool.translation = np.array([0, 0, 0.1857375])
wrist_to_tool.from_frame = "tool"
wrist_to_tool.to_frame = "wrist"
cam_to_wrist = RigidTransform.load(wrist_to_zed_mini_path)
world_to_extrinsic_zed = RigidTransform.load(world_to_extrinsic_zed_path)
if __name__ == "__main__":
    main()
