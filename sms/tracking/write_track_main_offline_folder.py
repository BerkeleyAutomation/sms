import pyzed.sl as sl
from sms.tracking.tri_zed import Zed
import time
import datetime
import os
import cv2
import numpy as np
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
folder_name = '/home/lifelong/sms/sms/data/utils/Detic/outputs/20240728_panda_gripper_light_blue_jaw_4/offline_video'
image_folder_name = folder_name + '/img'
depth_folder_name = folder_name + '/depth'
create_folder_if_not_exists(folder_name)
create_folder_if_not_exists(image_folder_name)
create_folder_if_not_exists(depth_folder_name)
extrinsic_zed_id = 22008760
zed = Zed(cam_id=extrinsic_zed_id,is_res_1080=True) # Initialize ZED
i = 0
while True:
    start_time = time.time()
    # l, r = zed.prime_get_frame()
    l,_,depth = zed.get_frame(depth=False)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Frequency: " + str(float(1./time_taken)))
    timestamp = datetime.datetime.now().timestamp()
    import pdb
    pdb.set_trace()
    cv2.imwrite(image_folder_name+'/img_'+str(i).zfill(10)+'.png',cv2.cvtColor(l.detach().cpu().numpy(),cv2.COLOR_BGR2RGB))
    np.save(depth_folder_name+'/depth_'+str(i).zfill(10)+'.npy',depth.detach().cpu().numpy())
    i += 1    