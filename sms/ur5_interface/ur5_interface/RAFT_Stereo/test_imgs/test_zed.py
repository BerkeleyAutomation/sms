from raftstereo.zed_stereo import Zed
import matplotlib.pyplot as plt
import numpy as np
cam = Zed()
left, right, depth = cam.get_frame()
plt.imsave("test_zed_depth_far.jpg", depth.cpu().numpy())
np.save("test_zed_depth_far.npy", depth.cpu().numpy())
