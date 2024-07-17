from sms.tracking.optim import Optimizer
from pathlib import Path
import torch
import viser.transforms as vtf
import numpy as np

CKPT = '/home/lifelong/sms/sms/data/utils/Detic/outputs/block_tape/sms-data/2024-07-16_160230/config.yml'
def main(
    config_path: Path = Path(CKPT),
    ):
    opt = Optimizer(
                config_path,
                # zed.get_K(),
                # l.shape[1],
                # l.shape[0], 
                # # zed.width,
                # # zed.height,
                # init_cam_pose=torch.from_numpy(
                #     vtf.SE3(
                #         wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])
                #     ).as_matrix()[None, :3, :]
                # ).float(),
            )

if __name__ == '__main__':
    main()