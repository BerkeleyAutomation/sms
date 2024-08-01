from sms.tracking.visualizer import GaussianVisualizer
from pathlib import Path
import time
CKPT = '/home/yujustin/Desktop/sms/sms/data/utils/Detic/outputs/20240728_panda_gripper_light_blue_jaw_4/sms-data/2024-07-28_231908/config.yml'
def main(
    config_path: Path = Path(CKPT),
    ):
    viz = GaussianVisualizer(
                config_path,
            )
    while True:
        time.sleep(0.1)
        
if __name__ == '__main__':
    main()