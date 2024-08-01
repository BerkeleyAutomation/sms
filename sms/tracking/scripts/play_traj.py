from sms.tracking.visualizer import GaussianVisualizer
from pathlib import Path
import time
CKPT = '/home/lifelong/sms/sms/data/utils/Detic/outputs/20240730_drill_battery2/sms-data/2024-07-31_032305/config.yml'
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