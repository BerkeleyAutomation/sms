# sms

## Installation
```
conda create --name sms_env -y python=3.10.14
conda activate sms_env
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
cd ~/
git clone https://github.com/BerkeleyAutomation/sms.git --recurse-submodules
cd sms
conda deactivate
git clone --recurse-submodules -b feature/legs_ros_ws https://github.com/BerkeleyAutomation/L3GS
source /opt/ros/humble/setup.bash
cd L3gs/legs_ws
colcon build --packages-select lifelong_msgs
conda activate l3gs_env2
. install/setup.bash
cd ~/sms/L3GS/l3gs/
python -m pip install -e .
ns-install-cli
conda activate sms_env7
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' # From https://detectron2.readthedocs.io/en/latest/tutorials/install.html

# From https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md
cd ~/sms/sms/data/utils/Detic
pip install -r requirements.txt
cd third_party/Deformable-DETR/models/ops
./make.sh
cd ~/sms
python -m pip install -e .
```
