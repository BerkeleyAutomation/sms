# sms

## Installation
```
conda create --name sms -y python=3.10.14
conda activate sms
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
cd ~/
git clone https://github.com/BerkeleyAutomation/sms.git --recurse-submodules
cd sms
cd ~/sms
python -m pip install -e .
conda deactivate
git clone --recurse-submodules -b feature/legs_ros_ws https://github.com/BerkeleyAutomation/L3GS
source /opt/ros/humble/setup.bash
cd L3gs/legs_ws
colcon build --packages-select lifelong_msgs
conda activate sms
. install/setup.bash
<!-- cd L3GS/l3gs/ -->
python -m pip install -e .
ns-install-cli

```

## Detic
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' # From https://detectron2.readthedocs.io/en/latest/tutorials/install.html
# From https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md
cd ~/sms/sms/data/utils/Detic
pip install -r requirements.txt
cd third_party/Deformable-DETR/models/ops
./make.sh
```

## RapidsAI CUML
cuml and cudf install for HDBSCAN on cuda-11.8 and python 3.10:

```
conda install -c rapidsai -c conda-forge -c nvidia  \                       
    rapids=24.06 python=3.10 cuda-version=11.8
```

Ensure pyarrowlib is from conda env and not local python
```
python
import pyarrow
print(pyarrow.__version__)
print(pyarrow.__file__) # Make sure this comes from conda env
```
<!-- Install detectron2
```
python -m pip install --user 'git+https://github.com/facebookresearch/detectron2.git'
``` -->
<!-- Install clip
```
 pip install git+https://github.com/openai/CLIP.git
``` -->

Specific commit of gsplat is required
```
python -m pip uninstall gsplat
python -m pip install git+https://github.com/nerfstudio-project/gsplat.git@d01e6c0561f5c51d4372ff6b1d3c45f7b1e28fd5
```
