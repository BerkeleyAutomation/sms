# sms

## Installation
```
conda create --name sms_env -y python=3.10.14
conda activate sms_env
cd ~/
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
git clone https://github.com/BerkeleyAutomation/sms.git --recurse-submodules
cd sms
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' # From https://detectron2.readthedocs.io/en/latest/tutorials/install.html

# From https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md
cd ~/sms/sms/data/utils/Detic
pip install -r requirements.txt
cd third_party/Deformable-DETR/models/ops
./make.sh
cd ~/sms
python -m pip install -e .
```