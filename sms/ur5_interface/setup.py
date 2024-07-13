"""
Setup of UR5 Interface used for Arms and Legs.
Author: Kush Hari kush_hari@berkeley.edu
"""
from setuptools import setup,find_packages

setup(
    name="ur5_interface",
    version="0.1.0",
    description="UR5 interface building on ur5py to also include camera calibration and some hemispherical movements for NERFs and Gaussian Splats",
    author="Kush Hari",
    author_email="kush_hari@berkeley.edu",
    include_package_data=True,
    packages=find_packages(),
)
