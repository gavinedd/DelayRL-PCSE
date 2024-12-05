from setuptools import setup, find_packages
import os

# Get the absolute path to the directory containing setup.py
root_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pcse @ file://" + os.path.join(root_dir, "pcse"),
        "cropgym @ file://" + os.path.join(root_dir, "PCSE-Gym"),
        "stable-baselines3",
        "sb3-contrib",
        "gymnasium",
        "numpy",
        "torch",
        "lib_programname"
    ],
) 