from setuptools import setup, find_packages
import os

# Get the absolute path to the directory containing setup.py
root_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="project",
    version="0.1.0",
    python_requires=">=3.10,<3.13",
    packages=find_packages(),
    install_requires=[
        f"pcse @ file://{os.path.join(root_dir, 'pcse')}",
        f"cropgym @ file://{os.path.join(root_dir, 'PCSE-Gym')}",
        "stable-baselines3>=2.3.2",
        "sb3-contrib>=2.3.0",
        "gymnasium>=0.29.1",
        "numpy>=1.26.4",
        "torch>=2.3.1",
        "lib_programname>=2.0.9",
        "matplotlib",
        "tensorboard",
        "scipy",
        "tqdm",
        "pandas"
    ],
    description="Reinforcement Learning for Crop Management",
    author="Your Name",
    license="EUPL",
)
