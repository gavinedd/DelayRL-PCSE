# Project Setup

## Prerequisites

Ensure you have the following installed on your system:
- Git
- Python 3.10, 3.11, or 3.12
- pip (Python package manager)

### GPU Support (Optional but Recommended)
If you want to use GPU acceleration (highly recommended for faster training):

1. NVIDIA GPU with CUDA support
2. [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
3. [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

You can verify your CUDA installation by running:

```bash
nvidia-smi
```

## Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/gavinedd/DelayRL-PCSE
cd DelayRL-PCSE
```

## Initialize Submodules

This project uses Git submodules. To initialize and update them, run:

```bash
git submodule update --init
```

## Install Packages

### Basic Installation
To install all required packages (including PCSE and PCSE-Gym), simply run:

```bash
pip install -e .
```

This will install:
1. The PCSE package
2. The PCSE-Gym package
3. The custom project package
4. All other dependencies

This installation is in editable mode (-e), allowing you to make changes to the code and have them immediately reflected in your environment.

### GPU Support Installation
If you have CUDA installed and want to enable GPU support:

1. First, check your CUDA version:

```bash
nvidia-smi
```

2. Uninstall existing PyTorch installation:

```bash
pip uninstall torch torchvision torchaudio
```

3. Install PyTorch with CUDA support (choose the version matching your CUDA installation):

```bash
# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Verify your PyTorch CUDA installation:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Running Training

To start training:

```bash
# For PPO algorithm (recommended to use CPU)
python train_gavinwheat.py --device cpu --algo ppo

# For other algorithms (DQN, SAC, etc. - GPU recommended)
python train_gavinwheat.py --device cuda --algo dqn
```

Note: The PPO (Proximal Policy Optimization) algorithm performs better on CPU due to its parallel processing nature. For other algorithms like DQN or SAC, GPU acceleration is recommended for faster training.

## Viewing Training Progress

During training, the model saves progress data to the `tensorboard_logs` directory. To view these logs:

```bash
tensorboard --logdir tensorboard_logs
```

Then open `http://localhost:6006` in your web browser. The graphs show:
- Rewards over time (higher is better)
- Training losses (should decrease)
- Other training statistics

## Project Structure

```
DelayRL-PCSE/
├── pcse/           # PCSE crop simulation models (v5.5.6)
├── PCSE-Gym/       # Reinforcement learning environment
└── project/        # Custom project code
```

## Additional Information

For more details on the packages used:

- [PCSE Documentation](https://pcse.readthedocs.io/en/stable/)
- [PCSE-Gym Documentation](https://cropgym.ai/)

## License & Academic Integrity Notice

This project was created as a final project for CS 6110 at Utah State University.

### Academic Usage
This code is provided for academic review purposes. If you are a student, be aware that copying or reusing this code without proper attribution may violate your institution's academic integrity policies.

### Third-Party Components
This project uses:
- PCSE under the European Union Public License (EUPL) v1.2
- PCSE-Gym under the GNU General Public License (GPL) v3.0

The use of GPLv3 for this project ensures compliance with both component licenses.

### Authors
- Gavin Eddington
- Braxton Geary
- Chandler Justice

CS 6110 Final Project - Spring 2024
Utah State University

### Component Licenses
For the complete license texts of third-party components, see:
- [pcse/LICENSE](pcse/LICENSE) - PCSE license
- [PCSE-Gym/LICENSE](PCSE-Gym/LICENSE) - PCSE-Gym license




