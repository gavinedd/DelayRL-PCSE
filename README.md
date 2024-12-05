# Project Setup

## Prerequisites

Ensure you have the following installed on your system:
- Git
- Python 3.10, 3.11, or 3.12
- pip (Python package manager)

## Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/gavinedd/cs6110Final
cd cs6110Final
```

## Initialize Submodules

This project uses Git submodules. To initialize and update them, run:

```bash
git submodule update --init
```

## Install Packages

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
cs6110Final/
├── pcse/           # PCSE crop simulation models (v5.5.6)
├── PCSE-Gym/       # Reinforcement learning environment
└── project/        # Custom project code for CS 6110 Final
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




