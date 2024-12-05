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

This project uses Git submodules. To initialize and update them, run the following commands:

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

## Additional Information

For more details on using the `PCSE` and `PCSE-Gym` packages, refer to their respective documentation:

- [PCSE Documentation](https://pcse.readthedocs.io/en/stable/)
- [PCSE-Gym Documentation](https://cropgym.ai/)




