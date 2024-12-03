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

To install the required packages in editable mode, use the following commands:

1. Install `pcse`:

   ```bash
   pip install -e pcse
   ```

2. Install `PCSE-Gym`:

   ```bash
   pip install -e PCSE-Gym
   ```

This will allow you to make changes to the code in these directories and have them immediately reflected in your environment.

## Additional Information

For more details on using the `PCSE` and `PCSE-Gym` packages, refer to their respective documentation:

- [PCSE Documentation](https://pcse.readthedocs.io/en/stable/)
- [PCSE-Gym Documentation](https://cropgym.ai/)




