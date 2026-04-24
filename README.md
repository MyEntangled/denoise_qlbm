# Quantum Lattice Boltzmann with Denoising Collision Operators

[![Paper](https://img.shields.io/badge/Paper-Preprint-blue.svg)](#) 
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/1083855369.svg)](https://doi.org/10.5281/zenodo.19482608)

This repository contains the official Python implementation for the paper **"Quantum Lattice Boltzmann with Denoising Collision Operators"** by Erio Trong Duong, Matthias Möller, and Norbert Hosters. 

The codebase provides a framework to simulate fluid dynamics using both Classical and Quantum Lattice Boltzmann Methods (CLBM/QLBM). Specifically, it introduces a quantum algorithm for LBM that employs denoising collision operators designed to approximate the effects of the nonlinear collision dynamics in the standard LBM formulation.

## Directory Structure

The project code is organized into dedicated modules for solvers and test configurations:

```text
denoise_qlbm/
├── src/
│   ├── run_experiment.py      # Main entry point for running all simulations
│   ├── lattices/              # Configurations for common LBM lattices from 1D to 3D.
│   ├── clbm/                  # Classical LBM solver definitions
│   ├── qlbm/                  # Quantum LBM solver definitions
│   └── testcases/             # Domain setup and boundary conditions
│       ├── fourier.py         # Settings for 1D Fourier flow
│       ├── gaussian.py        # Settings for Gaussian pulse propagation
│       ├── taylor_green.py    # Settings for 2D Taylor-Green vortex decay
│       └── cylinder.py        # Settings for 2D flow past a cylinder
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Available Test Cases

We provide four distinct fluid dynamic benchmarks to evaluate the classical and quantum LBM implementations. The physical parameters, lattice dimensions, and boundary conditions for each case are entirely self-contained in their respective files within the `src/testcases/` directory:

1. **`fourier`**: Advection-diffusion of a 1D Fourier mode.
2. **`gaussian`**: Advection-diffusion of a 2D Gaussian hill.
3. **`taylor_green`**: Flow of a 2D Taylor-Green decaying vortex.
4. **`cylinder`**: Flow around a 2D cylinder.

> **Note on Reproducibility:** Running the test case files with their default parameters will automatically reproduce the paper's baseline numerical results. To reproduce the specific parameter variations shown in the **`gaussian`** and **`cylinder`** plots, use the `--parameters` argument with the exact values provided in the examples below.

## Usage

Users can run classical, quantum, or both LBM experiments side-by-side using the main execution script: `src/run_experiment.py`. 

### Command-Line Arguments

* `--testcase` **(Required)**: The benchmark to run (`fourier`, `gaussian`, `cylinder`, `taylor_green`).
* `--steps` **(Required)**: The number of time steps to simulate.
* `--run-classical`: Flag to execute the Classical LBM solver.
* `--run-quantum`: Flag to execute the Quantum LBM solver.
  * *(Note: If neither solver flag is provided, the script defaults to running **both** solvers sequentially).*
* `--parameters`: Optional space-separated values to override test case specific setups.
* `--save-every`: Store intermediate distributions every k steps (Default: 1).
* `--outdir`: Directory to save the output files (Default: current directory `.`).

### Examples

**1. Run both Classical and Quantum solvers with default parameters to reproduce numerical results shown in the paper (applied to any test case):**
```bash
python src/run_experiment.py --testcase fourier --steps 10000
```

**2. Run ONLY the Classical LBM for flow past a cylinder:**
```bash
python src/run_experiment.py --testcase cylinder --steps 10000 --run-classical
```

**3. Run the solvers with custom parameters to reproduce additional numerical results for **`gaussian`** and **`cylinder`** plots
```bash
python src/run_experiment.py --testcase gaussian --steps 10000 --parameters 1 5 --save-every 20
python src/run_experiment.py --testcase gaussian --steps 10000 --parameters 1 20 --save-every 20
python src/run_experiment.py --testcase cylinder --steps 10000 --parameters 0.05773502691 0. --save-every 20
python src/run_experiment.py --testcase cylinder --steps 10000 --parameters 0.05773502691 0.66666666667 --save-every 20
```

## Output Format

The script computes and stores the full trajectories of the distributions (F), macroscopic density (rho), and macroscopic velocity (u). 

Upon completion, all data is compiled into a compressed NumPy archive (`.npz`) saved to the specified `--outdir`. The filename includes the test case, custom parameters (if specified), and a timestamp for easy tracking:
`lbm_{testcase}_{custom_params}_{timestamp}.npz`

You can load and analyze this data in Python using:
```python
import numpy as np
data = np.load("lbm_taylor_green_20260408_153000.npz", allow_pickle=True)

# Access trajectories
classical_data = data["classical_traj"].item()
quantum_data = data["quantum_traj"].item()
metadata = data["meta"].item()

print(classical_data["rho"].shape) # Density over time
```

## Visualization
To facilitate the reproduction and analysis of our numerical results, we have provided four dedicated Jupyter notebooks alongside the source code. Readers can load the .npz trajectory files generated by the execution script into these notebooks to process the data and recreate all the plots and visual comparisons presented in the paper.

## Citation
If you use this code or our methodology in your research, please consider citing our paper:

```bibtex
@article{duong2026quantum,
  title={Quantum Lattice Boltzmann with Denoising Collision Operators},
  author={Duong, Trong and M{\"o}ller, Matthias and Hosters, Norbert},
  journal={arXiv preprint arXiv:2604.09997},
  year={2026}
}
```