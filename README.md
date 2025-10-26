# Quantum Lattice Boltzmann Methods (QLBM) with Amplitude Encoding

This repository provides a Python implementation of Quantum Lattice Boltzmann Methods (QLBM) focused on amplitude encoding. The main codebase is located in `/src`, and supports research and development in quantum algorithms for computational fluid dynamics.

## Features
- **Collision Operators**: Two types of collision operators are implemented:
  - **Denoising-based**: Manifold-aware denoising operators for equilibrium distributions, designed to suppress noise and retract states onto the equilibrium manifold.
  - **Learned-based**: Collision operators obtained via machine learning techniques for enhanced performance and adaptability.

- **Quantum Circuit Realization**: The collision operators can be mapped to elementary quantum gates using the tools in `gate_decomposition`, enabling practical implementation on quantum hardware.

- **Amplitude Amplification**: The `amplitude_amplification` module provides gate sequences to boost the probability of the desired post-collision state in a quantum superposition. This is the key to achieving efficient multi-round QLBM simulations.

## Directory Structure
- `/src/qlbm/`: Utilities for creating different lattice configurations and symmetries.
- `/src/qlbm/operations/collision/`: Contains implementations of denoising and learned collision operators.
- `/src/qlbm/gate_decomposition/`: Tools for decomposing collision operators into quantum gates.
- `/src/qlbm/amplitude_amplification/`: Gate sequences for amplitude amplification.
- `/src/qlbm/data_generation/`: Utilities for generating sample data and state vectors for QLBM simulations.

## Getting Started

1. Clone the repository and install dependencies.
2. Explore the main modules in `/src` to understand the implementation of QLBM and collision operators.
3. Use the provided tools to generate sample data, encode states, and evaluate quantum circuit realizations.

## Applications

This project is intended for researchers and developers working on quantum algorithms for fluid dynamics and related fields. The modular design supports different lattice configurations and encoding functions.

## License

See `LICENSE` for details.