# Atom - Atomic DFT Solver

Welcome to the Atom documentation!

Atom is a comprehensive implementation of Atomic Density Functional Theory (DFT) solver using finite element method for solving the Kohn-Sham equations for atomic systems.

## Features

- Multiple exchange-correlation functionals (LDA, GGA, Meta-GGA, Hybrid, RPA, etc.)
- All-electron and pseudopotential calculations
- Self-consistent field (SCF) iterations with convergence control
- High-order finite element discretization with Legendre-Gauss-Lobatto nodes
- Generalized multipole descriptor post-processing with an explicit angular/radial basis split
- Machine learning exchange-correlation potentials
- Data generation and management tools

## Quick Start

Here's a simple example to get you started:

```{code-cell} python
from atom import AtomicDFTSolver

# Solve Hydrogen atom with LDA
solver = AtomicDFTSolver(
    atomic_number=1,
    xc_functional="LDA_PW",
    domain_size=20.0,
    finite_element_number=15,
    polynomial_order=20,
)

result = solver.solve()
print(f"Total energy: {result['energy']:.6f} Ha")
print(f"Converged: {result['converged']}")
print(f"Number of iterations: {result['iterations']}")
```

## Documentation Structure

- **[Installation Guide](installation.md)**: How to install and set up Atom
- **[Basic Solver Usage](tutorials/01_basic_solver.md)**: Learn the basics of using the solver
- **[Generalized Multipole API](tutorials/03_generalized_multipole_api.md)**: Branch-specific guide for multipole descriptor usage and extension
- **[Loading Data](tutorials/02_data_loading.md)**: How to load and work with datasets
- **[Cookbook](cookbook.md)**: Common tasks and examples
- **[API Reference](api/reference.md)**: Complete API documentation

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/tjdonley/atom-dft-mlxc/issues)
- **Repository**: [View source code](https://github.com/tjdonley/atom-dft-mlxc)

## Authors

- Qihao Cheng <qcheng61@gatech.edu>
- Shubhang Trivedi <strivedi44@gatech.edu>
- Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

Material Physics & Mechanics Group, Georgia Tech
