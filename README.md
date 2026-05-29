<div align="center">
<img src="logo.png" alt="logo" width="250"></img>
</div>

> **Note:** This project is still in development.

# ATOM — Atomic DFT with finite elements

<!-- After making the repo public, use the dynamic badge: [![CI](https://github.com/tjdonley/atom-dft-mlxc/actions/workflows/ci.yaml/badge.svg)](https://github.com/tjdonley/atom-dft-mlxc/actions/workflows/ci.yaml) -->

[**Features**](#features)
| [**MCSH descriptors**](#mcsh-descriptors)
| [**Quick start**](#quick-start)
| [**Installation**](#installation)
| [**Change log**](ChangeLog)
| [**Documentation**](docs/)


## What is ATOM?

**ATOM** is a Python library for atomic (spherical) density functional theory (DFT) with a finite-element discretization in real space. It is heavily optimized and aims for high numerical accuracy.

ATOM solves the Kohn–Sham equations self-consistently and supports a wide range of exchange–correlation functionals. Calculations can be run in all-electron mode or by reading norm-conserving pseudopotential (PSP) files.

Advanced capabilities include the optimized effective potential (OEP) method, hybrid functionals with exact exchange (e.g. HF), RPA with parallelization, configurable parameters and advanced options, and more.

This is a research code. Please try it out, [report issues](https://github.com/tjdonley/atom-dft-mlxc/issues), and share feedback.

```python
from atom import AtomicDFTSolver

# Single-atom DFT with GGA-PBE
solver = AtomicDFTSolver(atomic_number=13, xc_functional="GGA_PBE")
results = solver.solve()

# Access total energy, density, eigenvalues, etc.
print(results["energy"])
```

### Contents

- [ATOM — Atomic DFT with finite elements](#atom--atomic-dft-with-finite-elements)
  - [What is ATOM?](#what-is-atom)
    - [Contents](#contents)
  - [Features](#features)
  - [MCSH descriptors](#mcsh-descriptors)
  - [Quick start](#quick-start)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Instructions](#instructions)
  - [Project structure](#project-structure)
  - [Optional dependencies](#optional-dependencies)
  - [Citing ATOM](#citing-atom)
  - [Reference documentation](#reference-documentation)


## Features

* **Finite-element discretization** — Real-space mesh and operators in `atom.mesh`.
* **Pseudopotentials** — Norm-conserving pseudopotential support (e.g. psp8) in `atom.pseudo`.
* **SCF driver** — Density, Hamiltonian, eigensolver, Poisson, mixing, and convergence in `atom.scf`.
* **Exchange–correlation** — LDA, GGA-PBE, hybrid (HF), and ML-XC in `atom.xc`.
* **Data and ML** — Dataset generation, loading, and ML-XC training interfaces in `atom.data` and `atom.xc.ml_xc`.
* **MCSH descriptors** — Maxwell Cartesian Spherical Harmonic multipole descriptors with Heaviside and Legendre polynomial radial kernels in `atom.descriptors`.


## Quick start

```python
from atom import AtomicDFTSolver

# xc_functional can be any supported functional (e.g. GGA_PBE, LDA_PZ, PBE0, ...)
solver = AtomicDFTSolver(atomic_number=29, xc_functional="GGA_PBE")
results = solver.solve()

# Many options available: domain_size, mesh, grid, SCF settings, verbose, etc.
solver = AtomicDFTSolver(
    atomic_number=6,
    xc_functional="LDA_PZ",
    domain_size=15.0,
    verbose=True,
)
results = solver.solve()
```


## MCSH descriptors

ATOM can compute multipole descriptors from the self-consistent electron density. The current implementation supports the MCSH (Maxwell Cartesian Spherical Harmonic) angular basis as one concrete choice within that broader framework.

For the full branch-specific usage and extension guide, see [`docs/tutorials/03_generalized_multipole_api.md`](docs/tutorials/03_generalized_multipole_api.md).

### Basic usage

Pass descriptor calculators to the solver to compute descriptors inline with the SCF calculation:

```python
from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator

calc = MultipoleCalculator(
    angular_basis="mcsh",                     # current angular basis option
    radial_basis="heaviside",                # or "legendre"
    rcuts=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    l_max=2,
    box_size=16.0,
    spacing=0.4,
)

solver = AtomicDFTSolver(
    atomic_number=6,
    xc_functional="GGA_PBE",
    descriptor_calculators=[calc],
)
results = solver.solve()

# Descriptors are in the result dict
mp = results["descriptor_results"]["multipole"]
print(mp.descriptors.shape)  # (n_eval_points, n_rcuts, n_l)
```

### Post-hoc computation

You can also compute descriptors after the fact from a saved density:

```python
from atom.descriptors import MultipoleCalculator

calc = MultipoleCalculator(
    angular_basis="mcsh",
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
)

# From solver results
mp = calc.compute_from_solver_result(results)

# Or from raw radial arrays
mp = calc.compute_from_radial(r_quadrature, rho)

# Or from a pre-built 3D density grid
mp = calc.compute_from_3d(rho_3d, spacing=(h, h, h))

# Extract radial profile (distance from atom center)
profile = calc.extract_radial_profile(mp)
```

### Legendre polynomial kernels

By default, descriptors use the Heaviside radial basis. You can also use Legendre polynomial kernels, which weight the density differently within the cutoff sphere:

```python
calc_lp2 = MultipoleCalculator(
    angular_basis="mcsh",
    rcuts=[1.0, 2.0, 3.0, 4.0],
    l_max=2,
    radial_basis="legendre",  # "heaviside" (default) or "legendre"
    radial_order=2,
)
```

Legendre kernels provide additional information about the radial distribution of charge within the cutoff sphere. Order 0 is identical to Heaviside.

### Validation

End-to-end validation results for H, He, Li, Be, C, N, O are in [`docs/validation/`](docs/validation/), including a PDF report and figures demonstrating charge sum rule convergence, dipole vanishing, and kernel comparisons.


## Installation

### Requirements

* Python ≥ 3.8
* NumPy ≥ 1.20
* SciPy ≥ 1.7

### Instructions

| Use case        | Command |
|-----------------|---------|
| Core (CPU)      | `pip install -e .` |
| With ML-XC      | `pip install -e ".[ml]"` |
| With viz        | `pip install -e ".[viz]"` |
| Dev + tests     | `pip install -e ".[dev]"` |
| All optional    | `pip install -e ".[all]"` |

From the repository root:

```bash
pip install -e .
```


## Project structure

| Directory / module | Description |
|--------------------|-------------|
| `atom/mesh`        | Grid construction and operators |
| `atom/pseudo`      | Pseudopotential reading and evaluation (local / non-local) |
| `atom/scf`         | SCF loop: density, Hamiltonian, eigensolver, Poisson, mixer |
| `atom/xc`          | XC functionals: LDA, GGA, HF, ML-XC, etc. |
| `atom/data`        | Data generation, loading, and processing |
| `atom/descriptors`  | MCSH multipole descriptors (Heaviside and Legendre kernels) |
| `atom/utils`       | Occupation states, periodicity helpers |
| `tests`            | Unit and integration tests |
| `docs`             | Tutorial and documentation source |


## Optional dependencies

| Extra   | Purpose |
|---------|---------|
| `ml`    | PyTorch, scikit-learn for ML-XC |
| `viz`   | Matplotlib for plotting |
| `dev`   | pytest, Jupyter for development |
| `threadpool` | threadpoolctl for RPA/thread control |
| `docs`  | Jupyter Book, Sphinx for building docs |


## Citing ATOM

If you use this code in your research, please cite the repository:

```bibtex
@software{atom2026,
  author = {Qihao Cheng and Shubhang Trivedi and Phanish Suryanarayana},
  title = {{ATOM}: Atomic DFT with finite elements and ML-XC density descriptors},
  url = {https://github.com/tjdonley/atom-dft-mlxc},
  version = {0.1.0},
  year = {2026},
}
```


## Reference documentation

For API details and tutorials, see the [documentation](docs/) in this repository.

For development and contribution guidelines, see the [repository](https://github.com/tjdonley/atom-dft-mlxc).
