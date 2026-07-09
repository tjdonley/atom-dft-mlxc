# Basic Solver Usage

This tutorial demonstrates the basic usage of the `AtomicDFTSolver` to solve simple atomic systems.

## Overview

The `AtomicDFTSolver` is the main class for performing atomic DFT calculations. It supports various exchange-correlation functionals and can work with both all-electron and pseudopotential calculations.

## Creating a Solver

Let's start by creating a solver for a Hydrogen atom:

```python
from atom import AtomicDFTSolver

# Create a solver instance
solver = AtomicDFTSolver(
    atomic_number=1,        # Hydrogen
    xc_functional="LDA_PW",  # LDA Perdew-Wang functional
    domain_size=20.0,        # Domain size in Bohr
    finite_element_number=15,  # Number of finite elements
    polynomial_order=20,     # Polynomial order
)

print(f"Solver created for atomic number: {solver.atomic_number}")
print(f"XC functional: {solver.xc_functional}")
```

## Running a Calculation

Now let's solve the system:

```python
# Execute the SCF calculation
result = solver.solve()

# Check results
print(f"Total energy: {result['energy']:.6f} Ha")
print(f"Converged: {result['converged']}")
print(f"Number of SCF iterations: {result['iterations']}")
```

## Accessing Results

The `solve()` method returns a dictionary. Detailed energy terms are available
through its `energy_components` value:

```python
# Access different result properties
print("Result properties:")
components = result["energy_components"]
print(f"  Total energy: {result['energy']:.6f} Ha")
print(f"  Kinetic energy: {components.total_kinetic:.6f} Ha")
print(f"  Potential energy: {components.total_potential:.6f} Ha")
print(f"  Exchange energy: {components.exchange:.6f} Ha")
print(f"  Correlation energy: {components.correlation:.6f} Ha")
```

## Visualizing Results

Let's visualize the electron density:

```python
import matplotlib.pyplot as plt
import numpy as np

# Get radial grid and density
grid = result["quadrature_nodes"]
density = result["rho"]

# Plot electron density
plt.figure(figsize=(8, 6))
plt.plot(grid, density, 'b-', linewidth=2, label='Electron density')
plt.xlabel('Radius (Bohr)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Hydrogen Atom - Electron Density', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Different XC Functionals

Atom supports various exchange-correlation functionals. Let's try a different one:

```python
# Solve with GGA_PBE functional
solver_gga = AtomicDFTSolver(
    atomic_number=1,
    xc_functional="GGA_PBE",
    domain_size=20.0,
    finite_element_number=15,
    polynomial_order=20,
)

result_gga = solver_gga.solve()
print(f"LDA_PW energy: {result['energy']:.6f} Ha")
print(f"GGA_PBE energy: {result_gga['energy']:.6f} Ha")
print(f"Energy difference: {result_gga['energy'] - result['energy']:.6f} Ha")
```

## Multiple Atoms

You can easily solve different atoms:

```python
# Solve Helium atom
solver_he = AtomicDFTSolver(
    atomic_number=2,
    xc_functional="LDA_PW",
    domain_size=20.0,
    finite_element_number=15,
    polynomial_order=20,
)

result_he = solver_he.solve()
print(f"Helium total energy: {result_he['energy']:.6f} Ha")
print(f"Converged: {result_he['converged']}")
```

## Next Steps

- Learn about [Loading Data](02_data_loading.md) for working with datasets
- Check out the [Cookbook](../cookbook.md) for more examples
- See the [API Reference](../api/reference.md) for detailed documentation
