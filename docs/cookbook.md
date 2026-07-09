# Cookbook

A collection of common tasks and examples for using Atom.

## Basic Tasks

### Calculate Energy for a Single Atom

```python
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(
    atomic_number=13,  # Aluminum
    xc_functional="GGA_PBE",
)

result = solver.solve()
print(f"Total energy: {result['energy']:.6f} Ha")
```

### Compare Different XC Functionals

```python
from atom import AtomicDFTSolver

functionals = ["LDA_PW", "GGA_PBE", "SCAN"]
energies = {}

for func in functionals:
    solver = AtomicDFTSolver(
        atomic_number=1,  # Hydrogen
        xc_functional=func,
    )
    result = solver.solve()
    energies[func] = result["energy"]
    print(f"{func}: {result['energy']:.6f} Ha")
```

### Extract Electron Density

```python
from atom import AtomicDFTSolver
import numpy as np

solver = AtomicDFTSolver(atomic_number=1, xc_functional="LDA_PW")
result = solver.solve()

# Get radial grid and density
r = result["quadrature_nodes"]
rho = result["rho"]

print(f"Grid points: {len(r)}")
print(f"Density range: [{rho.min():.6e}, {rho.max():.6e}]")
```

## Visualization

### Plot Electron Density

```python
import matplotlib.pyplot as plt
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(atomic_number=1, xc_functional="LDA_PW")
result = solver.solve()

r = result["quadrature_nodes"]
rho = result["rho"]

plt.figure(figsize=(8, 6))
plt.plot(r, rho, 'b-', linewidth=2)
plt.xlabel('Radius (Bohr)')
plt.ylabel('Electron Density')
plt.title('Hydrogen Atom - Electron Density')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Advanced Tasks

### Custom Mesh Parameters

```python
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(
    atomic_number=1,
    xc_functional="LDA_PW",
    domain_size=30.0,      # Larger domain
    finite_element_number=20,  # More elements
    polynomial_order=25,  # Higher order
    mesh_type="polynomial",
    mesh_concentration=2.0,
)

result = solver.solve()
print(f"Energy with custom mesh: {result['energy']:.6f} Ha")
```

### Check Convergence

```python
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(
    atomic_number=1,
    xc_functional="LDA_PW",
    scf_tolerance=1e-10,  # Stricter tolerance
)

result = solver.solve()
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Final error: {result['rho_residual']:.2e}")
```

## Tips and Best Practices

1. **Start with simple examples**: Begin with Hydrogen (atomic_number=1) to test your setup
2. **Check convergence**: Always verify that `result["converged"]` is `True`
3. **Adjust mesh parameters**: For heavier atoms, you may need a larger `domain_size` or more `finite_element_number`
4. **Monitor iterations**: If `result["iterations"]` is very high, consider adjusting SCF parameters
