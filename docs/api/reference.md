# API Reference

Complete API documentation for Atom.

## Generalized Multipole Descriptors

For the usage and extension guide for the generalized descriptor framework, see [`tutorials/03_generalized_multipole_api.md`](../tutorials/03_generalized_multipole_api.md).

Public descriptor-facing classes:

- `atom.descriptors.MultipoleCalculator`
- `atom.descriptors.MultipoleResult`
- `atom.descriptors.DescriptorCalculator`
- `atom.descriptors.DescriptorContext`

Current public string options:

- `angular_basis="mcsh"`
- `radial_basis in {"heaviside", "legendre"}`

## Main Classes

### AtomicDFTSolver

The main solver class for atomic DFT calculations.

```python
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(
    atomic_number=8,
    xc_functional="GGA_PBE",
)
```

#### Parameters

- `atomic_number` (int): Atomic number of the element
- `xc_functional` (str): Exchange-correlation functional name
  - Valid options: `"LDA_PZ"`, `"LDA_SVWN"`, `"LDA_PW"`, `"GGA_PBE"`, `"SCAN"`, `"RSCAN"`, `"R2SCAN"`, `"HF"`, `"PBE0"`, `"EXX"`, `"RPA"`
- `domain_size` (float, optional): Size of the computational domain in Bohr
- `finite_element_number` (int, optional): Number of finite elements
- `polynomial_order` (int, optional): Polynomial order for basis functions
- `mesh_type` (str, optional): Type of mesh (`"exponential"`, `"polynomial"`, `"uniform"`)

#### Methods

- `solve()`: Execute the SCF calculation and return results. RPA correlation
  energy density is not implemented, so RPA requires
  `solve(save_energy_density=False)`.

#### Returns

Dictionary containing:
- `energy`: Total energy in Ha
- `energy_components`: Detailed `EnergyComponents` object
- `converged`: Whether the calculation converged
- `iterations`: Number of SCF iterations
- `rho`: Electron density array
- And more...

## Data Management

### AtomicDataManager

Class for managing atomic DFT datasets.

```python
from atom.data import AtomicDataManager

manager = AtomicDataManager(
    data_root="./dataset",
    scf_xc_functional="GGA_PBE",
    forward_pass_xc_functionals=["LDA_PZ"],
    auto_confirm=True,
)
```

See the [Data Loading tutorial](../tutorials/02_data_loading.md) for a complete workflow.

## Module Structure

- `atom.solver`: Main solver class
- `atom.scf`: SCF iteration components
- `atom.xc`: Exchange-correlation functionals
- `atom.mesh`: Finite element mesh
- `atom.data`: Data management and loading
- `atom.pseudo`: Pseudopotential handling
- `atom.utils`: Utility functions

For more detailed documentation, see the source code or individual module docstrings.
