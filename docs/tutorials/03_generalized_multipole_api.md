# Generalized Multipole API

This guide describes the current generalized multipole implementation.

The descriptor API is organized around a generalized multipole framework:

- `MultipoleCalculator(...)` is the public entry point.
- MCSH is one angular basis inside that framework, not the framework itself.
- Radial basis choice is explicit.
- Solver integration uses the generic `descriptor_calculators=[...]` hook.

If you only need to compute descriptors, start with [Quick Start](#quick-start-inline-with-the-solver). If you want to extend the framework, jump to [Adding New Implementations and Options](#adding-new-implementations-and-options).

## Mental Model

A descriptor calculation on this branch is the combination of three choices:

1. An angular basis
2. A radial basis
3. A density source

Today that means:

- angular basis: `"mcsh"`
- radial basis: `"heaviside"` or `"legendre"`
- density source:
  - the converged radial density from `AtomicDFTSolver`
  - a raw radial density array
  - a prebuilt 3D density grid

The solver always runs SCF first. Descriptor calculators are post-processing steps on the converged density. They do not change the SCF energy or density.

## What Is Supported Today

| Item | Current support | Notes |
| --- | --- | --- |
| Public calculator | `MultipoleCalculator` | Exported from `atom.descriptors` |
| Angular basis | `"mcsh"` | Current concrete basis implementation |
| MCSH orders | `l = 0, 1, 2` | In practice, use `l_max <= 2` |
| Radial basis | `"heaviside"`, `"legendre"` | Selected with `radial_basis=` |
| Radial order | integer `radial_order` | Used by Legendre kernels |
| Density input | solver result, radial arrays, 3D grids | Separate entry points |
| Solver integration | `descriptor_calculators=[...]` | Results stored in `results["descriptor_results"]` |
| Result container | `MultipoleResult` | Includes metadata and `to_npz(...)` |

Important current limits:

- `max(rcuts)` must be less than or equal to `box_size / 2`.
- `name` must be a unique, non-empty string when used with the solver.
- The radial-to-3D projection path builds a Cartesian box and supports a
  distinct uniform spacing along each axis.
- `compute_from_3d(...)` defaults to sampling a line along `x` through the selected center unless you pass `eval_indices` explicitly.

(quick-start-inline-with-the-solver)=
## Quick Start: Inline With the Solver

The most common workflow is to attach a multipole calculator directly to the solver.

```python
from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator

calc = MultipoleCalculator(
    angular_basis="mcsh",
    radial_basis="heaviside",
    rcuts=[0.5, 1.0, 1.5, 2.0, 3.0],
    l_max=2,
    box_size=20.0,
    spacing=0.3,
)

solver = AtomicDFTSolver(
    atomic_number=1,
    xc_functional="GGA_PBE",
    descriptor_calculators=[calc],
    verbose=False,
)

results = solver.solve()
mp = results["descriptor_results"]["multipole"]

print(results["energy"])
print(mp.descriptors.shape)
print(mp.angular_basis, mp.radial_basis)
```

What comes back:

- `results["descriptor_results"]` is always present.
- If no calculators are attached, it is an empty dict.
- Each calculator stores its result under its `name`.
- The default name for `MultipoleCalculator` is `"multipole"`.

## Understanding `MultipoleResult`

`MultipoleCalculator` returns a `MultipoleResult` with:

- `grid_indices`: integer grid indices used for evaluation
- `grid_positions`: physical coordinates in Bohr
- `descriptors`: descriptor tensor with shape `(n_eval_points, n_rcuts, l_max + 1)`
- `rcuts`: list of cutoff radii in Bohr
- `l_max`: maximum angular order
- `spacing`: grid spacing in Bohr
- `angular_basis`: basis name used to compute the result
- `radial_basis`: radial basis name used to compute the result
- `radial_order`: radial order used for polynomial kernels
- `center`: center used for default sampling and radial profiling

You can save a result for later inspection:

```python
mp.to_npz("hydrogen_multipole.npz")
```

## Post-Hoc From a Solver Result

You do not need to decide all descriptor settings up front. You can run SCF once, then compute descriptors afterward.

```python
from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator

solver = AtomicDFTSolver(atomic_number=1, xc_functional="LDA_PZ", verbose=False)
results = solver.solve()

calc = MultipoleCalculator(
    angular_basis="mcsh",
    radial_basis="legendre",
    radial_order=2,
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
)

mp = calc.compute_from_solver_result(results)
print(mp.descriptors.shape)
```

This is useful when:

- you want to reuse an existing converged density
- you want to compare multiple radial bases on the same density
- you want to sweep over `rcuts`, `l_max`, or `radial_order` without rerunning SCF

## Direct Usage From Radial Density

You can also call the calculator directly on radial arrays.

```python
from atom.descriptors import MultipoleCalculator

r = results["quadrature_nodes"]
rho = results["rho"]

calc = MultipoleCalculator(
    angular_basis="mcsh",
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
)

mp = calc.compute_from_radial(r, rho)
print(mp.descriptors.shape)
```

This path:

- projects the 1D radial density onto a 3D Cartesian grid
- centers the atom at `(box_size / 2, box_size / 2, box_size / 2)`
- computes descriptors on the projected grid

Use this when your data already lives on the solver quadrature grid or any compatible radial grid.

## Direct Usage From a 3D Density Grid

If you already have a 3D density, use `compute_from_3d(...)`.

```python
import numpy as np
from atom.descriptors import MultipoleCalculator

n = 21
h = 1.0
center = (10.0, 3.0, 7.0)
coords = np.arange(n) * h
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

rho_3d = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2))

calc = MultipoleCalculator(
    angular_basis="mcsh",
    rcuts=[2.0],
    l_max=1,
    box_size=20.0,
    spacing=h,
    periodic=False,
)

mp = calc.compute_from_3d(
    rho_3d,
    spacing=(h, h, h),
    center=center,
)
```

Notes for 3D usage:

- `spacing` can be a single float or a 3-tuple.
- If `center` is omitted, the geometric grid midpoint is used.
- If `eval_indices` is omitted, the calculator evaluates a line along `x` that passes through the chosen center.
- If you want a custom line, plane, or arbitrary set of points, pass `eval_indices` explicitly.

## Working With Radial Profiles

`extract_radial_profile(...)` converts the sampled grid points into distances from a center.

```python
profile = calc.extract_radial_profile(mp)

print(profile.keys())
print(profile["r"].shape)
print(profile["descriptors"].shape)
```

Returned keys:

- `r`
- `descriptors`
- `rcuts`
- `l_max`

For plotting, it is usually best to sort by `r` first:

```python
order = np.argsort(profile["r"])
r_sorted = profile["r"][order]
d_sorted = profile["descriptors"][order]
```

If your result came from an externally defined 3D grid, you can override the center used for profiling:

```python
profile = calc.extract_radial_profile(mp, center=center)
```

## Comparing Radial Bases

This branch makes the radial basis explicit.

### Heaviside

The Heaviside basis integrates density inside each cutoff sphere.

```python
heaviside = MultipoleCalculator(
    angular_basis="mcsh",
    radial_basis="heaviside",
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
)
```

### Legendre

The Legendre basis reweights density inside the cutoff sphere.

```python
legendre2 = MultipoleCalculator(
    angular_basis="mcsh",
    radial_basis="legendre",
    radial_order=2,
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
)
```

Useful current fact:

- `radial_order=0` for the Legendre basis matches the Heaviside basis in the current implementation.

So a quick sanity check looks like this:

```python
h_result = heaviside.compute_from_solver_result(results)

legendre0 = MultipoleCalculator(
    angular_basis="mcsh",
    radial_basis="legendre",
    radial_order=0,
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
)

l0_result = legendre0.compute_from_solver_result(results)
```

## Using Multiple Calculators in One Solver Run

You can attach multiple calculators, as long as their names are unique.

```python
from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator

hsmp = MultipoleCalculator(
    angular_basis="mcsh",
    radial_basis="heaviside",
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
    name="hsmp",
)

lp2 = MultipoleCalculator(
    angular_basis="mcsh",
    radial_basis="legendre",
    radial_order=2,
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
    name="lp2",
)

results = AtomicDFTSolver(
    atomic_number=1,
    xc_functional="LDA_PZ",
    descriptor_calculators=[hsmp, lp2],
    verbose=False,
).solve()

print(results["descriptor_results"].keys())
```

If two calculators use the same name, solver initialization raises an error.

## Common Pitfalls and Current Limits

### Units

- `rcuts`, `box_size`, `spacing`, and `center` are all in Bohr.

### `l_max`

- The public API accepts any non-negative integer.
- The current MCSH implementation only defines `l = 0, 1, 2`.
- Asking for higher orders with `angular_basis="mcsh"` will fail until those orders are implemented.

### Box size and cutoffs

- The calculator validates that the largest cutoff fits inside the box.
- If `max(rcuts) > box_size / 2`, construction fails immediately.

### Default 3D sampling

- `compute_from_3d(...)` does not evaluate every grid point by default.
- Instead, it evaluates a line along `x` through the selected center.
- Pass `eval_indices` when you need a different sampling geometry.

### Periodic vs non-periodic evaluation

- `periodic=True` wraps stencil indices around the box boundaries.
- `periodic=False` zero-pads outside the box.
- For isolated, finite 3D grids, `periodic=False` is often the clearer choice.

### Radial projection is still cubic

- `compute_from_radial(...)` projects onto a Cartesian 3D grid with optional
  anisotropic spacing.
- The direct 3D path supports a 3-tuple spacing input.
- The radial projection path is still effectively isotropic.

## Validation and What Is Tested

This branch already has test coverage for the generalized multipole surface and the MCSH specialization.

Key test areas:

- generic solver contract for any `DescriptorCalculator`
- solver wiring and name validation
- parity between inline solver results and post-hoc computation
- parity between calculator methods and low-level descriptor engine calls
- direct 3D-center semantics
- physical invariants for spherical densities
- master-parity regression tests for the MCSH case

Relevant files in this repository:

- `tests/test_descriptor_solver_generic_contract.py`
- `tests/test_mcsh_descriptors.py`
- `tests/test_mcsh_solver_wiring.py`
- `tests/test_mcsh_solver_integration.py`
- `tests/test_mcsh_e2e_hydrogen.py`
- `tests/test_multipole_master_parity.py`

(adding-new-implementations-and-options)=
## Adding New Implementations and Options

This section is the bridge between the current code and the theory in the thesis.

The thesis frames multipole descriptors as a descriptor family that should be:

- complete within the chosen basis and cutoff range
- symmetry-aware, especially for translation and 3D rotation
- systematically improvable
- computationally scalable

For radial probe functions, it also highlights three useful design goals:

- orthogonality
- convergibility
- efficiency

That is a good checklist for every new option you add here.

### 1. Add a New Angular Basis

Angular bases live behind the `AngularBasis` contract in `atom/descriptors/basis.py`.

A new basis must define:

- `name`
- `component_specs(l)`
- `evaluate_component(dx, dy, dz, l, label)`
- `combine_invariant(l, components)`

Minimal skeleton:

```python
from atom.descriptors.basis import AngularBasis


class MyAngularBasis(AngularBasis):
    name = "my_basis"

    def component_specs(self, l):
        ...

    def evaluate_component(self, dx, dy, dz, l, label):
        ...

    def combine_invariant(self, l, components):
        ...
```

Then register it in `resolve_angular_basis(...)` in `atom/descriptors/basis.py`.

What the thesis suggests you think about here:

- Does the basis improve completeness or compactness?
- Is it orthonormal or over-complete?
- How will you construct rotational invariants without throwing away too much information?

Concrete thesis-motivated directions:

- real spherical harmonics as a more standard complete angular basis
- alternative invariant construction beyond the current Euclidean norm reduction
- Clebsch-Gordan style coupling if you want richer invariant structure

### 2. Extend MCSH to Higher Orders

The current MCSH implementation is in `atom/descriptors/mcsh.py`.

To extend it:

1. Add the component labels and weights for the new `l` to `_components`.
2. Extend `evaluate_component(...)` with the new polynomial forms.
3. Confirm that `combine_invariant(...)` still uses the right normalization.
4. Add low-level and solver-level tests before exposing the higher order in examples.

Why this matters:

- the thesis notes that MCSH is currently over-complete
- higher orders increase expressiveness, but they also increase implementation and validation burden
- if you want higher-order MCSH to stay scientifically meaningful, test parity and invariance carefully

### 3. Add a New Radial Basis or Kernel

Radial kernels live in `atom/descriptors/kernels.py`.

The public builder is `build_radial_kernel(...)`. A kernel must provide:

- `evaluate(q)`
- `as_dict()`

Minimal skeleton:

```python
class MyKernel:
    name = "my_kernel"

    def evaluate(self, q):
        ...

    def as_dict(self):
        return {"name": self.name}
```

Then register it in `build_radial_kernel(...)`.

Current code already includes `GaussianKernel` and `CosineCutoffKernel`, but they are not exposed through the public string dispatch yet.

What the thesis suggests you think about here:

- Is the basis orthogonal or strongly correlated?
- Does it converge efficiently as you add more radial functions?
- Is its length scale physically interpretable?
- Does it preserve good numerical behavior inside a compact support?

Concrete thesis-motivated directions:

- more orthogonal radial probe families
- polynomial bases beyond the current Legendre path
- Fourier-like radial probes
- scale-aware or scale-invariant radial constructions

### 4. Add a Completely New Descriptor Calculator

If you want something beyond the multipole family, use the generic solver contract.

Implement `DescriptorCalculator` from `atom/descriptors/base.py`:

```python
from atom.descriptors import DescriptorCalculator, DescriptorContext


class MyDescriptorCalculator(DescriptorCalculator):
    @property
    def name(self) -> str:
        return "my_descriptor"

    def compute(self, context: DescriptorContext):
        ...
```

Then attach it with:

```python
from atom import AtomicDFTSolver


solver = AtomicDFTSolver(
    atomic_number=1,
    xc_functional="LDA_PZ",
    descriptor_calculators=[MyDescriptorCalculator()],
)
```

Current `DescriptorContext` exposes only:

- `quadrature_nodes`
- `density`

If your descriptor needs more solver outputs, extend `DescriptorContext` and the solver packaging path deliberately rather than reaching into solver internals from the calculator.

### 5. Testing Checklist for New Options

For any new basis, kernel, or calculator, a good minimum bar is:

1. Input validation tests
2. Low-level engine parity tests
3. Solver wiring tests
4. Result-shape and metadata tests
5. Symmetry or invariance tests
6. Regression tests against a trusted reference if you are reproducing an existing method

This matches the overall theme of the branch: generalize the API without giving up the scientific validation story.

### 6. Future Directions Suggested by the Thesis

The thesis points to several natural next steps for this generalized framework:

- replace over-complete MCSH-style angular probes with more compact complete bases where useful
- introduce orthogonal radial probe families to reduce correlation between features
- reduce information loss from simple norm-based invariant compression
- construct scale-invariant descriptor variants for functional-design work
- build ML or analytical xc models on top of these descriptors
- use the existing variational-derivative framework to support self-consistent functionals driven by multipole features

Those are future directions, not features already exposed by this branch. The current code gives you the extension seams needed to start that work without keeping MCSH as the only conceptual story.
