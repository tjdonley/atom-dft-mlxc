# SIMPLE writeup → code map

Every equation/section label below is the label in `writeup/` (`main.tex`,
`appendix.tex`). Each maps to the function(s) that implement it. Where a function is an
efficiency refactor that obscures the equation, an **explicit** sibling that maps
directly to the text is provided, with a test asserting `explicit ≈ production`.

The SIMPLE feature symbol in the writeup is `ϱ_{nℓm}` (`\varrho`, the non-dimensional
features); the code variables are named `descriptors`/`d`/`result[l]` for historical
reasons — same object.

**Status:** the feature-definition and invariant-construction parts (Theory I.A–B,
Results II.A) are **finalized**: production settings `R_c = 6 bohr`, `n_out = 10`,
`ℓ_max ≤ 3`; the reduced gradient `s` is carried, the reduced Laplacian `q` is excluded
(numerically unstable), and the exchange hole is the second-order scale-free form. The
open frontier is **functional design** (Theory I.C / Results II.B), flagged in-progress.

## SIMPLE feature definitions (Theory I.A)

| Eq. / concept | Code |
|---|---|
| (projection) window projection `c_{nℓm}` (I.A.1) | `descriptors/simple/bessel.py` (`RadialBesselBasis`), `pipeline.py:project_window` |
| (adaptive) adaptive radius `g(R_ad)=ξ*` / local Wigner–Seitz scale (I.A.1) | `pipeline.py:adaptive_radius`, `invert_enclosed_moment` (enclosed-charge inversion; `LatticeStencil.enclosed_moment_curve` on grids) |
| (dilation) dilation covariance (I.A.2) | `pipeline.py` (scale-transform path) |
| (meansplit),(transfer) mean-split transfer (I.A.2) | `pipeline.py:transfer_matrix`, `simple_from_window` |
| (transfer-closed) Sturm–Liouville closed form | `pipeline.py:transfer_matrix`; `bessel.py:a_n_closed_form` |
| (dnlm) non-dimensional features `ϱ_{nℓm}` (I.A.2) | `pipeline.py:simple_descriptors`, `grid_descriptors`; `features.py:get_simple_features` |

## Construction of SIMPLE invariants (Theory I.B)

| Eq. / concept | Code |
|---|---|
| (sq) reduced gradient `s` from ℓ=1 (I.B.1) — **carried** | `derivatives.py:build_spectral_gradient_operator` + `reduced_gradient_from_grad`; validated to ~1% (`tests/simple/test_scale_free_gradient.py`). |
| (sq) reduced Laplacian `q` from ℓ=0 (I.B.1) — **reconstructable but not carried** | `derivatives.py:build_spectral_laplacian_operator` (`_SpectralLaplacian`) + `reduced_laplacian_from_grad`. Numerically unstable in practice (like a discrete Laplacian: the `k_n²` eigenvalue sum amplifies high-wavevector noise); excluded from the production functional. See `app:gea`, `app:stress`. |
| power spectrum, bispectrum (I.B.2) — complete-basis invariants | `invariants.py:power_spectrum`, `bispectrum_components`, `flatten_bispectrum` |
| (coulomb) Coulomb monopole projection `Σ w_n C_n` (I.B.2) — targeted operator | `derivatives.py:build_window_operator` (the `P_n`); `build_coulomb_potential_operator` |

> Legacy moment/calibrated decoders have been **dropped**; `features.py` no longer
> derives s/q (they come from the spectral operators on the density grid).

## Exchange-hole functional (Theory I.C) — in progress

The production construction is the **kernel-mapped fixed-point hole**
`xc/simple_hole_expansion.py:SIMPLE_HOLE_KERNEL_FP`: the hole is expanded directly in the
radial SIMPLE basis (`eq:hole-direct`), the energy is the direct hole integral
`ε_x = 2π R_ad² (ϱ̃·β)` (`eq:eps-direct`), and the coefficients (scale-free shape `σ`) are
interpolated over fixed points by a kernel whose per-`ℓ` SIMPLE distances are the kernel
coordinates (`ℓ=1` ≡ `s²`). There is **no explicit gradient term and no enhancement factor**.

| Eq. | Explicit reference (maps to text) | Production (`SIMPLE_HOLE_KERNEL_FP`) |
|---|---|---|
| (exact-hole),(hole-direct),(eps-direct) | `xc/simple_hole_explicit.py:hole_solve` (direct `u`-integral); `simple_hole_expansion_explicit.py:eps_x_map` | `_kernel_eps` (direct hole integral over the basis Coulomb moments `β`) |
| LDA limit (C4) | `simple_hole_expansion_explicit.py:heg_anchor` (moment-matched HEG hole) | `_heg_mm` — anchors the kernel at the HEG node so `F_x=1` exactly at finite basis |
| Fermi–Amaldi (C5) | `map_coeffs` (`-C/Q`), `enclosed_charge_switch` | `W_FA` charge gate blends the density-following hole |
| GEA2 (C6) `F_x→1+(10/81)s²` | — | `_build_fp_nodes` fixes the `ℓ=1` node amplitude `c_G` so the slope is exact (no GEA term) |
| exact-hole references | `xc/orbital_hole.py` (exact atomic holes → kernel fixed points) | `_build_fp_nodes` adds them as further nodes; each carries only the deviation beyond LDA |
| (adjoint) | — | `compute_xc` (discrete adjoint through C/ρ/gradient channels; `gauge_fix=False` = pure adjoint) |

`tests/simple/test_simple_hole_expansion.py` (PHASE FP) asserts the uniform→LDA limit
(`F_x=1`), the GEA2 slope `10/81` from the `ℓ=1` kernel node, the discrete adjoint == FD of
the direct-integral energy, and SCF convergence (reference-free reduces to LDA+FA). The base
machinery (`SIMPLE_HOLE_EXPANSION`, `SIMPLE_HOLE`) is covered by PHASE A–D.

## Numerical implementation (Theory I.D) / 3D validation (App.)

| Topic | Code |
|---|---|
| fixed-stencil convolutions; constant annihilation | `derivatives.py` spectral operators |
| 3D Cartesian pipeline | `pipeline.py:grid_descriptors`, `LatticeStencil`; `tests/simple/cartesian_validation.py` |

## Numerical validation of feature properties (Results II.A)

| Topic (writeup) | Code / script → figure |
|---|---|
| orthogonality, HEG limit, ρ→0 stability (II.A.1) | `writeup/scripts/simple_validation_figures.py` → `simple_heg.pdf`, `simple_vacuum.pdf` |
| radial scale invariance (II.A.2) | `simple_validation_figures.py` → `simple_scale.pdf` |
| R_c / n_out by Hartree energy; cost (II.A.2; App.~`app:params`) | `writeup/scripts/parameter_selection.py` → `params_main.pdf` (main), `hartree_convergence.pdf`, `hartree_cartesian.pdf`, `scale_vs_lambda.pdf` (SI); `data/parameter_selection.json` |
| Cartesian invariance of s/P/bispectrum, q flagged (II.A.3; App.~`app:stress`) | `writeup/scripts/invariant_stress_test.py` → `invariance_summary.pdf` (main), `invariance_{grid,scale,scale_resolution,rotation,translation}.pdf` (SI); `data/invariant_stress_test.json` |
| PBE-on-SIMPLE-gradient energy check (II.A.3) | `writeup/scripts/cartesian_s_pbe.py` → `data/cartesian_s_pbe.json` |

## Done (this phase)
- Feature definitions and invariant construction **finalized** and folded into the main
  writeup (Theory I.A–B, Results II.A); `q` dropped; `R_c=6`, `n_out=10`, `ℓ_max≤3`.
- Hole construction is the kernel-mapped fixed-point form (`SIMPLE_HOLE_KERNEL_FP`):
  direct hole-integral energy, coefficients interpolated over fixed points, LDA enforced by
  anchoring the kernel and GEA2 by the `ℓ=1` node amplitude (no explicit GEA term). The old
  envelope-deformation variants (`SIMPLE_HOLE_GEA`/`GGA`) and the bolt-on-GEA kernel prototype
  were removed.
- Spectral `s`/`q` operators; `test_scale_free_gradient.py` tests spectral-`s`-vs-FD;
  invariant stress-test + parameter-selection scripts added (SI `app:stress`, `app:params`).

## Open (next phase: functional design)
1. **Exchange-hole functional** (Theory I.C / Results II.B): reference-free the functional
   reduces to LDA+FA on atoms (the kernel-GEA is a slowly-varying limit that does not reach
   atomic inhomogeneity); atom binding comes from adding exact atomic holes as kernel fixed
   points. The active next step is the full referenced self-consistent benchmark vs OEP/HF
   across the periodic table. The richer-functional route is explicit `ℓ≥1` hole multipoles.
2. **Self-consistent benchmarks**: KS gaps; open-shell N/P vs the unrestricted
   exact-exchange reference.
