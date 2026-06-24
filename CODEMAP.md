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

| Eq. | Explicit (maps to text) | Production (speed) |
|---|---|---|
| (exact-hole),(ansatz),(eps-x),(Fx-S) | `xc/simple_hole_explicit.py:hole_solve` (direct `u`-integral) | `xc/simple_hole.py:_eps_from_coeffs` (precomputed `α,β` envelope tables, monotonic on-top inversion) |
| (QPhi) on-top `Q_S(ζ)=2` | `hole_solve` (`brentq`) | `_eps_from_coeffs` (`np.interp` on the monotonic `Q_S(ζ)` table — same machinery as `adaptive_radius`) |
| (adjoint),(adjoint-discrete) | — | `simple_hole.py:compute_xc` (operator-transpose adjoint; `gauge_fix=False` = pure adjoint) |
| (amp),(fx) second-order scale-free deformation | `simple_hole_explicit.py:hole_solve_def` (full `[g0+χφ]²`) | **`simple_hole.py:SIMPLE_HOLE_GGA`** — production: one envelope mode whose amplitude carries the exact `(10/81)s²`, one HEG-calibrated constant (not fit), LO-saturated. `SIMPLE_HOLE` is the bare hole; **`SIMPLE_HOLE_GEA`** (4th-order `q²,s²q`) is a **deprecated/experimental** variant retained for reference — it requires the unstable `q` channel and does not converge robustly. |

`tests/simple/test_simple_hole.py` asserts production ≈ explicit `hole_solve` on the
same density, and the discrete adjoint == FD of the energy.

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
- Hole revised to the second-order scale-free form (`SIMPLE_HOLE_GGA`); `SIMPLE_HOLE_GEA`
  (4th-order) retained only as a deprecated experimental variant.
- Spectral `s`/`q` operators; `test_scale_free_gradient.py` tests spectral-`s`-vs-FD;
  invariant stress-test + parameter-selection scripts added (SI `app:stress`, `app:params`).

## Open (next phase: functional design)
1. **Exchange-hole functional** (Theory I.C / Results II.B): the second-order scale-free
   hole is preliminary; validate/benchmark vs OEP/HF across the periodic table and tune
   the envelope. The richer-functional route is explicit `ℓ≥1` hole multipoles
   (App.~`app:hole`), not the deprecated `q`-based GEA4.
2. **Self-consistent benchmarks**: KS gaps; open-shell N/P vs the unrestricted
   exact-exchange reference.
