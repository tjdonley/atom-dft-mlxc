# SIMPLE-Xhole-writeup → code map

Every equation label below is the label in `../SIMPLE-Xhole-writeup/` (`main.tex`,
`appendix.tex`). Each maps to the function(s) that implement it. Where a function is
an efficiency refactor that obscures the equation, an **explicit** sibling that maps
directly to the text is provided, with a test asserting `explicit ≈ production`.

## SIMPLE features (Theory §I)

| Eq. | Meaning | Code |
|-----|---------|------|
| (projection) | window projection `c_nlm` | `descriptors/simple/bessel.py` (`RadialBesselBasis`), `pipeline.py:project_window` |
| (adaptive) | adaptive radius `g(R_ad)=ξ*` | `pipeline.py:adaptive_radius` (enclosed-charge inversion) |
| (dilation) | dilation covariance | `pipeline.py` (scale-transform path) |
| (meansplit),(transfer) | mean-split transfer | `pipeline.py:transfer_matrix`, `simple_from_window` |
| (transfer-closed) | Sturm–Liouville closed form | `pipeline.py:transfer_matrix`; `bessel.py:a_n_closed_form` |
| (dnlm) | non-dimensional `d_nlm` | `pipeline.py:simple_descriptors`; `features.py:get_simple_features` |
| invariants | power spectrum, bispectrum | `invariants.py:power_spectrum`, `bispectrum_components` |

## Gradient expansion (Theory §I E) — Eq. (sq)

| Eq. | Code |
|-----|------|
| (sq) `s` from ℓ=1 | `derivatives.py:build_spectral_gradient_operator` + `reduced_gradient_from_grad`; used by `simple_xc.py`. Validated to ~1% (`test_scale_free_gradient.py`). |
| (sq) `q` from ℓ=0 | `derivatives.py:build_spectral_laplacian_operator` (stable per-channel; matrix-free `_SpectralLaplacian`) + `reduced_laplacian_from_grad`. Recovers ∇²ρ to ~1% in the core/valence of the cached real N density. The wired folded-matrix form was numerically unstable (~300×) and was replaced. Exact adjoint `L.T` (stiff → frozen-gradient dual loop, Phase D). |

> Legacy moment/calibrated decoders have been **dropped**; `features.py` no longer
> derives s/q (they come from the spectral operators on the density grid).

## Operator projections (Theory §I F) — Eq. (coulomb)

| Eq. | Code |
|-----|------|
| (coulomb) Coulomb monopole `Σ w_n C_n` | `derivatives.py:build_window_operator` (the `P_n`); `build_coulomb_potential_operator` |

## Exchange-hole functional (Theory §I G)

| Eq. | Explicit (maps to text) | Production (speed) |
|-----|-------------------------|--------------------|
| (exact-hole),(eps-x) | `xc/simple_hole_explicit.py:hole_solve` (direct `u`-integral) | `xc/simple_hole.py:_eps_from_coeffs` (precomputed `α,β` tables, monotonic on-top inversion) |
| on-top `Q_S(ζ)=2` | `hole_solve` (`brentq`) | `_eps_from_coeffs` (`np.interp` on the monotonic `Q_S(ζ)` table — non-iterative, same machinery as `adaptive_radius`) |
| (adjoint),(adjoint-discrete) | — | `simple_hole.py:compute_xc` (operator-transpose adjoint; `gauge_fix=False` = pure adjoint) |
| (fx),(ex-gga) | `simple_hole_explicit.py:lda_exchange_per_particle` (ε_x^unif) | `simple_hole.py:SIMPLE_HOLE_GEA` — deterministic 4th-order GEA deformation: one envelope mode whose amplitude carries (10/81)s²+(146/2025)q²−(73/405)s²q, coefficients fixed by a HEG calibration of the envelope response (not fit). Preliminary; validate/tune in Phase D. |

`tests/simple/test_simple_hole.py` asserts production `SIMPLE_HOLE` ≈ explicit
`hole_solve` on the same density, and the discrete adjoint == FD of the energy.

## Numerical implementation (Theory §II) / 3D validation (App.)

| Topic | Code |
|-------|------|
| fixed-stencil convolutions; constant annihilation | `derivatives.py` spectral operators |
| 3D Cartesian pipeline | `pipeline.py:grid_descriptors`, `LatticeStencil`; `tests/simple/cartesian_validation.py` |

## Done (this phase)
- Solver glue ported (minimal): SIMPLE registration + `xc_params`/`quadrature_weights`
  plumbing in `atom/scf/driver.py`, `solver.py`, `xc/evaluator.py`,
  `xc/functional_requirements.py`. The `gga_pbe.py` refactor was NOT needed
  (`simple_xc` uses `compute_exchange_generic`, not `pbe_exchange_from_sigma`).
- Legacy decoders dropped; `features.py`, `simple_xc.py` rewired to the spectral
  operators [Eq. (sq)]; `test_scale_free_gradient.py` now tests spectral-`s`-vs-FD.
- `test_exchange_hole_operator.py` re-pointed to the explicit reference (production
  vs explicit, no SCF).

## Phase-D items (open)
1. **Frozen-gradient dual loop**: the q (ℓ=0) Laplacian adjoint that carries the
   fourth-order GEA term [Eq. (fx)] is stiff; the dual loop (freeze the
   gradient/Laplacian correction on the bare-hole potential through the outer loop)
   must be wired + benchmarked.
2. **GEA-deformed hole `SIMPLE_HOLE_GEA`** (wired, preliminary): validate the
   calibration and benchmark vs OEP/HF; tune the mode and the dual-loop SCF.
3. **`simple_xc.py` SIMPLE-PBE/SCAN** (Results §III B): 4 "reproduces standard
   functional" tests fail with the spectral gradient (the discrete-adjoint
   correctness tests pass). Re-benchmark vs PBE/r²SCAN and tune/diagnose.
