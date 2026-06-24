# Phase C — Exact orbital-based exchange-hole reference (ground truth)

**Provenance:** all-electron EXX SCF (domain 14 bohr), R_c = 6, n_channels = 16, angular
quadrature n_u = 160, n_mu = 80. Module `atom/xc/orbital_hole.py`; generator
`reports/hole_expansion/gen_orbital_hole_refs.py` → `tests/simple/data/orbital_hole_{He,Be}.npz`;
gates `tests/simple/test_simple_hole_expansion.py` (Phase C block). 25/25 gates green.

## What was built
`orbital_hole.py` reconstructs the **exact** spherically-averaged exchange hole from occupied
KS/HF orbitals (the honest ground truth for testing/fitting the map):
- `extract_s_orbitals` — radial orbitals g_i = φ_i/r from the solver (verified
  `ρ = Σ_i occ_i g_i²/4π` to 1e-15).
- `exchange_hole_s(r0, u)` — `n_x(r0,u) = -(2/ρ) ⟨|ρ₁_σ(r0,r0+u)|²⟩_Ω` via the per-spin
  1-RDM and a Gauss–Legendre angular average over û (restricted convention, matching `hf.py`).
- `exact_eps_x`, `exact_Ex` — `eps_x = ½∫n_x/u d³u` and the integrated energy.
- `project_exact_hole` — the exact hole's SIMPLE monopole coefficients `ϱ̃^exact_{n00}(r0)`
  (target for the learnable map).
- `spherical_avg_hydrogenic_1s` / `spherical_avg_radial` — closed-form vs numerical angular
  average (machinery validation).

**Scope:** s-only occupied manifolds (H, He, Be). l>0 (Ne 2p⁶) needs the spherical-harmonic
addition theorem (the Legendre/Wigner machinery already in `hf.py`); raised as a clear
`NotImplementedError` and documented as an extension.

## Verified
- **C1 (headline):** the integrated orbital-hole exchange energy matches the solver's exact
  `oep_exchange` to **<1 mHa**:
  - He: E_x(hole) = −1.025774 vs −1.025769 (0.005 mHa)
  - Be: E_x(hole) = −2.665775 vs −2.665778 (0.003 mHa)
  This validates the orbital convention, the angular-averaged 1-RDM hole construction, and
  the `eps_x = ½∫n_x/u` energy formula on real atoms.
- **C1 (reproducibility):** `eps_x(r0)` recomputed from saved orbitals matches stored values to 1e-3.
- **C1b (representability):** projecting the exact hole onto the SIMPLE basis and re-evaluating
  the exchange **energy** reproduces it to 0.05% (He) / 2.1% (Be) on the reference r0 grid.
- **C2:** the numerical angular average matches the closed-form hydrogenic-1s `⟨ρ(r0+u)⟩_Ω`
  to <0.2% (Z = 1, 2).

## Finding: where the windowed direct expansion is and isn't faithful
Per-point `eps_x(r0)` from the projected hole is excellent in the energy-dominant core/valence
region but degrades in two regimes, both physically understood:
1. **Diffuse tail (large r0):** the exact exchange hole is intrinsically *long-ranged* there —
   it sits back in the bulk, far from r0 — so a window localized around r0 cannot represent it
   (30–55% per-point error at r0 ≈ 5–6 for Be). But ρ is negligible there, so the **energy is
   unaffected** (C1 still <1 mHa).
2. **High-density core (small r0):** the hole is narrow (~1/k_F), needing `N ≳ k_F R_c/π`
   channels; at fixed R_c=6 the Be 1s core is under-resolved at N=16 (core error 15%→1.8% as
   N: 16→40).

Both are the **fixed-R_c** limitations already identified in Phase A; the **scale-free
production frame** (Phase D, projecting at fixed ξ* via the adaptive radius) makes resolution
density-independent and is the proper fix. The honest takeaway: the exact-hole reference is
validated (energy to <1 mHa), and the direct expansion represents the *energy* faithfully
(He 0.05%, Be 2%), with documented degradation of the *pointwise* hole in the diffuse tail.
