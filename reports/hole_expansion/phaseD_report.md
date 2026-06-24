# Phase D — Production functional `SIMPLE_HOLE_EXPANSION` (adjoint, SCF, atoms)

**Provenance:** R_c = 8 bohr, n_channels = 24, HEG table 96 log-rho points, all-electron SCF
(domain 12 bohr). Module `atom/xc/simple_hole_expansion.py`; gates Phase-D block of
`tests/simple/test_simple_hole_expansion.py`. 29/29 gates green; existing `test_simple_hole.py`
(4/4) unaffected. Production `SIMPLE_HOLE`/`SIMPLE_HOLE_GGA` untouched.

## What was built
`SIMPLE_HOLE_EXPANSION(SIMPLE_HOLE)` — self-consistent, parameter-free. Reuses the fixed
monopole window operators `P_n`, the gauge fix, and the discrete-adjoint structure. The
energy kernel `_eps_from_coeffs(C, rho0)` runs the parameter-free map (HEG ⊕ one-electron
anchors, enclosed-charge switch, 2-constraint projection). Registered additively in
`evaluator.py`, `functional_requirements.py`, `solver.py`, and `scf/driver.py`.

**Key implementation point — the on-top density.** Reconstructing `rho0` from C via
`Σ Cm_n R_{n0}(0)` is an ill-conditioned alternating series (gave ρ0≈0, breaking the HEG
limit). Fixed by using the *actual* density `rho(r0)` for the on-top scale; `compute_xc` is
overridden to add the resulting explicit ρ-derivative term to the adjoint:
`v_x = eps + rho·(∂eps/∂rho0) + Σ_n P_n^T[ew rho ∂eps/∂C_n]/ew`.

## Verified
- **D1 (adjoint == FD):** the discrete-adjoint `v_x` matches the finite-difference `dE_x/dρ`
  to **max rel 5.5e-8** (gate 5e-6) — the coupled C-channel + explicit-ρ adjoint is exact.
- **D2 (convolutional == explicit):** the production functional reproduces the explicit HEG
  limit to <2% at ρ = 0.5/1/2.
- **D3 (SCF atoms):** all-electron SCF converges (He 13 iters, Be 24 iters); exchange energies
  at the documented **LDA level**:
  | atom | E_x (this work) | exact (oep) | LDA-level? |
  |------|-----------------|-------------|------------|
  | He   | −0.863          | −1.0258     | yes (~16% under, ≈ LDA) |
  | Be   | −2.477          | −2.6658     | yes (~7% under, ≈ LDA) |

## Honest assessment of the parameter-free map's accuracy
The functional is a *working, self-consistent, parameter-free* exchange functional with the
exact HEG and one-electron limits and an exact variational potential. Its accuracy with the
**simple HEG⊕1e blend** is at the **LDA level** (~7–16% under exact for He/Be). The diagnosis
(reports + diagnostics):
- For He (2 electrons, `Q_window≈2` everywhere) the switch selects the HEG anchor → essentially
  LDA exchange (He LDA ≈ −0.88). A pure Fermi–Amaldi anchor `−Cm/Q` instead nails He
  (−1.0269 vs −1.0258, *better* than production) — but underbinds Be (−1.87), because Be's
  dense 1s core needs *pair-localized* exchange that a single global enclosed charge cannot
  resolve. The constraint-projected HEG-anchor default is the better-**balanced** choice
  (both atoms ~LDA), so it is what ships.
- This is the textbook situation: the parameter-free direct-expansion hole reduces to LDA in
  the slowly-varying limit; reaching GGA/OEP accuracy requires the **inhomogeneity / gradient
  correction (Phase E)** and, ultimately, the feature-based learnable map (Phase F). The
  representation itself is *not* the bottleneck — Phase C showed the exact hole projects to
  0.05% (He) / 2% (Be); the map from density to hole coefficients is the frontier.

**Bottom line:** the foundation is complete and validated end-to-end — representation (A),
exact limits (B), exact-hole ground truth (C), and a self-consistent parameter-free functional
with an exact adjoint (D) at LDA accuracy. Phase E adds the gradient correction on top.

## Update — spin convention fixed; He now essentially exact
The Phase-D table above (He −0.863, Be −2.477) used the total-charge switch, which wrongly
treated spin-paired He (`Q_total≈2`) as bulk → LDA. With the **per-spin** switch (`Q_σ=Q/2`) and
the **Fermi-Amaldi** anchor (Phase-B update), the self-consistent exchange energies are:

| atom | EXPANSION (corrected) | exact | note |
|------|-----------------------|-------|------|
| He   | **−1.028**            | −1.0258 | near-exact (1 e/spin → density-following hole) |
| Be   | −2.469                | −2.6658 | LDA-level (2 e/spin: genuine two-orbital same-spin exchange) |

This confirms the framework reproduces spin-paired He exactly (the representation was never the
limit — Phase C already showed the exact He hole projects to 0.05%); the earlier miss was purely
the spin convention in the map's switch. The discrete adjoint (D1) and explicit/convolutional
agreement (D2) are unaffected and remain green.

## Update 2 — n_in / n_out and the n=10 "blow up" (clarification)
The earlier claim that n=10 "blows up" was imprecise. Diagnosis:
- On a FIXED Be density (no SCF) the map gives FINITE energies at every n (n=10: -3.38,
  n=16: -2.55, n=20: -2.42 vs exact -2.67). Truncation only degrades accuracy -- it does not
  blow up, as expected.
- The -5e14 at n=10 was an SCF SELF-CONSISTENCY instability: the under-resolved dense Be core
  produces a too-deep (but finite, max|eps|~9.4 at the nucleus) exchange potential that
  positive-feedbacks over SCF iterations. n=16 and n=20 are SCF-stable (He -1.030/-1.027,
  Be -2.640/-2.407).

Why n>=16 is needed for the HOLE here (not n_out=10): the current functional uses a SINGLE
fixed-R_c SIMPLE basis for BOTH the density projection and the hole expansion (no adaptive-
radius transfer). In that frame the dense core needs n >~ k_F R_c/pi (~16 at R_c=6). In the
PROPER SIMPLE pipeline -- project at n_in=16-20 (resolution), transfer to n_out=10 scale-free
descriptors at the ADAPTIVE radius -- n_out=10 would resolve the hole, because the adaptive
window co-scales with the local density (the dense core gets a small window). Implementing that
transfer (the scale-free frame) is what makes n_out=10 sufficient and removes the n=10 SCF
instability; it is the documented next architectural step.

Note on accuracy: the map's *converged* (well-resolved, n=20) Be is ~-2.41 (underbinding -- the
genuine 2-electron-per-spin limitation of the FA/HEG blend); n=16's -2.64 is partly a low-n
resolution artifact compensating that underbinding. He is genuinely accurate at n>=16
(-1.027 at n=20 ~ exact -1.026). Default n_channels=16.
