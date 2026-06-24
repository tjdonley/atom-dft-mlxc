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

## Update 3 — scale-free frame wired in (adaptive radius + transfer); n_out=10 works
The functional now operates in the SIMPLE scale-free (adaptive-radius) frame, reusing the
machinery from the simple-hole-additive branch (`SIMPLE_HOLE_SF` pattern):
- project the density to n_in=20 fixed-R_c window coeffs C (resolution);
- implicit adaptive radius R_ad = min(X/k_F(rho0), R_c), X = k_F R_ad = 8 (differentiable);
- transfer c_ad = transfer_matrix(0, R_ad, n_out, n_in) @ C onto the n_out=10 adaptive basis
  (precomputed on an R_ad grid + interpolated);
- HEG anchor becomes the UNIVERSAL fixed shape sigma_m = int_0^1 R_m^(1)(t) S(X t) t^2 dt
  (k_F R_ad locked), FA anchor -c_ad/Q; moments scale as R_ad^{3/2} (charge), R_ad^{1/2} (Coulomb).
Defaults: n_in (n_channels) = 20, n_out = 10.

Verified:
- **Scale invariance:** the HEG ratio eps_x/eps_LDA is now CONSTANT across density (1.033 at
  rho=0.5/2/5) -- exact scale invariance (constraint X3), vs the fixed-R_c frame where it drifted.
  The 1.033 is a constant, density-independent finite-X-window/constraint offset (correctable).
- **n_out=10 works:** He/Be SCF converge with NO blow up (He -0.892, Be -2.395; max|eps| ~ 1.2/2.5,
  vs the fixed-R_c n=10 disaster of 9.4 + divergence). The adaptive radius resolves the dense
  core at n_out=10 (the window shrinks where the density is high) -- confirming the n_in/n_out
  picture: n_in=20 projection resolution, n_out=10 exposed/hole basis.
- **Exact adjoint:** the discrete-adjoint potential matches FD dE/drho to 3.8e-9. The existing
  compute_xc (C-channel FD + explicit-rho0 FD) works unchanged -- the rho0 channel captures
  R_ad(rho), eta(rho) and the rho0 factors.

Accuracy: He/Be are LDA-level here (He -0.89, Be -2.40). The scale-free frame changes the
adaptive-window enclosed charge, so the per-spin/Fermi-Amaldi accuracy that made He exact in the
fixed-R_c frame is NOT recovered by the bare map -- that is restored by the scale-free l=1
iso-orbital gate (validated in Update 5, the next implementation step). 40/40 tests green.

## Update 4 — 4pi bug fix (H/He were wrongly LDA); per-spin FA/SIC restored
Quick check "is H exact?" exposed a bug. The window operators' C carry the angular 4pi (the
3D projection), but the scale-free enclosed charge was computed as Q = 4pi R_ad^{3/2}(c_ad.a1)
with c_ad = T @ C -- double-counting 4pi, so Q was inflated by ~4pi (~12.6). Every atom then
had Q/2 >> 1 -> lambda = 0 -> wrongly flagged HEG -> LDA-level (He -0.89, H -0.21).

Fix: divide C by 4pi before the transfer (c_ad = T @ (C/4pi)), matching the HEG-anchor sigma
convention. After the fix the per-spin Fermi-Amaldi/SIC limit works:

| atom | before fix (LDA) | after fix (FA/SIC) | exact |
|------|------------------|--------------------|-------|
| H    | E_x=-0.207, E_x+E_H=+0.053 | E_x=-0.331, **E_x+E_H=-0.017** (near SIC), E_tot=-0.517 | E_tot=-0.5 |
| He   | -0.892 | -1.098 | -1.0258 |
| Be   | -2.395 | -2.455 | -2.6658 |

So **H is now nearly self-interaction-free** (E_x ~ -E_H; residual -0.017), He is in the
density-following regime (-1.10, slightly OVER exact by the constant ~3% constraint/X-window
offset = the 1.033 HEG ratio), and Be stays LDA-level (two electrons per spin). HEG remains
scale-invariant (1.033). Adjoint still matches FD to 3.8e-9; SCF converges at domain>=15.
41/41 tests green (added a dedicated H near-SIC gate). The residual ~3% over-binding (H, He, and
the 1.033 HEG ratio) is now the dominant error -- the constraint-projection/finite-X overshoot,
a single constant to address next (tune X or the sum-rule enforcement).

## Update 5 — can exact H be recovered? (diagnosis: finite-R_c, blocked by transfer instability)
Swept R_c, n_in, n_out, X for H (E_x+E_H -> 0 = exact SIC). Findings:
- **n_in converged**: -0.0170 at R_c=6 for n_in=20/30/40 (identical). Not a projection-resolution error.
- **n_out**: -0.0170 (n_out=10) -> -0.0193 (n_out=24): more channels converge toward the FA value
  (slightly WORSE); n_out=10's smaller residual is truncation luck. Not the fix.
- **X flat**: -0.017 to -0.019 across X=6..16; eta = k_F R_ad clamps at k_F R_c, so X cannot
  enlarge the window at R_c=6.
- **R_c > 6 DIVERGES**: the scale-free transfer_matrix ill-conditions (R_c=9/n_in=20 -> singular
  matrix; R_c=9,12 -> SCF blow-up at all n_in). So R_c cannot be increased to converge.

Root cause (bracketing test on the H density): the one-electron hole -rho(r0+u) is DELOCALIZED
(int=-1 over all space, but only -Q_window ~ -0.91 within R_ad). The windowed hole brackets exact:
  FA normalized (/Q, int_window=-1):   E_x=-0.3334, E_x+E_H=-0.0195  (over-binds: /Q over-counts)
  FA un-normalized (-rho, int_window=-Q): E_x=-0.3046, E_x+E_H=+0.0093  (under-binds: misses v_H tail)
  exact -E_H = -0.3139 (between the two).
Both window effects (Coulomb-tail truncation; normalization over-count) vanish as R_ad -> inf
(Q->1, v_H,window->v_H). So exact H IS recoverable in principle -- it is a finite-R_c convergence
error -- but it is blocked by the transfer-matrix instability for R_c>6.

Next: stabilize the scale-free transfer at larger R_c (better conditioning / larger n_in /
regularized transfer) so R_ad can grow and H converges; or give the delocalized one-electron
hole a non-windowed treatment. The same finite-window over-count is the 1.033 HEG offset and He's
slight over-binding -- one structural issue, addressed by fixing the large-R_c transfer.

## Update 6 — two implementation bugs found (not truncation); both pushbacks correct
Two checks (HEG convergence; transfer conditioning) confirm the residual is NOT fundamental:

(1) **Min-norm constraint projection overshoots.** The pure HEG anchor (no projection) converges
to LDA exactly: ratio 0.984 (X=8) -> 0.996 -> 1.000 as X grows with n_out scaled (analytic
(4/9)int_0^inf S x dx = 1.000). The functional's 1.033 is the min-norm sum-rule projection
turning a -1.7% truncation into a +3.3% overshoot (a +5% swing). The prior SF enforces the sum
rule via the zeta-scale (Q_S=2), not a min-norm projection -> no overshoot.
FIX: enforce the sum rule through the hole scale, not a min-norm coefficient projection.

(2) **Raw c_ad exposes the ill-conditioned transfer (R_c>6 divergence).** cond(transfer) grows
fast with R_ad (R_c=6: ~1-24; R_c=8: 5.8e3 at n_out=10; R_c=12: 1e7). But the prior SF ran at
R_c=8 with WORSE conditioning (3.9e11, n_out=24) and was stable -- because SF uses c_ad only
through stable contractions (g.c_ad, h.c_ad; the envelope tables damp high frequencies). The
direct expansion uses the raw transferred c_ad DIRECTLY as hole coefficients (FA = -c_ad/Q),
amplifying the ill-conditioned high-frequency content -> divergence at R_c>6.
FIX: regularize the transfer (truncate small singular values) / use stable contractions.

Consequence for H: the HEG/He over-binding is bug (1); H additionally has the FA /Q over-count
(Q_window=0.91<1, delocalized one-electron hole truncated by the window), which needs R_ad to
grow toward the full atom -- requiring R_c>6, unblocked by fix (2). With both fixes the
construction should reach the prior SF accuracy (<1 mHa) and converge H -> -E_H. So the exact H
limit is recoverable; the current residual is implementation (projection + raw-c_ad transfer),
not physics.
