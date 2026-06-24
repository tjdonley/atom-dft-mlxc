# Phase B — Parameter-free map: anchors, enclosed-charge switch, constraints

**Provenance:** R_c = 6 bohr, n_channels = 16, nu = 1024–2048. Module
`atom/xc/simple_hole_expansion_explicit.py` (map functions); gates
`tests/simple/test_simple_hole_expansion.py` (Phase B block). 17/17 gates green.

## The map `M: {density monopole} -> {hole coeffs}`
```
C_n        = int rho_avg(u) R_{n0}(u) u^2 du              # density monopole (= P_n rho)
Q_window   = 4 pi sum_n C_n a_n                            # enclosed density charge
lambda     = switch(Q_window)                             # 1 (Q<=1) -> 0 (Q>=2), C^2 quintic
rhotilde^HEG = project(-(rho0/2) S(k_F u))   (k_F from on-top rho0)   # lambda=0 anchor
rhotilde^1e  = -C_n                                                    # lambda=1 anchor
rhotilde   = (1-lambda) rhotilde^HEG + lambda rhotilde^1e
rhotilde  <- constraint_project(rhotilde; sum rule = -1, on-top = -W rho0),  W = (1+lambda)/2
```

## Key correction to the plan: the on-top value is **not** universally −ρ/2
Verified against `simple_hole_explicit.hole_solve`: the production hole's `W = 1/Q_S`
prefactor gives on-top `−W ρ`, which is `−ρ/2` in the bulk (W=1/2, pair) but `−ρ` in the
one-electron limit (W=1). So the on-top *target* must interpolate with the same switch:
`W = (1+λ)/2`. Encoding this makes both anchors meet their own on-top and the blend
interpolate correctly. (Uniform: W=0.5; H1s-center: W=1.0 — both reproduced.)

## Verified limits
- **B1 (HEG, λ=0 → LDA):** uniform density gives `eps_x/eps_lda` = 1.0096 / 1.0037 / 1.0212
  at ρ = 0.5 / 2 / 5, with sum rule and on-top **exact to 1e-6**. The small overshoot is a
  finite-window artifact (see below).
- **B2 (one electron, λ=1 → SIC):** H-like 1s at the center → λ=1, `eps_x(0) = −0.520`
  (exact self-interaction correction −½ v_H(0) = −0.5; residual is N-truncation of the
  Coulomb kernel, improvable in N). Sum rule = −1 and on-top = −ρ₀ exact; the projection
  perturbs the pure 1e anchor by <1e-2.
- **B3 (switch):** λ(Q) is monotone non-increasing and C² (quintic smoothstep on [1,2]).

## The sum-rule / energy tension (resolved, documented)
Enforcing the sum rule `∫n_x = −1` inside a finite window is in tension with the energy:
the lost charge lives in the hole *tail past R_c* (low 1/u weight, little energy), but a
min-norm restoration adds it broadband (more energy) → a small LDA overshoot. Tested
weightings:

| enforcement | ρ=0.5 | ρ=2 | ρ=5 | 1e SIC |
|---|---|---|---|---|
| none (natural projection) | 0.995 | 0.998 | 0.988 | — |
| on-top only | 0.995 | 0.998 | 1.013 | ok |
| **sum+on-top, unweighted (chosen)** | 1.010 | 1.004 | 1.021 | −0.520 ✓ |
| sum+on-top, w=k⁴ (low-k favored) | 1.012 | 1.004 | 1.551 | −0.985 ✗ |

High-k-penalizing weights wreck the high-density HEG limit and the 1e limit, so **unweighted
min-norm is chosen**. The HEG overshoot shrinks monotonically with the window
(ρ=0.5: 1.0096 → 1.0052 → 1.0034 → 1.0017 at R_c = 6/8/10/14), confirming it is a controlled
finite-R_c effect — and it is **absent in the scale-free production frame** (Phase D), where
the hole is always projected at fixed ξ* so resolution and tail capture are
density-independent. For the prototype this ≤~1% valence-density deviation is accepted in
exchange for exact constraint satisfaction and an exactly self-interaction-free one-electron
limit.

## Update — per-spin switch + Fermi-Amaldi anchor (corrects the spin convention)
The map above keyed the HEG↔SIC switch on the **total** enclosed charge and used the
unscaled `−C` one-electron anchor. Exchange is a *same-spin* interaction, so the
self-interaction-free limit is one electron **per spin** (`Q_σ = Q_total/2 ≤ 1`). The map now:
- switches on the **per-spin** charge `Q/2` (so spin-paired He, `Q_total=2 → Q_σ=1`, is in the
  density-following limit);
- uses the **Fermi-Amaldi anchor** `−C/Q` (density-following, `∫ = −1` by construction), whose
  on-top `−ρ0/Q` gives `−ρ0` for one electron (`Q=1`) and `−ρ0/2` for paired He (`Q=2`) — the
  spin factor falls out of `/Q`;
- on-top target blends `(1−λ)(−ρ0/2)` [HEG pair] + `λ(−ρ0/Q)` [FA].

Limits preserved (HEG λ=0 → LDA; one-electron-per-spin λ=1 → SIC), and spin-paired He becomes
essentially exact (see Phase-D update).
