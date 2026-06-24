# Phase E — Parameter-free second-order gradient correction (GEA2)

**Provenance:** R_c = 8 bohr, n_channels = 24, l=1 spectral gradient operator (n_channels=40,
default). Class `SIMPLE_HOLE_EXPANSION_GGA` in `atom/xc/simple_hole_expansion.py`; gates
Phase-E block of `tests/simple/test_simple_hole_expansion.py`. 33/33 gates green.

## What was built
`SIMPLE_HOLE_EXPANSION_GGA(SIMPLE_HOLE_EXPANSION)` adds the exact second-order
gradient-expansion enhancement, parameter-free:

    eps_x = eps_x^map * (1 + (10/81) s^2_bounded),   s = |grad rho| / (2 k_F rho),

implemented as the energy effect of a charge- and on-top-neutral deformation of the hole.
`s` comes from the **proven-stable** ℓ=1 spectral gradient operator
(`build_spectral_gradient_operator`, k_n¹ growth — no stiff ℓ=0 Laplacian). `s²` is smoothly
saturated (`_bound`, Lieb–Oxford tail safety). The self-consistent potential adds the gradient
channel via the spectral-operator transpose:
`v_x = eps + rho f deps0/drho0 + Σ_n P_n^T[ew rho f deps0/dC_n]/ew + rho eps0 (df/drho) + G^T[ew rho eps0 (df/dg)]/ew`.

## Verified
- **E1 (GEA2 slope):** `F_x = eps_GGA/eps_LDA → 1 + (10/81) s²` — measured slope matches
  10/81 = 0.1235 to <2% at small s² (e.g. F_x = 1.00160 vs 1.00160 at s² = 0.013).
- **E1 (gradient adjoint):** the full discrete-adjoint `v_x` (C-channel + explicit-ρ +
  gradient channel) matches FD `dE_x/dρ` to **4.3e-9** (gate 5e-6). [A factor-ρ bug in the
  local `df/dρ` term — which blew up as ~1/ρ in the tail — was caught by this FD check and
  fixed.]
- **E1 (limit preserved):** at zero gradient (uniform density) `f = 1` exactly and the GGA
  energy density reduces to the gradient-free expansion (HEG and one-electron limits intact).
- **E1 (SCF + improvement):** all-electron SCF converges; the gradient correction moves atoms
  toward exact exchange:

  | atom | LDA-level (EXPANSION) | + GEA2 (EXPANSION_GGA) | exact |
  |------|-----------------------|------------------------|-------|
  | He   | −0.863 (16% under)    | −0.949 (7.5% under)    | −1.0258 |
  | Be   | −2.477 (7% under)     | −2.812 (5.5% over)     | −2.6658 |

## Assessment
The parameter-free GEA2 correction is recovered exactly (slope and limits) and is fully
self-consistent (exact adjoint). It substantially improves He (16% → 7.5%) and, as expected for
the bare GEA2 coefficient, over-corrects the sharp-core Be (the well-known reason production
GGAs use a fitted/saturated enhancement rather than the bare 10/81). This is the intended
LDA → GGA step on the direct-expansion hole: the next accuracy gains come from a
density-/feature-dependent enhancement (the learnable map, Phase F) and a proper saturation,
rather than the single universal GEA2 coefficient.

## Update — on the spin-corrected base, bare GEA2 overshoots
With the per-spin map (Phase-B/D updates) the base He is already near-exact (−1.028), so adding
the bare GEA2 gradient term overshoots: He −1.147, Be −2.802 (both past exact). This is the
expected, instructive outcome — the universal 10/81 coefficient is too aggressive on a correct
base, which is precisely why production GGAs use a *saturated/fitted* enhancement. The slope
(10/81), the exact gradient adjoint (4.3e-9), and the zero-gradient limit are all still
recovered; the takeaway is that the enhancement should be **feature-dependent and saturated**
(the limits-safe learnable layer, Phase F), not a single universal coefficient.

## Update 2 — gate GEA2 by (1-lambda): don't touch already-exact results
GEA2 is an exact constraint of the *slowly-varying* limit, not of the one-electron-per-spin
limit. Applying it everywhere (Update 1) corrupted the already-exact He. The enhancement is now
gated by the HEG-branch weight:

    eps_x = eps_map * (1 + (1 - lambda) (10/81) s^2_bounded).

Because lambda -> 1 in the one-electron-per-spin (Fermi-Amaldi) branch, the gradient term
switches off there (analogous to SCAN disabling its gradient term in single-orbital regions),
so an already-exact result is left untouched. The gate depends on C (through lambda(Q)), so the
self-consistent potential is the full discrete adjoint of the gated energy, taken by FD in all
three channels (C, on-top rho, gradient g); verified against FD `dE/drho` to 4.7e-8.

| atom | base | +GEA2 ungated | +GEA2 (1-lambda) gated | exact |
|------|------|---------------|------------------------|-------|
| He   | -1.028 | -1.147 | **-1.028 (unchanged)** | -1.0258 |
| Be   | -2.469 | -2.802 | -2.802                 | -2.6658 |

He is now preserved exactly (lambda=1 -> gate off). Be is unchanged by the gate because its
dense 1s core encloses ~2 electrons per spin (Q_sigma~2 -> lambda=0 -> "HEG branch"), so it
still receives full GEA2 and still over-enhances. That residual Be overshoot is a *separate*
issue: (i) the bare 10/81 is too large for real inhomogeneous systems (production GGAs use a
saturated/fitted coefficient), and (ii) a global enclosed-charge lambda is a crude
single-orbital detector for a sharp core. Both point to a feature-dependent, saturated
enhancement (the limits-safe learnable layer, Phase F) rather than the universal GEA2 constant.

## Update 3 — L2-distance-from-HEG gate (intrinsic SIMPLE inhomogeneity detector)
Replaced the enclosed-charge (1-lambda) gate with a gate keyed on the **L2 distance from the
HEG limit in SIMPLE feature space** -- the natural, scale-free inhomogeneity measure:

    eps_x = eps_map * (1 + g(C) * mu * s^2_b),   g(C) = exp(-c * D_HEG),
    D_HEG = sum_n (C_n/C_0 - (-1)^n/(n+1))^2 .

The non-dimensional SIMPLE monopole features vanish at HEG; D_HEG is their squared L2 norm
(monopole channel). Verified D_HEG = 0 (to 1e-12) for any uniform density. The gate turns GEA
on (g->1) only when the local density is HEG-like, and off (g->0) in inhomogeneous regions, so
it leaves already-exact results untouched. Parameters: `gea_gate_c` (c, default 1.0) and
`gea_mu` (mu, default 10/81). Adjoint (FD-all-channels, the gate's C-dependence included)
matches FD dE/drho to ~5e-8.

**Slope.** Because D_HEG ~ s^2 across the window, g = 1 - c k s^2 + ..., so the gated
enhancement is mu s^2 (1 - c k s^2) = mu s^2 - O(s^4). The s^2 coefficient is exactly mu in the
s->0 limit -- GEA2 is recovered WITHOUT tuning (measured effective slope 0.1234 vs 10/81 =
0.1235 at s^2 = 0.015). The gate's density-derivative contributes only an O(s^4) self-saturation
(the effect anticipated in the request); `gea_mu` is exposed to tune the *effective* enhancement
at finite gradient if desired.

**Gate-strength sweep (SCF, c with mu=10/81):**

| c | He E_x | Be E_x | (exact He -1.026, Be -2.666) |
|------|--------|--------|---|
| 1.00 (default) | -1.028 (exact, protected) | -2.472 (LDA-level, untouched) | |
| 0.10 | -1.046 | -2.554 | |
| 0.03 | -1.086 | -2.621 | |
| 0.01 | -1.118 | -2.693 | |

At the default c=1 the gate is highly selective: atoms (He <D>~33, Be <D>~81) are far from HEG,
so GEA is essentially off and both already-decent base results are preserved (He exact, Be
LDA-level -- and Be no longer overshoots as it did under the enclosed-charge gate). Lowering c
turns GEA on more broadly and improves Be toward exact, but it turns on for **He before Be**
(He is *closer* to HEG in the monopole L2 measure), so a single c cannot protect He while fixing
Be. That is an honest limit of the monopole-only distance; the full SIMPLE distance (l>0
channels, which carry the gradient/anisotropy) or a tau-based single-orbital indicator would
separate He and Be differently and is the natural next refinement.
