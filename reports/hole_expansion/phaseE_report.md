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

## Update 4 — point-wise confirmation, scale-free check, and the iso-orbital limit
Three findings while exploring the L2-from-HEG gate (and renaming the parameter c -> alpha_lda,
since c denotes the projected coefficients C_n):

1. **D_HEG is point-wise** (one value per r0). Its profile is *largest* in the dense core and
   *smallest* in the tail in the fixed-R_c form; in the proper SCALE-FREE form (pipeline,
   adaptive radius) it is small (~0.01) in He's smoothly-varying valence -- i.e. He *does* have
   HEG-like regions, so GEA turns on there and over-corrects He as the gate strength is lowered
   (He: -1.028 -> -1.046 -> -1.086 as alpha_lda 1 -> 0.1 -> 0.03).

2. **The fixed-R_c monopole distance is NOT scale-invariant** (a hydrogenic 1s gives a
   Z-dependent C_n/C_0), so a single H-1s reference cannot work there. The full SIMPLE pipeline
   descriptors (adaptive-radius non-dimensionalization) ARE scale-invariant: uniform density ->
   (-1)^n/(n+1) (Z-independent), and a hydrogenic 1s gives one universal signature for Z=1,2.
   The H-1s construction therefore requires the scale-free descriptors, not the fixed-R_c ones.

3. **The H-1s iso-orbital cancellation hits the fundamental iso-orbital problem.** Using the
   scale-free descriptors, the distance to the H-1s *manifold* (the curve of signatures a 1s
   traces over position -- a single point won't do, since one orbital gives different signatures
   at cusp/valence/tail) is small for ALL of He (0.011-0.023, correctly single-orbital) -- but
   it is *also* small for a slowly-varying ramp (0.033) and most of Be. The monopole density
   shape cannot separate a one-electron region from a slowly-varying multi-electron region (they
   can share rho, grad rho, lap rho). This is exactly why meta-GGAs use the kinetic-energy
   density tau (tau_W/tau) as the iso-orbital indicator, not the density. So the H-1s-distance
   gate, with monopole density features alone, cannot cancel He's HEG-like valence without also
   suppressing genuine slowly-varying GEA.

**Path forward.** A clean iso-orbital cancellation that recovers H/He while freeing alpha_lda
needs an indicator beyond the monopole density: (i) the kinetic-energy density tau (a meta-GGA
ingredient -- would require carrying tau in this exchange-only hole), or (ii) the higher-l
SIMPLE invariants (power spectrum / bispectrum at l>=1), which encode the orbital phase/gradient
structure the monopole channel lacks and may distinguish single-orbital from slowly-varying.
The parameter rename (alpha_lda) and the scale-free-descriptor analysis are in place; the gate
itself remains the monopole L2-from-HEG form (alpha_lda default 1) pending that indicator.

## Update 5 — basis alignment, l>=1 separation, and the two-parameter gate
Addressing three review points:

**(1) Shared basis (confirmed) + R_c=6/n=16.** The density projection C_n and the hole
expansion rhotilde_n use the identical SIMPLE basis (same R_c, same n_channels, same R_n0) --
required so the exchange integral becomes a sum and the FA anchor (rhotilde=-C/Q) is defined.
Aligned to the canonical R_c=6 (was 8). The hole EXPANSION needs n_in resolution
(n >~ k_F R_c/pi ~ 16 at R_c=6); the exposed n_out=10 is too few (Be diverges at n=10). So
n_channels=16 (=n_in); n_out=10 is the reduced feature count. 6/16 also beats 8/24 (He -1.030,
Be -2.640) and matches the orbital reference data.

**(2) l>=1 cleanly separates single-orbital from slowly-varying.** The monopole (l=0) distance
could NOT separate them; including the l=1 (gradient/dipole) and l=2 channels does, because a
radial density has nonzero l>=1 multipoles about off-center points and an orbital's gradient
structure differs from a gas. Full l=0,1,2 SIMPLE distances:

| region | D_HEG (l=0,1,2) | D_H1s (manifold) |
|--------|-----------------|------------------|
| He (r0=0.3-1.0) | 0.75 -> 3.4 (far from HEG) | 0.20 -> 0.49 (near H1s) |
| Be (r0=0.3-1.0) | 1.8 -> 7.7 | 0.37 -> 0.68 |
| slowly-varying ramp | 0.001-0.01 (HEG-like) | 0.90 (far from H1s) |

(l=0 only gave He valence D_HEG~0.01, indistinguishable from the ramp.) p-/d-orbital
reference signatures (l=1, l=2 indicators) are the natural extension for open-shell atoms.

**(3) Two-parameter gate; GEA2 kept exact by normalization; one DOF to sweep.**

    gate(r) = exp(-alpha_LDA D_HEG(r)) * [1 - exp(-alpha_H1s D_H1s(r))] / N,
    N = 1 - exp(-alpha_H1s D_H1s^HEG),   D_H1s^HEG = 0.904 (uniform's distance from the H1s manifold).

The /N normalization makes the slowly-varying limit (D_HEG->0, D_H1s->D_H1s^HEG) give the full
GEA2 enhancement BY CONSTRUCTION -- GEA2 is kept intact for any (alpha_LDA, alpha_H1s). He GEA
fraction (gated/ungated, density-weighted; ->0 = He exact):

| alpha_LDA \ alpha_H1s | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|
| 0.5 | 0.115 | 0.129 | 0.154 | 0.196 |
| 1.0 | 0.043 | 0.048 | 0.059 | 0.078 |
| 2.0 | 0.011 | 0.013 | 0.016 | 0.022 |
| 4.0 | 0.002 | 0.002 | 0.003 | 0.004 |

alpha_LDA does most of the He cancellation (He is far from HEG once l>=1 is included); alpha_H1s
is the secondary knob. He is ~99% recovered by alpha_LDA~2, with GEA2 exact by the
normalization. The ratio is set empirically for He-exact; one DOF (overall scale) remains to
sweep for the Be/general balance.

**Status:** the construction is VALIDATED at the analysis level (fixed densities). Implementing
it self-consistently requires the full l=0,1,2 SIMPLE features per grid point (multipole
profiles + adaptive-radius pipeline + manifold distance) inside the SCF and its discrete
adjoint -- the substantial next build. The current shipped gate remains the monopole
L2-from-HEG form (alpha_lda); the l>=1 two-parameter gate is the validated design for it.

## Update 6 — iso-orbital gate (LDA-distance + H1s-distance), non-self-consistent test
On the corrected scale-free base (H/He near-exact), the GEA2 gradient enhancement is gated to
turn ON only in slowly-varying regions and OFF in single-orbital regions:
    eps_x = eps_base * (1 + g * mu * s^2_b),   mu = 10/81,
    g(r)  = exp(-alpha_LDA D_HEG) * (1 - exp(-alpha_H1s D_H1s)) / Nrm,
    Nrm   = 1 - exp(-alpha_H1s D_H1s^HEG)   [D_H1s^HEG = 0.904 -> g -> 1 in the slowly-varying limit].
D_HEG = ||varrho_nlm - varrho_HEG||^2 (l=0,1,2 scale-free SIMPLE features), D_H1s = distance to
the hydrogenic-1s manifold (Z-invariant). alpha_LDA = alpha_H1s = 1.

Non-SCF test (gate + perturbative GEA2 on converged base densities):
| density | D_HEG | D_H1s | gate | dE_x |
|---------|-------|-------|------|------|
| slowly-varying ramp / density wave | ~0 | ~0.90 | **1.000** | (full GEA2) |
| H (cusp..tail) | 0.9..13 | 0.001..0.035 | **~0** | -0.0000 (preserved) |
| He | 0.35..12 | 0.15..0.48 | 0..0.17 | -0.0043 |
| Be | 0.17..7.6 | 0.37..2.6 | 0..0.43 | -0.0138 |

Verdict: the gate behaves exactly as designed -- g=1.000 for slowly-varying (GEA2 intact via
/Nrm), g~0 for the single-orbital H (preserved exactly). He/Be get a small cusp leak (they are
not exactly hydrogenic-1s: D_H1s(cusp) ~ 0.15/0.37, vs ~0 for true H), giving small over-binding
(GEA2 adds binding; atoms already near/over-exact, so GEA2 is not their corrective -- it is the
correction for inhomogeneous bonding regions, not atoms). The construction is validated; the
cusp leak is the iso-orbital-measure limitation (hydrogenic-1s reference doesn't perfectly match
screened He/Be), addressable by a broader single-orbital reference or a tau-based indicator.

SCF wiring (next): needs efficient per-grid-point l=0,1,2 scale-free features (l=1,2 axial
multipole window operators + transfer, analogous to the l=0 ops) and a smooth (soft-min)
manifold distance for a differentiable adjoint.

## Update 7 — additive two-correction test (gated GEA2 + H1s anti-binding), non-SCF, H/He/Be/Ne
Tested the additive form  eps = eps_base + dGEA2 + dH1s, with
  dGEA2 = exp(-alpha_LDA D_HEG) mu s^2 eps_base       (LDA-gated GEA2; adds binding)
  dH1s  = -A_H1s (1-exp(-alpha_H1s D_H1s)) eps_base   (H1s-gated; anti-binding; grows with Z)
on converged densities (alpha_LDA=4, alpha_H1s=1).

| atom | E_base | dGEA2 | <g_H1s> | exact | base error |
|------|--------|-------|---------|-------|------------|
| H    | -0.3098 | -0.0000 | 0.014 | -0.3098 (FA) | exact |
| He   | -1.0274 | -0.0007 | 0.320 | -1.0258 | +1.6 mHa over |
| Be   | -2.7135 | -0.0033 | 0.614 | -2.6658 | +48 mHa over |
| Ne   | -11.758 | -0.057  | 0.643 | -12.108 | -350 mHa UNDER |

Findings:
- **<g_H1s> grows with Z** (0.014, 0.32, 0.61, 0.64) -- the H1s indicator turns on increasingly
  with Z, as hypothesized; H (true 1s) ~ 0 so the FA limit is recovered automatically.
- The anti-binding H1s correction has the right SIGN for the over-binding light atoms (He, Be).
- BUT (1) no single A_H1s fits He and Be: their over-binding differs 30x (1.6 vs 48 mHa) but
  their H1s weights only 3.7x, so He wants A~0.005, Be wants A~0.04. (2) Ne UNDER-binds
  (-350 mHa, opposite sign); the anti-binding correction makes Ne worse.
- Ne's under-binding is the missing l>=1 HOLE anisotropy (the 2p shell): the monopole hole can't
  represent the anisotropic p-exchange hole. This is a *hole* ingredient (explicit l>=1 hole
  multipoles), NOT an l<=2 indicator/gate -- a different and more fundamental addition.

Conclusion: the H1s-indicator construction is conceptually right (Z-scaling, FA-preserving, fixes
the sign of light-atom over-binding) but the base error is not a monotonic over-binding it can
absorb -- He is already near-exact, Be slightly over, Ne under for a structural reason. Heavier
(p,d) atoms need the l>=1 hole multipoles (the "richer functional" route, CODEMAP open item),
which the gated-GEA2 + H1s-indicator (both l=0-hole) corrections do not provide.

## Update 8 — s-atom series references (He/Li/Be/Na/Mg): base error changes sign
Computed exact exchange (EXX oep_exchange) and the base scale-free hole exchange to calibrate the
H1s-correction magnitude on heavier s-atoms (per review):

| atom | Z | shell | E_x(exact) | E_x(base) | base error |
|------|---|-------|------------|-----------|------------|
| He | 2 | 1s^2          | -1.0258 | -1.0274 | -1.6 mHa (over)  |
| Li | 3 | 1s^2 2s^1     | -1.7106 | -1.7834 | -72.8 mHa (over) |
| Be | 4 | 1s^2 2s^2     | -2.6658 | -2.7136 | -47.8 mHa (over) |
| Na | 11| ..2p^6 3s^1   | -13.950 | -13.464 | +485.7 mHa (UNDER) |
| Mg | 12| ..2p^6 3s^2   | -15.988 | -15.273 | +715.6 mHa (UNDER) |

The base error CHANGES SIGN: the truly s-only atoms (He, Li, Be -- no p electrons) OVER-bind;
Na/Mg carry a closed 2p^6 shell and UNDER-bind by 486/716 mHa (same direction/mechanism as Ne).
So Na/Mg cannot calibrate the anti-binding H1s magnitude -- the correction would worsen them.
And within the s-only set the over-binding is non-monotonic (open-shell Li 4.3% > Be 1.8% > He
0.16%), while <g_H1s> grows monotonically with Z -- so a single magnitude cannot track it.

Conclusion (reinforced by the heavier-atom data, not just Ne): the dominant heavier-atom error is
UNDER-binding from the missing l>=1 hole anisotropy (every p-shell atom: Ne, Na, Mg), which the
anti-binding H1s/GEA2 indicator corrections (l=0 hole) cannot supply -- they have the wrong sign
for these. The H1s correction can only nudge the small closed-shell s-only over-binding (He, Be),
which is already near-exact. The l>=1 HOLE multipole expansion is the needed ingredient for
heavier atoms.

## Update 9 — REGIME CORRECTION: prior heavy-atom analysis was all-electron; PSP is the target
The Update-8 "base error changes sign / Na-Mg under-bind / l>=1 hole anisotropy" conclusion was
computed with all_electron_flag=True. That is the WRONG regime: the functional targets
pseudopotential calculations (Table-I: He -1.0019, Be -1.92, Ne -5.46). All-electron pulls in
the bare 1s core (rho(0) ~ Z^3; Ne core rho ~ 620), and the scale-free map has a genuine but
SEPARATE bug there (Update 10). PSP removes the core; max valence density is only ~1.3-2.8.

PSP-regime series (R_C = 6.0 bohr, all_electron_flag=False):

| atom | Z | E_x(EXX) | E_x(base) | base err | E_x(l-hole) | max rho | has p? |
|------|---|----------|-----------|----------|-------------|---------|--------|
| He | 2 | -1.0019 | -1.0022 |   -0.3 | -1.0019 | 1.35 | no  |
| Be | 4 | -1.9208 | -1.9924 |  -71.6 | -1.9207 | 2.83 | no  |
| Na | 11| -5.7751 | -6.0521 | -276.9 | -5.7748 | 1.62 | yes |
| Mg | 12| -6.9715 | -7.2407 | -269.2 | -6.9711 | 2.40 | yes |
(Ne PSP EXX did not converge this run; revisit.)

KEY: in PSP there is NO sign flip. EVERY atom OVER-binds, growing ~monotonically with Z
(He -0.3 -> Be -71.6 -> Na/Mg ~-270). Per-electron over-binding is ~27-36 mHa/e and is NOT
specifically p-driven (Be, no p, is the largest per electron). This is exactly the regime the
anti-binding H1s correction was designed for: a consistent over-binding that grows with Z, so a
single magnitude (with the He-exact alpha-ratio fixed by construction) can be calibrated on the
Be/Na/Mg series. The general-l orbital hole reproduces EXX to <1 mHa for PSP p-shell atoms too.
=> The l>=1-hole direction was chasing an all-electron artifact; revert to calibrating the
anti-binding correction on the PSP over-binding series.

## Update 10 — all-electron high-density map bug (documented, not the PSP path)
For completeness: in all-electron Ne the map's monopole hole under-binds the 1s core by +1280 mHa
(eps_map -2.68 vs exact -4.18 ~ LDA -4.12), partially cancelled by -900 mHa valence over-binding,
net +350. Root cause (uniform-density probe): the scale-free map's HEG limit is density-DEPENDENT
and collapses at high rho -- eps_map/eps_LDA = 0.99 (rho=2), 0.88 (rho=10), 0.66 (rho=100),
0.55 (rho=620), 0.52 (rho=1000) -- though it must be density-INDEPENDENT (scale-free). Mechanism:
the fixed-R_c=6 window cannot be faithfully transferred to small R_ad = X/k_F (~0.30 at Ne core).
More n_in helps but plateaus (~0.71 at rho=620); larger X is worse. This bug is real but only
bites all-electron cores, so it is OUT OF SCOPE for the PSP target. Tools: validate_lhole.py,
diagnose_map_ne.py, decomp_be_ne.py, uniform_highrho.py, fix_probe.py.

## Update 11 — PSP anti-binding calibration works; He preservation is the open knob
Non-SCF test on PSP densities (alpha_H1s=1.0, gate g_H1s = 1-exp(-alpha_H1s D_H1s), correction
dH1s = -A g_H1s eps_base; lever H1s_w = sum ew rho eps_base g_H1s):

| atom | E_base | E_exx | err(mHa) | <g_H1s> | A=err/lever |
|------|--------|-------|----------|---------|-------------|
| He | -1.0022 | -1.0019 |   -0.3 | 0.31 | 0.001 |
| Be | -1.9924 | -1.9208 |  -71.6 | 0.57 | 0.077 |
| Na | -6.0521 | -5.7751 | -276.9 | 0.56 | 0.085 |
| Mg | -7.2407 | -6.9715 | -269.2 | 0.59 | 0.068 |

A clusters at 0.068-0.085 across Be/Na/Mg -> a single magnitude. With mean A=0.076: Be -0.4,
Na -27.5, Mg +35.2 mHa residual (was -71/-277/-269): ~8x reduction. BUT He over-corrected to
+23.9 mHa because g_H1s(He)=0.31 != 0. The anti-binding correction must VANISH at the
one-electron-per-spin limit (Q_sigma -> 1) that already makes He's base exact. Recommended:
gate the correction on the base's own one-electron-per-spin indicator (not the hydrogenic-
manifold distance) so He is preserved by construction; then A is the lone free DOF calibrated
on Be/Na/Mg. Tool: psp_calibrate.py.

## Update 12 — He-pinned two-term construction: one DOF, ~30 mHa floor (PSP, non-SCF)
Correct construction (per review): TWO additive gated corrections with OPPOSITE sign whose
ratio is fixed by demanding they CANCEL for He (FA limit preserved), leaving the overall
magnitude as the lone DOF:
  eps = eps_base + M_G g_HEG s^2 eps_base + M_H g_H1s eps_base
  g_HEG = exp(-aHEG D_HEG)  (gradient/GEA2 term, on in HEG-like regions)
  g_H1s = 1 - exp(-aH1s D_H1s)  (anti-binding, zero at the H1s/one-electron limit)
He-exact: M_G I_G(He) + M_H I_H(He) = -err(He)  -> fixes M_H/M_G. H is exact automatically
(on the manifold, g_H1s=0). One free M_G calibrated on Be/Na/Mg.

Non-SCF on PSP densities (cached SCF + features; scan is cheap). He is exact (0.0) at EVERY
(aHEG,aH1s). Best one-DOF (min-max over Be/Na/Mg): aHEG=0.5, aH1s=8, M_G=0.628, M_H/M_G=-0.19:
  base err (mHa): He -0.3, Be -71.6, Na -276.9, Mg -269.2
  corrected (mHa): He 0.0, Be -29.5, Na -30.6, Mg +29.4   (2-9x reduction; ~0.4-1.5% of E_x)

FLOOR: across the whole alpha grid Na sits at ~-30 and Mg at ~+29 (~60 mHa apart, opposite
signs). Na wants more correction, Mg less; one magnitude cannot split them -- the irreducible
one-DOF residual against the Na/Mg pair. The gate shape only trades this against Be (smaller
aHEG hurts Be, larger helps Be but worsens Na). Zero-parameter variant (M_G = mu = 10/81)
fails (Be/Na/Mg -35..-280 mHa): the magnitude must be free, and the effective M_G (~0.6) is
~5x mu, so the "GEA2" term acts as a tunable counterweight, not the physical gradient slope.
Tools: psp_cache.py (slow, -> psp_cache.npz), psp_scan.py (cheap).
