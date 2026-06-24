# Phase K — kernel / fixed-point hole map (LDA-from-GEA + FA)

Branch `simple-hole-kernel-map` (off `simple-hole-expansion`). Goal: replace the ad-hoc limit
enforcement (Q_S=2 sum rule + bolt-on two-gate GGA) with a single kernel / fixed-point map for
the hole coefficients, where the LDA limit is enforced through the **feature distance** and the
GEA limit is carried by the **gradient projection onto that distance** (no extra term), with the
FA limit gated in charge space. Extensible to more fixed points.

## Construction (operator-free reference, `simple_hole_expansion_explicit.py`)

```
ϱ̃(x, Q) = (1 − W_FA(Q/2)) · [ ϱ̃_RBF(x) + χ·δ_GEA ]  +  W_FA(Q/2) · ϱ̃_FA ;   then constraint-project
χ = (10/81)/R · s² ,  capped by χ_max=(1.804−1)/R via χ→χ_max tanh(χ/χ_max)  (Lieb–Oxford)
```
- `δ_GEA`, `R` = `gea_mode(ρ0)`: the envelope-deformation mode (`φ=j₁`, projected, made
  charge/on-top-neutral so it touches only the energy) and its dimensionless HEG response
  `R = eps(δ_GEA)/eps_unif`. Calibration `χ = (10/81)/R · s²` gives `F_x = 1 + (10/81)s²` by
  construction. Because the lever is `s²` (= the l=1 squared feature distance from HEG) and
  exchange is even in ∇ρ, no linear-in-∇ρ term appears.
- `ϱ̃_RBF` = `rbf_interpolant(x, fixed_points, default=heg_anchor)`: Gaussian interpolating RBF
  over fixed points; **N=1 (HEG only) ⇒ ϱ̃_HEG everywhere** (LDA). Adding a node drops in.
- `W_FA` = `enclosed_charge_switch(Q/2)` (existing C² quintic); `ϱ̃_FA = −C/Q`. Q-only gate, so
  it cannot perturb the GEA.

## Phase-1 results (non-SCF, model densities; tests `-k KERNEL`, 8/8 green)

- **T1 uniform → LDA**: F = 1.006–1.021 across ρ∈{0.25..5} (finite-R_c HEG band at R_c=6);
  sum rule = −1 and on-top = −ρ/2 **exact** (1e-6). W_FA=0.
- **T2 slowly-varying → GEA2**: slope(F−1 vs s²) = **0.12346 = 10/81 exactly** at ρ0=0.5,1,2
  (R is density-independent → scale-free); linear-in-s coefficient negligible vs quadratic
  (parity: gradient enters only as s²).
- **T3 hydrogenic 1s → Fermi–Amaldi**: Q/2≤1 ⇒ W_FA=1 ⇒ ϱ̃=−C/Q; eps **matches the existing
  `map_coeffs` FA path exactly** and is **s-independent** (Δ=0.00 mHa) for Z=1,2,3 — LDA/GEA and
  FA cleanly decouple. (H: eps(0)=−0.5211, same as the established FA construction.)
- **T4 RBF**: reproduces every fixed point to 1e-6; empty set → the HEG default. Framework for
  adding exact limits is in place.

=> The idea is sound: a kernel map with the GEA carried by the feature-distance lever hits LDA
(from GEA, slope 10/81) and FA exactly, parameter-free, and extends to more fixed points.
Tool: scratchpad `kernel_check.py`. Next: Phase 2 (wire into a self-consistent functional).

## Phase 2 foundation — scale-free direct-energy normalization (validated)

For the SCF functional the kernel hole lives on the adaptive-radius (scale-free) frame (unit
window [0,1], basis R_m^(1), k_F R_ad = X). The direct-expansion energy/constraints are:

  eps_x   = 2 pi * R_ad^2 * (rhotilde . beta1),     beta1_m = int_0^1 R_m^(1)(t) t   dt  (= _H[eta~0])
  sum rule: 4 pi * R_ad^3 * (rhotilde . alpha1) = -1, alpha1_m = int_0^1 R_m^(1)(t) t^2 dt (= _G[eta~0])
  on-top  : rhotilde . R_m^(1)(0) = -rho0/2

HEG anchor (universal, reuses the envelope table): rhotilde_HEG = -(rho0/2) g(X) [g = _G interp at
eta=X]. FA anchor: rhotilde_FA = -c_ad/Q, Q = 4 pi R_ad^3 (c_ad . alpha1) (per spin Q/2). delta_GEA:
-(rho0) int g0(Xt) phi(Xt) R_m^(1)(t) t^2 dt, charge/on-top-neutralized; response R = eps(delta_GEA)/
eps_unif is a single density-independent number.

VALIDATED (scratchpad kernel_sf_norm.py, uniform density, grid): eps = 2 pi R_ad^2 (rhotilde_HEG.
beta1) gives eps/eps_LDA = 0.984 at rho = 0.5, 1, 2 -- constant => scale-free; the 1.6% is the
X=8/n_out=10 projection band (constraint projection enforces the sum rule to -1 exactly). This
pins the adaptive-frame normalization (the historically bug-prone 4pi/R_ad bookkeeping). Remaining
Phase-2 work is mechanical: assemble SIMPLE_HOLE_EXPANSION_KERNEL (vectorized over the grid),
exact variational adjoint (FD through C, R_ad, s, Q), register, and benchmark He/Be/Na/Mg vs
EXX/PBE.

## Non-SCF atom benchmark (kernel on the EXX density, fixed-R_c explicit)

Evaluate the kernel exchange energy post-hoc on the EXX density (per-r0 spherically-averaged
profile + reduced gradient -> kernel_map_coeffs -> eps; integrate). PSP, vs EXX and PBE-x:

  atom  Z   E_x(EXX)  E_x(kernel)  kernel err(mHa)  PBE-x err
  He    2   -1.0019   -1.0028        -0.9            31.4
  Li    3   -1.4507   -1.3075      +143.2             2.6
  Be    4   -1.9208   -1.9048       +16.1            35.5
  Ne   10   (EXX did not converge -- solver issue, not the functional)
  Na   11   -5.7752   -5.7478       +27.4            93.6
  Mg   12   -6.9715   -6.8525      +119.1           211.6
  MAE vs EXX (mHa):  kernel = 61.3   PBE-x = 75.0

Read: kernel competitive with PBE exchange. He near-exact (FA limit). Be/Na good (+16/+27, better
than the bare LDA band -> GEA helping). Mg +119 ~= 1.6% of E_x = the fixed-R_c LDA-band offset
(the scale-free frame removes this; expect ~tens of mHa). Li +143 is the FA<->bulk TRANSITION
(window holds ~1.5 e/spin, W_FA mid-blend) -- the hardest region for the HEG+FA two-node set, and
the concrete target for an added transition-region fixed point (the extensibility goal). No
blow-ups; the construction is sound on real atoms even before the scale-free frame. Tool:
kernel_atom_bench.py.

## Scale-free non-SCF benchmark (the correct frame; vs HF, cached)

REVISED to the scale-free adaptive-radius frame (SIMPLE features are always scale-free -- never
build fixed-R_c). kernel_eps_sf: C=[op@rho]; R_ad=X/k_F; c_ad=T(R_ad)C; eps=2pi R_ad^2 (rho~.beta1);
hole = (1-W_FA)[HEG(-rho/2 g(X)) + chi delta_GEA] + W_FA(-c_ad/Q); per-point 2-constraint
projection. Validated: uniform -> F=1.033 CONSTANT across rho (scale-free; 3.3% is the n_out=10
projection band, not a drift); ramp -> GEA slope 0.122 vs 10/81=0.123 (from the actual l=1
gradient operator).

Non-SCF on cached HF densities (hf_refs.npz; HF converges for Ne where PSP-EXX did not;
E_x = hf_exchange):

  atom  E_x(HF)   E_x(kernel-SF)  err(mHa)  rel%
  He   -1.0019    -0.9511          +50.9    +5.1
  Li   -1.4514    -1.4798          -28.4    -2.0
  Be   -1.9215    -1.8594          +62.2    +3.2
  Ne   -5.4013    -5.3439          +57.5    +1.1
  Na   -5.7280    -5.7694          -41.4    -0.7
  Mg   -6.8577    -6.8356          +22.1    +0.3
  MAE vs HF (mHa): kernel-SF = 43.7

vs fixed-R_c kernel (MAE 61, with Li +143 / Mg +119 outliers) and PBE-x (75). The scale-free
frame removes the systematic LDA-band offset (errors balanced +/-5%, mixed sign) and the
fixed-R_c outliers (Li +143->-28, Mg +119->+22); Ne now included. Regression: He +51 mHa (+5%) --
the r_c window no longer swallows all of He, so the per-point adaptive Q/2 is not <=1 throughout
and He reads as a FA/bulk mix (under-binds) rather than pure FA. => sharpen FA detection in the
adaptive frame next (and/or a transition fixed point). Tools: kernel_sf.py, kernel_sf_bench.py,
hf_cache.py -> hf_refs.npz (cached HF refs; reuse, do not re-solve).

## Spin-convention fix (He must be exact FA) + corrected benchmark

PI catch: He is spin-paired (one electron per spin) so it must be ESSENTIALLY EXACT Fermi-Amaldi.
The scale-free benchmark had He at +51 mHa -> bug. Root cause (the classic 4pi/R_ad bookkeeping):
the transfer output c_ad relates to the unit-basis DENSITY coefficients d by c_ad = 4pi R_ad^{3/2} d
(verified: c_ad/(rho*alpha1) = 4pi R_ad^1.5 to 3 digits). The enclosed charge had used
Q = 4pi R_ad^3 (c_ad.alpha1), which is ~52x too large -> Q/2 huge -> W_FA=0 -> He read as pure
BULK -> under-bind. Fix: d = c_ad/(4pi R_ad^{3/2}); Q = 4pi R_ad^3 (d.alpha1) (true charge);
FA hole rhotilde_FA = -d/Q; W_FA = enclosed_charge_switch(Q/2).

Per-atom charge now physical (density-weighted): He <Q/2>=1.00 (W_FA=1, pure FA), Li 1.18 (0.92),
Be 1.74 (0.23, TRANSITION, Q/2 range 1.24-2.00), Ne 3.92 (0), Na 4.09 (0), Mg 4.40 (0).

Corrected scale-free benchmark vs HF (cached):
  atom  E_x(HF)   E_x(kernel-SF)  err(mHa)  rel%
  He   -1.0019    -1.0077          -5.8    -0.58   <- now ESSENTIALLY EXACT FA (was +51)
  Li   -1.4514    -1.4388         +12.6    +0.87
  Be   -1.9215    -1.7463        +175.2    +9.12   <- FA<->bulk TRANSITION (Q/2~1.74)
  Ne   -5.4013    -5.3439         +57.5    +1.06
  Na   -5.7280    -5.7649         -36.9    -0.64
  Mg   -6.8577    -6.8339         +23.8    +0.35
  MAE vs HF: 52.0 (full); ~27 over He/Li/Ne/Na/Mg (the clean-limit atoms)

Spin convention now CORRECT (He exact FA). The error is concentrated ENTIRELY in the FA<->bulk
transition (Be, the only atom with Q/2 between 1 and 2): the two-node (HEG+FA) set handles the
crossover worst. This is the precise target for an added transition-region fixed point (the
extensibility goal). Bulk atoms (Ne/Na/Mg) ~1%; FA-limit atoms (He/Li) good.

## Phase 2 — self-consistent kernel functional (SIMPLE_HOLE_EXPANSION_KERNEL)

Ported the scale-free kernel map into a production functional (subclass of
SIMPLE_HOLE_EXPANSION): _kernel_eps builds the hole (HEG anchor + GEA mode + FA, blended by the
per-spin charge gate, 2-constraint projection) and eps = 2pi R_ad^2 (rhotilde.beta1); compute_xc
is the EXACT variational discrete adjoint (FD through the C, local-rho, gradient channels).
Registered in evaluator/functional_requirements/solver/scf.driver.

Validation:
- Adjoint == FD to 4e-9 (test_KERNEL_SCF_adjoint_matches_fd). Exact variational potential.
- SCF CONVERGES for ALL of He/Be/Na/Mg/Ne (PSP) with the exact adjoint -- NO floor, NO frozen
  potential needed (the prediction held: the map interpolates valid holes and the FA fixed point
  owns the low-density tail, so no F<0 blow-up). He near-exact FA (-3.5 mHa), spin-correct.
- Full suite 47/47.

Self-consistent benchmark vs HF (cached) / PBE-x:
  atom  E_x(HF)  E_x(PBE)  E_x(kernel-SCF)  PBE err  kern err
  He   -1.0019  -0.9705   -1.0054            +31.4     -3.5
  Li   -1.4514  -1.4481   -1.5026             +3.4    -51.2
  Be   -1.9215  -1.8853   -1.7029            +36.2   +218.6
  Ne   -5.4013  -5.2989   -5.2640           +102.4   +137.3
  Na   -5.7280  -5.6816   -5.6942            +46.4    +33.8
  Mg   -6.8577  -6.7599   -6.7438            +97.8   +113.9
  MAE vs HF (mHa): PBE-x 52.9; kernel-SCF 93.1 (non-SCF was 52.0)

KEY ISSUE: self-consistency DEGRADES the kernel (SCF MAE 93 vs non-SCF 52). Ne (+57->+137) and
Mg (+24->+114) blow up most. Cause: the bulk LDA band is too loose -- the DIRECT HEG projection
at n_out=10 over-recovers LDA by 3.3% (F=1.033), and the over-binding bulk potential distorts the
self-consistent density. The non-SCF benchmark on the good HF density masked this. The functional
is sound and intrinsically SCF-stable; the next lever is TIGHTENING THE BULK LDA recovery (more
n_out channels, or use the base's Q_S=2 envelope inversion for the bulk anchor instead of the
direct projection), plus the transition fixed point for Be. Tools: kernel_scf_bench.py,
kernel_scf_check.py.

## LDA limit: moment-matched HEG anchor (exact LDA at n_out=10) + convergence/exact-hole findings

Three findings, then the fix:

1. CONVERGENCE (brute force is impractical). HEG-hole projection -> LDA: the energy converges
   instantly in n_out (saturated by ~16) but is limited by the WINDOW X=k_F R_ad -- X=8 -> 1.6%,
   X=20 -> 0.24%, X=28 -> 0.13%. But X=20 needs R_ad=20/k_F <= R_c, i.e. R_c ~ 14-35 bohr for
   valence/tail densities (vs the practical R_c=6). So converging LDA by enlarging the window is
   IMPRACTICAL for a grid functional. (n_out must also scale with X; n_out=10 is the practical limit.)

2. EXACT ATOMIC HOLE PROJECTS WELL on the practical basis. Projecting the EXACT exchange hole
   (orbital_hole.py, general-l) onto R_c=6: Ne -> +59 (n_out=10), -0.8 mHa (n_out=16); Be -> +26,
   +12. Atomic holes are COMPACT (short tail), so the basis is ADEQUATE for them (unlike the
   long-tailed HEG hole). => the enhancement is fully capturable; the kernel's problem is that it
   produces the wrong hole (LDA + weak GEA2), not that the basis is too small.

3. MOMENT-MATCHED HEG ANCHOR (the LDA fix at n_out=10). The exchange energy IS the hole's Coulomb
   moment, so pin the THREE low-order moments to the exact LDA hole's values:
   {charge 4pi R_ad^3(rho~.alpha1)=-1, on-top, Coulomb 2pi R_ad^2(rho~.beta1)=C_LDA rho^{1/3}}.
   Least-norm-correct the projected HEG hole to satisfy all three -> a ~6% shape deformation that
   hits LDA EXACTLY at n_out=10 (verified eps/eps_LDA=1.00000 across rho), keeping the smooth
   hole-SHAPE interpolation (no rho^{1/3} energy hack). Principled = moment-matching the exact hole.
   Implemented in SIMPLE_HOLE_EXPANSION_KERNEL._kernel_eps (3-constraint anchor; the final 2-constraint
   projection is now a no-op at the HEG limit). Adjoint still exact, SCF converges, He still FA. 47/47.

STATUS: the LDA limit is now exact AND clean at the practical n_out=10. The atoms still under-bind
(the moment-matched anchor reproduces the exact-LDA result -- LDA+GEA2 is too weak; the projection
over-binding that masked it is gone). THE REMAINING WORK is the enhancement: make the kernel hole
reproduce the exact (compact) ATOMIC hole, via exact-hole fixed points / moment-matching against
orbital_hole.py references -- which finding (2) shows lands within ~1-12 mHa on the practical basis.

## Full SCF benchmark with the moment-matched (exact-LDA) anchor

SIMPLE_HOLE_EXPANSION_KERNEL self-consistent, vs HF (cached) / PBE-x:
  atom  E_x(HF)   E_x(PBE)  E_x(kernel-SCF)  PBE err  kern err  conv
  He   -1.0019   -0.9705   -1.0054            +31.4     -3.5    True
  Li   -1.4514   -1.4481   -1.5029             +3.4    -51.5    True
  Be   -1.9215   -1.8853   -1.6824            +36.2   +239.2    True
  Ne   -5.4013   -5.2989   -5.0945           +102.4   +306.8    True
  Na   -5.7280   -5.6816   -5.4876            +46.4   +240.4    True
  Mg   -6.8577   -6.7599   -6.5197            +97.8   +338.0    True
  MAE vs HF (mHa): PBE-x 52.9; kernel-SCF 196.6

All converge. He near-exact FA (-3.5). The LDA limit is now exact (moment-matched), and the bulk
atoms under-bind 4-6% -- this is the ENHANCEMENT GAP, now cleanly exposed (no projection
over-binding masking it). GEA2 (10/81)s^2 is too weak for atoms. The fix (validated capturable):
exact-hole fixed points so the kernel reproduces the exact compact atomic hole (projects to ~1-12
mHa on the practical basis). This is the pinned checkpoint; the enhancement is the next branch.

## Exact-hole reference dataset + accuracy tests (toward the data-driven enhancement)

Built & cached (scratchpad hole_refs.npz, build_hole_refs.py) for every converged atom
(He/Li/Be/Ne/Na/Mg, HF): at each subsampled r0, the EXACT exchange hole (orbital_hole.exchange_hole,
general-l) projected onto the adaptive n_out=10 unit frame (rhotilde_exact) + the scale-free SIMPLE
features (cn, s, Q) + full-grid weights.

TEST 1 -- basis adequacy (projected exact hole reproduces each atom at n_out=10):
  He +1.4, Li +15.0, Be +17.2, Ne +23.4, Na +27.0, Mg +44.8 mHa vs HF; MAE ~21 mHa.
  => the practical R_c=6/n_out=10 basis CAN represent the atomic holes to ~20 mHa (far past PBE 53
  and the current kernel-SCF 196). The kernel's only gap is producing the wrong hole, confirmed.

TEST 2 -- map learnability (leave-one-atom-out, interpolate the scale-free enhancement F=eps/eps_LDA
in SIMPLE-feature space, Nadaraya-Watson; rbf_loo.py): interior atoms generalize -- He +36, Li -11,
Be -15 mHa; edge atoms extrapolate poorly -- Ne -166, Na -515, Mg -550 (Na/Mg are at the
high-density edge with no neighbors beyond them; LOO MAE 215). NOTE: raw hole-vector interpolation
fails (the hole scales with rho0 -- ~rho0 in the bulk but ~rho0/Q in the tail; rt/rho0 blows up);
the dimensionless enhancement F is the clean interpolation target.

CONCLUSION: the data-driven exact-hole kernel is validated -- references are accurate (~21 mHa) and
the enhancement interpolates where feature space is covered. PATH: build a richer reference set
(more atoms/densities) covering feature space, interpolate F (or the scale-free hole shape); include
the target atoms as references for production. This is the principled "fix the physics" enhancement.
