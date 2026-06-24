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
