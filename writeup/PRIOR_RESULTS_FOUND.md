# Prior results located — and why the current branch disagrees

Date: 2026-06-18 (deep dive per PI request). **This supersedes the pessimistic
conclusions in PHASE_D_DEBRIEF.md, which were artifacts of a regression.**

## Where the prior results live

Branch **`simple-hole-SR`** (commit 7443977), the branch *immediately before*
`simple-hole-LB94` (LB94 = SR + the single commit 8b94557 "Spectral fixed-window
gradient & Laplacian … wired into learned hole"). The authoritative record is
`scripts/learned_hole/README.md` on SR (a ~680-line lab notebook) plus steps 20/21/24/26.

The full-GEA hole results you remembered are real and documented there.

## The canonical prior numbers

### GEA4 self-consistent (step 21, frozen-correction loop, LEGACY operators)
ALL atoms converge (stable via the frozen/two-stage loop); closed-shell near-exact:

| atom | E_OEP | E_bare | E_GEA2(SC) | E_GEA4(SC) |
|------|-------|--------|------------|------------|
| He | −1.0019 | −1.0020 | −1.0021 | **−1.0023** |
| Be | −1.9208 | −1.9705 | −1.9243 | **−1.9204** ✓ |
| C  | −0.9640 | −1.2842 | −1.7086 | −1.7078 |
| O  | −2.5729 | −3.1492 | −3.8197 | −3.8288 |
| Mg | −6.9715 | −7.3308 | −7.2140 | −7.2702 |

### Fair benchmark (step 26, H–Ar, spin-resolved, vs TRUE unrestricted exact exchange)
- **MAE(E_x), Ha: SCAN 0.075 · PBE 0.079 · GEA4 0.118 · bare 0.154.** GEA4 competitive
  with PBE, clearly improves the bare hole.
- **Closed shells (He,Be,Ne,Mg,Ar): GEA4 1.00–1.06× exact — best-in-class, beats PBE**
  (PBE under-binds ~2%).
- **Open shells: PBE best, GEA4 next, bare worst** ("slightly worse than PBE").
- Empirical limit-safe fit (steps 28/29): reaches **PBE parity** (MAE 4.97% vs 4.90%),
  beats PBE on 10/18 atoms (all closed + half-filled), ~2–3% behind on early-p.

→ Matches your recollection exactly: competitive with PBE, near-exact/best on closed
shells, slightly worse than PBE on open shells.

### The spin convention you remembered (step 24)
The HF/OEP reference was **spin-restricted** (`hf.py` used spin-averaged occupations
`f_i = ½(n↑+n↓)`), which under-counts same-spin exchange for open shells — wrong by up
to 2× (H: restricted −0.128 vs true −0.256). Fix: the **unrestricted** reference
`E_x^U = ½(E_x^R[2ρ↑] + E_x^R[2ρ↓])`. Against E_x^U the open-shell "over-binding" drops
from 22–33% to ~10%; H becomes exact, He stays exact. Spin-scaling the *model* was
tested and refuted (model is spin-blind at fixed total ρ; it already embeds the correct
N-dependent SIE: H→−E_H, He→−½E_H).

## Why my current (curated `simple-writeup`) branch disagrees — the regressions

1. **Derivative operators (THE big one).** The prior good results used the **legacy**
   `build_gradient_operator/build_laplacian_operator` (steps ≤26) and later the
   machine-accurate **local-polynomial stencil** `stencil_derivatives.py` (steps 31/32).
   My curated branch uses the **spectral** `build_spectral_gradient_operator/
   build_spectral_laplacian_operator` — the LATE LB94 addition (8b94557), whose Laplacian
   I already had to rewrite once (the ~300× error). The README step 30 explicitly found
   the windowed/global gradient *degrades* and the stencil route is the fix.
   **Evidence:** step-21 GEA4 on LB94 (spectral) → He −0.2228 / Be −5.1269 (collapse);
   on SR (legacy) → He −1.0023 / Be −1.9204 (near-exact). Same code except the operators.
   → My "4th-order GEA is intrinsically unstable" conclusion was a SPECTRAL-OPERATOR
   ARTIFACT, not a property of the functional.

2. **Spin-restricted reference.** My `phase_d_benchmark.py` compares against restricted
   OEP/HF (no E_x^U correction) → the open-shell references (N −1.61, P −1.06) are the
   known-wrong spin-restricted ones. The prior work fixed this with the unrestricted
   reference. → My "open-shell references look unphysical" was the same artifact the prior
   work already diagnosed and corrected.

3. **Calibration.** I independently re-derived and fixed the `mu_bare`-subtracting
   net calibration (now in `_net_coeffs`); this matches the prior step-07/20 scheme. Good
   — but it was applied on top of the broken spectral operators, so the SCF still failed.

## Tail / SCF damping check (PI question)

Two distinct dampings existed in the prior code:
1. **Potential −1/r tail damping** (`_apply_gauge`, `_RHO_DAMP=1e-8·ρmax`, gauge-fix +
   damp v_x→eps in the tail): **PRESENT and identical** in the curated branch (base
   `SIMPLE_HOLE`, lines 148–170). The "LB94" branch name refers to this −1/r-tail goal;
   there is NO separate additive van-Leeuwen term in the hole code.
2. **Deformation-amplitude low-density damping** (`rho_damp` → `c → f_damp(ρ)·c`,
   `f_damp=ρ²/(ρ²+rho_damp²)`, in `simple_hole_learned.py`): **WAS MISSING** from my
   `SIMPLE_HOLE_GEA`. Its purpose (per the learned-hole comment): *"tames the
   gradient-coupling dc/dg ~ g/ρ² that blows up at low density."* This is exactly the
   regime where the spectral operators misbehave — so the PI's hypothesis that it could
   offset the spectral-operator instability is well-motivated.
   → **Now ported** into `SIMPLE_HOLE_GEA` (param `rho_damp`, default 0; `_damp` method;
   applied in `_amplitude` with the product-rule derivative for the adjoint). [Testing
   whether it stabilizes Ne/Mg under the spectral operators — see /tmp/damp_test.log.]

Note: the prior near-exact GEA4 (step 21) achieved stability via the FROZEN-CORRECTION
loop with `rho_damp` OFF; the amplitude damping was a separate earlier stabilizer
(step 12: "rho_damp=0.1 gets He/Be/C/Mg to converge; Mg failed undamped"). So both
mechanisms exist; for the spectral-operator branch, amplitude damping is the natural
thing to try since the spectral failure is low-density.

## Systematic comparison plan (for "we pin down the issues")

1. **[in progress] Reproduce SR step-21 GEA4 (legacy ops)** in a worktree to confirm
   He −1.0023 / Be −1.9204 — pins the spectral operators as the regression.
2. **Swap the curated branch's derivative operators** from spectral → the stencil
   route (`stencil_derivatives.py`, steps 31/32 — machine-accurate, r_c-independent) and
   re-run the GEA4 frozen-loop benchmark; expect the prior numbers to return.
3. **Add the unrestricted-exchange reference** (`E_x^U`) to the benchmark for open shells
   (port the step-24 reconstruction), so open-shell comparisons are fair.
4. Re-decide the writeup framing with correct numbers: the headline IS the GEA4 hole
   (it works), best-in-class on closed shells, PBE-competitive overall.

## Artifacts
- SR worktree: `/tmp/sr-wt` (legacy ops); LB94 worktree: `/tmp/lb94-wt` (spectral ops).
- Prior derivative routes: legacy `descriptors/simple/derivatives.py:build_gradient_operator`;
  stencil `descriptors/simple/stencil_derivatives.py` (on SR/LB94, NOT yet on simple-writeup).
- Authoritative notebook: `scripts/learned_hole/README.md` on `simple-hole-SR`.
