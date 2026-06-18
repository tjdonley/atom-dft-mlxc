# Phase D debrief — exchange-hole benchmark (overnight investigation)

> ⚠️ **SUPERSEDED — see `PRIOR_RESULTS_FOUND.md`.** The key pessimistic conclusion
> below ("4th-order GEA is intrinsically unstable; converges nowhere") was a
> **regression artifact of the spectral derivative operators** my curated branch
> inherited (LB94 commit 8b94557). With the prior LEGACY operators the GEA4 hole
> converges self-consistently and is near-exact for closed shells (He −1.0023, Be
> −1.9204). The open-shell "unphysical references" were the known spin-restricted-
> reference artifact, already fixed upstream by an unrestricted `E_x^U`. The
> calibration fix in this doc is correct and still applies. Read `PRIOR_RESULTS_FOUND.md`.

Date: 2026-06-18 (overnight, autonomous)

## TL;DR (read this first)

Three findings, in priority order:

1. **Calibration regression — FIXED.** The ported `SIMPLE_HOLE_GEA` over-bound badly
   (He −1.25, Be −2.40) because it made the deformation supply the *full* GEA target
   on top of the bare hole's own finite-window gradient slope `mu_bare`. Ported the old
   `mu_bare`-subtracting calibration (`_net_coeffs`). He/Be now near-exact.

2. **The 4th-order GEA is intrinsically unstable** — collapses/blows up self-
   consistently in the ORIGINAL code too (He −0.22, Be −5.13). The clean functional is
   the **2nd-order (s²-only) gradient-deformed hole**. → Make that the headline; drop
   4th-order from the headline (or present as future work).

3. **The 2nd-order hole has a self-consistency limit for DENSE atoms.** It converges
   (self-consistently) for He, Be, N, P; for Ne, Mg it fails to converge, and for Ar it
   converges to a spurious over-bound state. This resists mixing, two-stage relaxation,
   AND frozen amplitude — so it is intrinsic to the deformation potential, not the
   calibration. **Robust path: report POST-SCF energies** (GGA hole on the converged
   bare-hole density), with self-consistent v_x shown for He/Be where it converges.

Also: **rSCAN replaces r²SCAN** (r²SCAN never converges); **open-shell N, P** have
unreliable restricted HF/OEP; **Ar references are unreliable** (OEP errors, HF
converges to a wrong state −3.38, bare SCF is bistable −3.4/−5.4).

## Final benchmark table (fixed calibration v2; all atoms)

```
atom     OEP      HF      PBE    rSCAN   r2SCAN    bare   GGA(SC)    GEA       trust
 He   -1.0019 -1.0019  -0.9705 -0.9971 -0.9980* -1.0020  -1.0028  +35.7~   GGA-SC ok
 Be   -1.9208 -1.9215  -1.8853 -1.9105 -1.8982* -1.9881  -1.9494   -2.29   GGA-SC ok
 N    -1.6129 -1.6052  -2.5462 -2.6131 -2.6031  -2.7276  -2.6468   -3.85*  open-shell: HF/OEP bad
 Ne   -5.4223*-5.4012  -5.2989 -5.4110 -5.3937  -5.5788  -5.88*    -5.52   GGA-SC FAILS; post-SCF -5.4691
 Mg   -6.9715 -6.8577  -6.7599 -6.8878 -6.8671* -7.2086  -9.37*    -7.32~  GGA-SC FAILS; post-SCF -7.3337
 P    -1.0820 -1.0560  -1.6649 -1.7300*-1.7205  -1.7531  -1.6849   -2.27*  open-shell: HF/OEP bad
 Ar    err    -3.3773* -5.0318 -5.4515 -5.4410  -5.4049  -5.4589~  -4.64~  Ar refs unreliable (bistable)
```
*=flagged not converged (or wrong-state for Ar HF). ~=outer-loop note. GEA collapses.
Best CLEAN closed-shell comparisons (self-consistent GGA vs exact-x):
  He −1.0028 vs −1.0019 (excellent); Be −1.9494 vs −1.9208 (improves bare −1.9881).
Post-SCF GGA@bare (where SC fails): Ne −5.4691 (vs HF −5.4012, improves bare),
  Mg −7.3337 (over-corrects; bare −7.2086, HF −6.8577). [Ar post-SCF invalid — bad
  bare density from the bistable SCF; recompute on the benchmark's −5.40 density.]

## Reference + bare numbers (trustworthy, regenerated on clean branch)

| Atom | OEP | HF | PBE | rSCAN | r²SCAN | bare hole |
|------|-----|----|----|-------|--------|-----------|
| He | −1.0019 | −1.0019 | −0.9705 | −0.9971 | −0.9980 (no conv) | −1.0020 |
| Be | −1.9208 | −1.9215 | −1.8853 | −1.9105 | −1.8982 (no conv) | −1.9881 |

- **rSCAN converges everywhere; r²SCAN converges nowhere** (He, Be both fail). Per
  your instruction, swap the reference column in the writeup from r²SCAN → rSCAN.
- Bare hole He matches OEP/HF to 0.01%. Bare Be is ~3.5% over OEP (−1.9881 vs
  −1.9208); note old branch bare Be = −1.9705 (small spectral-operator difference
  between clean-branch and old-branch operators — worth a look, minor).

## The bug (deformed holes)

`atom/xc/simple_hole.py :: SIMPLE_HOLE_GEA`

- `_calibrate_response()` measures `rho_resp = dF_x/dc` at a **single HEG point**
  (uniform ρ=0.1), then `_amplitude` sets `c = (1/rho_resp)(A s² + B q² + C s²q)`
  with `A,B,C = 10/81, 146/2025, −73/405` (the exact GEA target).
- This makes the **deformation supply the *full* GEA enhancement**, on top of the
  bare hole, which **already has a substantial intrinsic finite-window gradient
  slope** `mu_bare`. Double-counting → over-enhancement → over-binding.
- Measured `mu_bare` (low-s regression on the atom's own density): He **0.137**
  (> target 0.123!), Be **1.01**. So for He the deformation should be essentially
  OFF; my port instead drove it hard (effective amplitude +3.24 s², vs the old
  calibrated A_gea = −3.031).

### Old (correct) calibration — `scripts/learned_hole/07_gea_base.py`, `20_gea4_calibrate.py`
```
mu_bare = Σ w s² (F_bare−1) / Σ w s⁴            # bare hole's own slope (low-s)
kappa   = Σ w s² (F_unit−F_bare) / Σ w s⁴       # unit-deformation response (low-s)
A_gea   = (mu_target − mu_bare) / kappa         # NET amplitude
```
For GEA4 this is the 3×3 system `K @ A = target − mu_bare` over monomials (s², q², s²q).
The target coefficients are the exact GEA constants → "parameter-free" (no fit to
energies), but `mu_bare`/`kappa`/`K` are calibrated on **atom grids offline** and the
amplitude vector is **baked into** the production functional (`gea4_base.npz`).

## Fix validated (on clean-branch machinery)

Applying the `mu_bare`-subtracting calibration (per-density low-s regression) to the
current code, then integrating on the converged bare density:

| Atom | bare | current (broken) | **GGA-fixed** | OEP | old GEA2 |
|------|------|------------------|---------------|-----|----------|
| He | −1.0020 | −1.1641 | **−1.0020** | −1.0019 | −1.0024 |
| Be | −1.9881 | −2.3945 | **−1.9327** | −1.9208 | −1.9106 |

→ near-exact He, good Be. Matches the old GEA2 (s²-only) behavior.

## Old-code self-consistent results (authoritative, regenerated tonight in worktree)

- `07` GEA2 (s²-only, properly calibrated): **He −1.0024, Be −1.9106** (vs OEP
  −1.0019 / −1.9208). Clean, improves Be over bare.
- `20` GEA4 *non-self-consistent*: blows up (He −2.58, Be −4.46), huge amplitudes
  A=[4.89, 0.18, −16.2]. The 4th-order Laplacian term is only meaningful under the
  self-consistent frozen loop.
- `21` GEA4 self-consistent (frozen two-stage): **He −0.2228 (collapsed!), Be −5.1269
  (blown up, outer not converged)**. The 4th-order Laplacian (q) term is unstable even
  in the original code — the self-consistent GEA4 does NOT give a sensible energy.

## ⭐ KEY CONCLUSION: headline = 2nd-order (s²-only) gradient-deformed hole, NOT 4th-order

The authoritative old-code numbers settle it:

| Hole | He | Be | self-consistent? |
|------|----|----|------------------|
| bare (HEG+FA) | −1.0020 | −1.9705 | yes |
| **GEA2 (s²-only)** | **−1.0024** | **−1.9266** | **yes, stable** |
| GEA4 (s²+q, 4th-order) | −0.2228 | −5.1269 | **no — collapses/blows up** |
| OEP / HF (exact-x) | −1.0019 | −1.9208 | — |

The 4th-order Laplacian term is fundamentally unstable self-consistently (old code AND
new). The clean, near-exact, stable result is the **2nd-order s²-only gradient
deformation**. My earlier GEA collapse to −0.246 was NOT primarily my bug — the old
GEA4 collapses to −0.2228 the same way.

**Recommendation for the writeup:** make the headline the parameter-free *second-order*
(s²) gradient-deformed hole. Drop the Laplacian/4th-order term from the headline
functional (or present GEA4 only as an attempted extension that needs future
stabilization). This is a substantive change to the Theory/Results framing — discuss
in the morning.

## Fix implemented + validated (clean branch `simple_hole.py`)

Replaced the single-point HEG response with a per-density **net** calibration
(`_net_coeffs`): solve `K A = target − mu_bare` over the low-s regime, exactly the
old steps 7/20 scheme. Forward and self-consistent checks:

| Atom | bare | broken (old port) | **fixed GGA forward** | **fixed GGA SCF** | old GEA2 | OEP |
|------|------|-------------------|----------------------|-------------------|----------|-----|
| He | −1.0020 | −1.1641 | −1.0020 | **−1.0029 (conv)** | −1.0024 | −1.0019 |
| Be | −1.9881 | −2.3945 | −1.9317 | **−1.9290 (not conv)** | −1.9266 | −1.9208 |

He converges directly and is near-exact; Be's energy is right but its direct SCF needs
the two-stage loop to fully converge (stiffer deformation potential). Fixed-code GEA
forward is now tame (He −1.033, Be −1.978) but the SC GEA4 is still expected to
collapse (Laplacian stiffness) — consistent with the old code.

## Open design decision for the writeup (morning)

1. **Which hole is the headline?** GEA2 (s²-only) is clean and near-exact (He) /
   good (Be) and self-consistently stable. GEA4 (the "to-4th-order" parameter-free
   hole) needs the frozen two-stage loop and is stiff; need step-21 numbers to judge
   whether it's better than GEA2 or just stickier.
2. **Calibration design:** (a) bake in a single offline-calibrated amplitude vector
   (faithful to old code, reproduces numbers, but "parameter-free" needs careful
   wording since mu_bare is atom-grid-calibrated); or (b) per-density self-calibration
   inside the functional (subtract the functional's own finite-window mu_bare at each
   density — fully self-contained, deterministic, no stored data). I lean (b) for the
   writeup's parameter-free story; needs a stability check inside SCF.
3. The writeup currently describes "one-time HEG calibration of the envelope
   response (ρ_resp = ∂F_x/∂c)". This is **insufficient** — it misses `mu_bare`
   (a finite-window effect, zero at true HEG). The Theory/Appendix calibration text
   must be updated to the `mu_bare`-subtracting scheme.

## Two-stage SCF loop (separate, working)

The frozen-correction two-stage loop is implemented and works for the stable channel:
- `_bare_v_x` + `external_v` frozen branch reproduce the gauged-full potential to 3.5e-15.
- Outer under-relaxation (mix density) needed; β≈0.12–0.15 stable for GGA.
- GEA (Laplacian channel) is stiffer; inner SCF eventually destabilizes — but with the
  calibration fixed the deformation is far smaller, so this should ease. Re-test after fix.

## Knock-on: Phase C figure must be regenerated

The calibration change shrinks the deformation amplitude, so the re-summed atom F_x
points in `figures/fx_enhancement.pdf` (Fig. 1, via `atom_trajectories.py` →
`atom_sqfx.npz`) will move closer to 1. The analytic GEA curves (the target
1+10/81 s²+…) are unchanged. With the headline shifting to 2nd-order, both the figure
and its caption need rework anyway — fold into the morning writeup discussion.
Two-stage GGA (fixed) self-consistent: He −1.0028 (1 outer iter), Be −1.9494 (6 iters).

## Unit-test impact (1 expected failure — design semantics)

`tests/simple/test_simple_hole.py`: 3 pass, 1 fails —
`test_gea_hole_reduces_to_bare_without_deformation`. The test zeroes the module
globals `_GEA_A/B/C` (target=0) and expects the deformation to vanish → bare hole.
Under the NET calibration, target=0 → `A = K⁻¹(−mu_bare) ≠ 0`: the deformation now
*cancels* the bare hole's spurious finite-window gradient slope instead of vanishing.
This is the intentional fix, not a bug — but it changes the "zeroed coefficients ⇒
bare" invariant the test encodes. Resolve alongside the calibration-design decision:
- If we keep net calibration: rewrite the test to assert the deformation is OFF when
  `target == mu_bare` (or that `_net_coeffs` returns ~0 there), and add a test that
  target=0 flattens the net F_x slope to ~0.
- Optionally special-case `all(target==0) ⇒ A=0` to preserve the literal invariant.
The adjoint-consistency and HEG-limit tests still pass.

## Consolidated results so far (fixed calibration v2; Mg/P/Ar still running)

```
atom    OEP       HF       PBE     rSCAN    r2SCAN     bare    GGA(head)   GEA
 He  -1.0019  -1.0019  -0.9705  -0.9971  -0.9980*  -1.0020   -1.0028    +35.7(collapse)
 Be  -1.9208  -1.9215  -1.8853  -1.9105  -1.8982*  -1.9881   -1.9494    -2.29(collapse)
 N   -1.6129  -1.6052  -2.5462  -2.6131  -2.6031   -2.7276   -2.6468    -3.85(collapse)   <- open-shell: HF/OEP unreliable
 Ne  -5.4223* -5.4012  -5.2989  -5.4110  -5.3937   -5.5788    (run)      (collapse)
```
* = that SCF flagged not-converged. GEA always collapses (4th-order instability).

**Ne AND Mg GGA FAILED to converge** (two-stage inner SCF dies on outer iter 1;
reported −5.88 / −9.37 over-bind and are unreliable). The two-stage worked for
He/Be/N but fails for the heavier closed-shell atoms. **Direct single-loop SCF also
fails** for Ne/Mg at every mixing (amix=0.4/0.2/0.1 all conv=False, energies thrash
−2…−8.9). So it is NOT a mixing/relaxation issue.

Leading hypothesis: the **per-density self-calibration adds feedback** — `A` is
re-derived from the changing density every SCF step, which oscillates for the more
structured Ne/Mg densities. The old code avoided this with a FIXED baked-in amplitude.
Added a `frozen_coeff` option (hold `A` fixed, calibrated once on the bare density) and
am testing whether it converges Ne/Mg. **[frozen-A test result pending — /tmp/frozen_test.log]**
If frozen-A converges, that decides the calibration design (baked-in, not per-density)
and likely fixes the closed-shell set.

P GGA converged fine (−1.6849, 17 outer iters) — open-shell but the two-stage held.
P rSCAN did NOT converge (rSCAN isn't universal for open shells), though it still beats
r²SCAN (which fails everywhere).

### Frozen-A test result: per-density feedback is NOT the cause

Holding `A` fixed (calibrated once on the bare density) STILL fails to converge Ne/Mg
(conv=False, energies thrash −0.95…−7.8). So the self-consistency failure is intrinsic
to the GGA deformation **potential** for these denser atoms, not the calibration
feedback. Note Be (A=−5.2) converges via two-stage while Ne (A=−1.0) and Mg (A=−2.6)
do not — so it is the potential structure, not the amplitude size. This is a genuine
self-consistency limitation needing deeper work (deformation-channel adjoint /
gauge-tail behavior for many-electron densities).

### Practical path: post-SCF GGA@bare gives a complete, sensible table

Evaluating the GGA hole energy on the converged BARE-hole density (the forward energy
is well-behaved everywhere; only the self-consistent potential is unstable):

```
atom    bare    GGA@bare   reference        Δ(GGA−ref)
 He  -1.0020   -1.0020   -1.0019 OEP/HF      +0.000   (exact)
 Be  -1.9881   -1.9317   -1.9208 OEP         -0.011   improves bare (-0.067)
 Ne  -5.5788   -5.4691   -5.4012 HF          -0.068   improves bare (-0.178)
 Mg  -7.2086   -7.3337   -6.8577 HF          -0.476   WORSENS (bare -0.351) -- over-corrects
 Ar  -3.4912   -3.4100   (HF pending)         ?       improves vs bare
```
GGA@bare helps He/Be/Ne, over-corrects Mg. Honest, mixed.

### Convergence summary (headline GGA hole, fixed calibration)

| atom | self-consistent? | post-SCF GGA@bare |
|------|------------------|-------------------|
| He | ✓ (−1.0028) | −1.0020 |
| Be | ✓ (−1.9494, two-stage) | −1.9317 |
| N (open) | ✓ (−2.6468) | — |
| Ne | ✗ (all methods fail) | −5.4691 |
| Mg | ✗ | −7.3337 |
| P (open) | ✓ (−1.6849) | — |
| Ar | ✗ (likely) | −3.4100 |

**Decision for the morning:** the self-consistent GGA hole converges for light/diffuse
atoms but NOT for the denser closed-shell atoms (Ne, Mg, Ar). Options:
1. Present the hole as a **post-SCF** functional (energy on the bare-hole density),
   with self-consistency demonstrated where it converges (He, Be). Honest and complete.
2. Invest in stabilizing the deformation potential for dense atoms (deeper; the
   two-stage helped Be/N but not Ne/Mg — needs a better scheme for the l=1 adjoint
   tail at high electron count).
My lean: report post-SCF energies for the full closed-shell set + self-consistent v_x
vs OEP for He/Be (where it converges), and state the dense-atom self-consistency
limitation plainly as a scope item.

Headline (GGA, 2nd-order) vs best exact-x reference, closed-shell:
- He: −1.0028 vs OEP/HF −1.0019  → +0.0009 (excellent)
- Be: −1.9494 vs OEP −1.9208     → −0.029 (bare was −0.067; GGA improves)
- Ne: pending vs HF −5.4012 (OEP −5.42 not fully conv; HF is the reliable ref)
r²SCAN never converges; rSCAN always does → use rSCAN.

## Open-shell atoms (N, P) — reference looks unphysical (spin treatment)

N OEP came out **−1.6129**, *smaller in magnitude than Be's −1.92* despite N having
more valence electrons (5 vs 2) — physically wrong. Almost certainly a spin-restricted
treatment of an open-shell atom (N = 2s²2p³, 3 unpaired). The bare hole N = −2.7276 is
comparably affected. This is the plan's known scope item ("spin-restricted HF
reference + unrestricted-exact open-shell correction"). **Recommendation:** build the
headline energy table on the CLOSED-SHELL set (He, Be, Ne, Mg, Ar) where the restricted
reference is exact; treat N, P separately with the open-shell correction (or drop them
from the headline table). Verify each closed-shell atom's OEP magnitude is monotonic /
sensible before trusting it.

## Status of artifacts

- Benchmark stopped mid-sweep; pickle has trustworthy He/Be references + bare hole.
  GGA/GEA entries cleared (were from the broken calibration).
- Old-branch worktree at `/tmp/lb94-wt` (simple-hole-LB94) with regenerated
  `data/oep_{He,Be}.npz`, `gea_base.npz`, `gea4_base.npz`.
- Next: implement calibration fix, re-validate He/Be, re-run full benchmark.
