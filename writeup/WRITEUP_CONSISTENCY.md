# Writeup ⇄ code/results consistency ledger

Purpose: track exactly what the writeup CLAIMS vs what is VERIFIED, so the writeup and
the shipped code/results stay in lock-step. Update this whenever either side changes.

Status date: 2026-06-18. **We are in exploration mode — the writeup is NOT yet
consistent with verified results. Do not treat the writeup's hole-functional results
as final until the items below are resolved.**

## Git state (code repo `atom-dft-SIMPLE`)
- `simple-writeup` @ 279f0ab — CLEAN baseline; the LaTeX is written against this code.
  Uses the SPECTRAL derivative operators. NOTHING exploratory committed here.
- `simple-hole-phaseD` @ (off 279f0ab) — exploration WIP: net calibration fix, two-stage,
  `frozen_coeff`, `rho_damp`. **Re-merge the validated subset into simple-writeup only
  once we conclude** (operator choice + calibration design).
- `simple-hole-SR` @ 7443977 — prior work, LEGACY operators, near-exact GEA4 (reference).
- `simple-hole-LB94` @ 8b94557 — SR + spectral operators (the regression commit).
- Worktrees: `/tmp/sr-wt` (legacy), `/tmp/lb94-wt` (spectral). Throwaway; remove later.

The writeup dir (`SIMPLE-Xhole-writeup/`) is NOT under git. Consider `git init` here so
LaTeX + figure-scripts are versioned alongside the conclusion (open question for PI).

## What the writeup currently CLAIMS (simple-writeup state) — and consistency

| Writeup claim / content | Verified? | Action needed |
|---|---|---|
| Headline = parameter-free GEA **4th-order** hole | ⚠️ partial | 4th-order DOES work with legacy ops (He −1.0023, Be −1.9204) but COLLAPSES with the spectral ops the branch currently uses. Resolve operators before finalizing. |
| Calibration = "one-time HEG calibration ρ_resp=∂F_x/∂c" (Theory/App.) | ❌ WRONG | Insufficient — misses the bare hole's finite-window slope `mu_bare`. Must describe the net `K A = target − mu_bare` calibration. |
| Fig. 1 `fx_enhancement.pdf` (atom F_x overlay) | ❌ stale | Atom points come from `atom_trajectories.py` using the (buggy/spectral) deformation. Regenerate after the operator+calibration decision. |
| `\PH{}` placeholders: He HOMO eig, Table energies, `vx` figure | ❌ open | Fill from the FINAL verified benchmark (closed-shell set; unrestricted ref for open shells). |
| r²SCAN as the meta-GGA reference column | ❌ change | r²SCAN never converges; use rSCAN (and note open-shell rSCAN caveats). |
| Open-shell comparison vs (restricted) OEP/HF | ❌ change | Reference was spin-restricted; use unrestricted E_x^U (prior step-24 fix). |
| Feature sections (SIMPLE defs, scale, vacuum, operators) Figs/derivations | ✅ believed OK | Phase C features were validated; unaffected by the hole-operator issue. Re-confirm scale/vacuum figs unchanged. |
| Two-stage / frozen-correction dual loop (App.) | ✅ consistent | Implemented and verified (3.5e-15); matches App. text. |

## VERIFIED results (to flow into the writeup once operators are settled)
- Bare hole closed-shell: He −1.0020 (=OEP), Be −1.9881, Ne −5.5788 (refs sensible).
- GEA, LEGACY ops (prior, reproduced): near-exact closed shells; PBE-competitive overall;
  open shells slightly worse than PBE (vs UNRESTRICTED exact-x). See PRIOR_RESULTS_FOUND.md.
- GEA, SPECTRAL ops (current branch): collapses for dense atoms — REGRESSION, not physics.

## Open decisions blocking writeup finalization
1. **Operators:** spectral (current) vs legacy vs machine-accurate stencil (`stencil_derivatives.py`).
   Likely swap to stencil. [Testing if `rho_damp` rescues spectral — /tmp/damp_test.log.]
2. **Calibration design:** per-density `_net_coeffs` vs baked-in fixed amplitude.
3. **Headline:** 4th-order (works with right ops) vs 2nd-order. Prior work used 4th (GEA4).
4. **Reference set:** closed-shell (clean) + unrestricted E_x^U for open shell; rSCAN not r²SCAN.

## Companion exploration notes (not writeup content)
- `PRIOR_RESULTS_FOUND.md` — where the prior results live + the regression analysis (authoritative).
- `PHASE_D_DEBRIEF.md` — overnight investigation; SUPERSEDED conclusions (banner at top).
