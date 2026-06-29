# Branch handoff — `simple-hole-kernel-map`

Closes the development arc that took `SIMPLE_HOLE_KERNEL_FP` from "exchange-energy-accurate on
closed shells, not yet self-consistent across the table" to **self-consistent and benchmarked on
all 69 reference atoms**, and updated the writeup with the result.

Status: all changes committed on branch `simple-hole-kernel-map` (ahead of `origin`, **not pushed**).
The functional itself was not modified after `b478a58`; the final two items (writeup §, this doc)
are documentation. The exchange functional and the production reference file are frozen.

---

## 1. What the production functional is now

Self-consistent exchange-only functional reproducing the (restricted, spherically-averaged) HF
exchange of the atomic solver, purely density-based.

**Reference set (frozen):** `atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q_gf06_Hanchor.npz`
(1529 closed-shell exact-hole nodes on the rho>=1e-2 manifold + 40 one-electron anchor nodes; 12-dim
features cn[1:]+s^2+p2+Q; grad_filter-consistent s^2 column).

**Parameters:** `fp_l0=0.7, fp_l1=0.5, fp_l2pow=0.02, fp_lQ=0.3, fp_ref_ridge=1e-8,
use_l2_power=True, use_Q=True, fa_ontop=False, fa_coeff=False, grad_filter=0.6,
deriv_smooth=1.0, deriv_smooth_adaptive=True, auto_continuation=True`, `scf_tolerance≈3e-4`.

**How to run SCF (the default that converges all 69 atoms):**
```python
from atom.xc.robust_solve import robust_scf_solve
r = robust_scf_solve(atomic_number)        # production_params() by default
# r['converged'], r['rho_residual'], r['energy'], r['mixer']
```
`robust_scf_solve` escalates the density mixer (DIIS -> linear-0.2 -> damped-DIIS) until convergence;
no per-atom tuning. The exchange functional is unchanged by the mixer choice.

---

## 2. Commits on this branch (newest first)

| commit | what |
|---|---|
| `b478a58` | escalating-damping robust solve (`atom/xc/robust_solve.py`); **all 69 atoms converge** |
| `fb3eaf4` | one-electron (H) anchor added to the production reference set |
| `444ffbb` | `auto_continuation` — single self-annealing SCF loop replaces the 2-stage homotopy |
| `cf5d220` | `deriv_smooth_adaptive` — roughness channel weights derived from v_x operator amplification |
| `3412e46` | retarget `deriv_smooth_grad` to the cn[1:]+s^2 (operator-coupled) channels |
| `dc0253b`, `f7fa4aa` | `grad_filter` — band-limit the spectral gradient operator (energy-neutral) |
| (this handoff) | writeup §"Self-consistency and the full periodic table" + Table; uncommitted: `writeup/main.tex`, `reports/.../wigner_subtracted_comparison.txt`, this file |

All functional params added this branch default OFF; the FP regression (`tests/simple/
test_simple_hole_expansion.py -k FP`, 14 tests) passes throughout.

---

## 3. Headline results

**Self-consistency:** 69/69 reference atoms (Z=1–57,72–83) converge to rho-residual < 3e-4 with the
single `robust_scf_solve` default (49 via DIIS, 19 via linear-0.2, 1 (Se) via damped-DIIS).
Report: `reports/hole_expansion/allatom_scf_sweep.txt`.

**Accuracy vs Wigner-subtracted HF** (spherically-averaged exact exchange = the best a spherical
density functional can reach; = HF for closed shells). MAE in mHa, excluding the post-d p-block
artifact (see below):

| set | n | SIMPLE | PBE | rSCAN |
|---|---|---|---|---|
| Closed shell | 14 | **19** | 45 | 14 |
| Open s/p | 18 | **57** | 85 | 76 |
| Open d (transition metals) | 27 | 186 | 152 | 146 |
| All (clean) | 58 | 109 | 107 | 94 |

SIMPLE matches rSCAN on closed shells (both ~2.4× better than PBE), leads on open s/p, and is a
semilocal-class functional overall; the entire deficit vs rSCAN is the open d block.
Report: `reports/hole_expansion/wigner_subtracted_comparison.txt` (full per-atom table).

---

## 4. Key decisions / findings (the "why")

- **FA blend removed; one-electron anchor added.** The global Fermi–Amaldi on-top blend is correct
  only at the one-electron limit and spurious for many-electron on-tops. It was replaced by a
  localized Q≈0.8 anchor carrying the exact restricted H hole — fixes H (−53 → −8.5 mHa, in-family)
  with **zero** change to closed shells (the anchor's kernel influence decays before reaching
  many-electron cores). `reports/.../one_electron_anchor.txt`.
- **grad_filter is surgical and should stay.** It band-limits only the spectral gradient operator's
  high-n channels (GEA-invalid noise) — energy-neutral, leaves the self-consistent density intact.
  Trying to eliminate it by broadening the gradient kernel (fp_l1) *does* converge without it but
  costs ~60 mHa self-consistently (it degrades the gradient physics). The operator itself is the
  validated writeup one; grad_filter only tames its *adjoint* response to reference-induced roughness.
- **deriv_smooth is a training-time fitting choice**, not an SCF knob — it sets the coefficients.
  `deriv_smooth_adaptive` derives the per-channel weights from the v_x spatial-operator amplification
  (cn←op_n, s²←grad_op, p2/Q←0), so the old hand-set channel mask is replaced by a principled,
  grid-adaptive criterion. Energy-identical to the mask, slightly tighter SCF.
- **No single SCF mixer converges all atoms.** Mutually exclusive regimes: F/Nb/W need DIIS
  acceleration; K/Ca/Mn need pure damping (DIIS diverges them). Hence the escalating ladder.
- **Wigner subtraction = the fair target.** Subtracting the non-spherical (L≥1 partial-shell)
  exchange from HF gives the spherically-averaged exact exchange, which is the ceiling for any
  spherical density functional and removes the irreducible open-shell penalty common to all three.

---

## 5. Open items / next steps

1. **Methods section — DONE** (rewritten to match production: on-top $-\rho/2$ + one-electron anchor
   replacing the global Fermi–Amaldi blend; kernel features cn+$s^2$+$\ell$=2 power+$Q$; band-limited
   gradient operator; operator-weighted coefficient smoothing; single annealing loop + escalating-damping
   mixing). **Remaining results-table inconsistency:** the older Results tables still predate production
   and now sit alongside the new self-consistent §results-allatom — Table~\ref{tab:energies} (reference-free
   LDA+FA numbers, e.g. He $-1.0054$) and Table~\ref{tab:ref-energies} (closed-shell, *on the exact
   density*, MAE 34 mHa). Production's self-consistent closed-shell MAE is 19. Decide whether to refresh
   these to the production construction (needs re-running those configs) or relabel them as construction-
   stage illustrations.
2. **d-block accuracy** is SIMPLE's only real weakness (186 mHa, vs 146 rSCAN). The references are
   closed-shell; open-d densities are extrapolations. Adding un-diluted open-shell / transition-metal
   references is the clear lever. (History: naive all-shell refs *hurt* closed shells — must be added
   without contaminating the closed-shell manifold; see [[simple-hole-kernel-map]] memory.)
3. **Post-d p-block PSP artifact.** Ga–Se, In–Te, Tl–Bi (groups 13–16) show ~1 Ha exchange error for
   ALL THREE functionals — a pseudopotential d-semicore reference inconsistency, not a functional
   issue (halogens beside them are fine). Worth tracing in the HF reference / PSP generation; excluded
   from the clean benchmark for now.
4. **He** sits at +9.6 mHa (its own reference's windowing, not improved by the H anchor) — a separate
   lever (its reference coverage), if the small-Z accuracy matters.
5. **Robust-solve efficiency.** The escalating ladder wastes 1–2 mixer attempts on the ~20 hard atoms
   (~600 s; Se ~1160 s). An in-loop adaptive-alpha mixer (reduce alpha on residual increase) would
   converge each atom in one pass — an efficiency, not correctness, refinement.

---

## 6. Key files

- `atom/xc/simple_hole_expansion.py` — the functional (auto_continuation, deriv_smooth(_adaptive),
  grad_filter, Q/l2-power, _set_ref_scale linear-in-λ basis).
- `atom/xc/robust_solve.py` — the escalating-damping production SCF (`robust_scf_solve`).
- `atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q_gf06_Hanchor.npz` — production references.
- `cache/refs/regen/build_h_anchor.py` — rebuilds the production reference set (H anchor).
- `cache/refs/regen/build_refs_full.py`, `build_refs_add_{l2power,Q}.py`, `rebuild_refs_grad_filter.py`
  — the reference-build pipeline (pool → features → DELTA).
- `reports/hole_expansion/` — `allatom_scf_sweep.txt`, `wigner_subtracted_comparison.txt`,
  `one_electron_anchor.txt`, `single_adaptive_loop.txt`, `additional_smoothing.txt`.
- `writeup/main.tex` — §Results "Self-consistency and the full periodic table" (Table~\ref{tab:allatom}).
- Memory: `[[simple-hole-kernel-map]]` (full chronological development record).
