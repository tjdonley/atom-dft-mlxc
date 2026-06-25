# Kernel exchange-hole functional: exact-hole fixed points + benchmark

Branch `simple-hole-kernel-map`. Successor work to `phaseK_report.md` (the kernel construction).
This phase added a data-driven enhancement from exact atomic-hole references and benchmarked it.

## What was built

- **HF reference fleet** (`scratchpad/hf_fleet.py` → `refs/hf_Z*.npz`): HF/PSP solves over the PSP
  set (Z=1–57, 72–83), caching density, energies, and the **exact orbitals** (for hole
  reconstruction). Idempotent, with a domain/mixing/iteration retry ladder.
- **PBE + rSCAN baselines** (`baselines.py` → `refs/base_Z*.npz`): exchange energies for the same
  set (69/69 converged).
- **Enriched exact-hole references** (`build_refs_from_cache.py` → `hole_refs.npz`): 13 **closed-shell**
  atoms (He/Be/Ne/Mg/Ar/Ca/Zn/Kr/Sr/Cd/Xe/Ba/Hg), each the moment-matched exact hole reproducing
  HF exchange to **≤2.7 mHa** (reusing the fleet's cached orbitals — no re-solving). Exact-hole
  references are restricted to closed shells because `orbital_hole.exchange_hole` uses a restricted
  (spin-summed) 1-RDM.
- **Kernel integration** (`atom/xc/simple_hole_expansion.py`, `SIMPLE_HOLE_EXPANSION_KERNEL`):
  the enhancement beyond LDA+GEA is a **dimensionless residual `dF(features)`** interpolated
  (Nadaraya–Watson with a HEG background weight `w0`) over the reference nodes
  (`atom/xc/data/kernel_fixed_points.npz`, 428 bulk nodes), applied through the GEA deformation
  mode with a Lieb–Oxford cap on the total. Features: scale-free `cn[1:]`, bounded `s`, charge `Q`.

## Key construction lesson (a bug, then the fix)

The first attempt interpolated the **ρ-scaled hole-coefficient residual** `δ = (rt−heg−gea)/ρ`.
It **over-binds catastrophically** (Ne −12.4 Ha vs −5.4 even non-SCF) — the *exact* failure mode
recorded in memory: the same scale-free shape occurs at different densities, and the linear-in-ρ
re-scaling amplifies the energy (which scales as ρ^{1/3}). **Fix:** interpolate the
**dimensionless enhancement `dF = F_exact − 1 − (10/81)s²`** (scale-free, the validated target)
and realize it through the GEA mode (`χ_total = (GEA2 s² + dF)/R`, LO-capped). This is
scale-correct, and the LO cap on the total bounds runaway. All limits preserved by construction
(`dF→0` far from nodes): **all 10 PHASE-KERNEL tests stay green** (LDA, GEA-slope 10/81, FA,
adjoint==FD, SCF converges).

## Benchmark — the honest metric is NON-SCF at a common (HF) density

Comparing *self-consistent* E_x across HF (exchange-only), PBE/rSCAN (with correlation), and the
kernel is confounded (different densities) and is further corrupted by **SCF density drift** (see
below). Evaluating every functional's exchange at the **same HF density** isolates functional
quality:

**Non-SCF exchange MAE vs HF at the HF density (mHa), production `w0=4`, ell=1.1, 428 nodes:**

| group | n | LDA+GEA (no fixed pts) | kernel + fixed pts |
|---|---|---|---|
| **closed-shell (in-sample)** | 13 | 218 | **106** |
| open-shell main-group | 14 | 138 | 173 |
| d-block + open (non-ref) | 19 | 186 | 301 (max 1047) |
| all | 32 | 199 | 222 |

- **In-sample (closed-shell): the references halve the error** (218→106). Ne −52, Ar +33,
  Ca −0.3, Kr +35, Sr +12, Ba +17, Xe +58 are all tens of mHa.
- **Out-of-sample (open-shell, d-block): the references HURT** (186→301). Closed-shell-trained
  `dF` over-enhances open-shell/d exchange (O −272→−433) — open-shell exchange differs (spin),
  and no open-shell references exist (restricted-hole limitation).
- **Over the full set the references are net-harmful** (199→222), and a `w0` sweep is
  **monotonic** (larger `w0` → closer to LDA+GEA): *no* hyperparameter makes the closed-shell-only
  references help the general set. This is a **transferability limit, not a tuning failure**. The
  enhancement is a validated, specialized improvement for the **closed-shell class it is built
  from**; `w0=4` is kept as the documented research-checkpoint compromise (strong closed-shell
  gain, references dormant elsewhere as `w0→∞`).

## Open problems (the real next steps)

1. **Open-shell non-transferability.** The enhancement is trained on closed shells and
   over-applies to open shells. Fixing this needs **spin-resolved exact-hole references**
   (extend `orbital_hole` to per-spin 1-RDM) so B/C/N/O/F… can be references. This is the single
   biggest accuracy lever.
2. **SCF density drift.** The kernel's exchange-only SCF over-contracts diffuse-valence/heavy
   atoms: e.g. Ar is correct non-SCF at the HF density (−3.42 vs HF −3.38) but the SCF density
   drifts (−5.14). The functional is right *at* the reference densities but its potential drives
   the self-consistent density away. Needs a stabilization that does not rely on the retired
   LB94 floor (candidate: damp/condition the residual's contribution to the potential, or
   constrain dF to the trust region of the references).
3. **d¹⁰ over-shoot** (Zn/Cd/Hg) and the **odd-electron FA spin factor** (H is 2× off) — both
   point at the same spin/shell-resolution gap as (1).

## Reproduce

- References: `build_refs_from_cache.py` → `hole_refs.npz`; bundle: `make_fixed_points.py` →
  `atom/xc/data/kernel_fixed_points.npz`.
- Clean benchmark: `nonscf_bench.py` (non-SCF at HF density, by group).
- Tests: `pytest tests/simple/test_simple_hole_expansion.py -k KERNEL` (10 green).
