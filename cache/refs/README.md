# SIMPLE exchange-hole reference cache

Durable, reusable reference data for the kernel-mapped fixed-point hole functional work.
Load via `cache/refs/loader.py` (`load_hf`, `load_baseline`, `load_hole_ref`). ~7 MB total,
committed in full (orbital files are small, ~90 KB each).

## Contents

| dir | files | what | keys |
|---|---|---|---|
| `hf/` | `hf_Z{NN}.npz` (Z=1–57, 72–83) | HF/PSP solve, **with exact orbitals** | `Z, converged, Ehf(=hf_exchange), Etot, domain, attempt, r, rho, w, r_sorted, g_sorted, occ, l_values` |
| `baselines/` | `base_Z{NN}.npz` (same Z) | self-consistent PBE & rSCAN exchange | `Z, pbe_Ex, rscan_Ex, converged` |
| `holes/` | `hole_refs.npz` | moment-matched exact-hole refs, 13 closed-shell atoms (He/Be/Ne/Mg/Ar/Ca/Zn/Kr/Sr/Cd/Xe/Ba/Hg) | per atom `<sym>_rt, _cn, _s, _Q, _Rad, _rho, _eps_sel, _rsel, _Ehf` |
| `holes/` | `hole_refs_full.npz` | **full flat (atom×r0) table**, SPIN-RESOLVED holes (Hund per-spin) for **all 69 atoms** (Z=1–57,72–83), 150 pts each, 10350 total (2100 closed-shell + 8250 open-shell) | stacked `X, cn, rt` (Npts,n_out); `Z, r0, rho, Rad, Q, s, eps_win, eps_full, eps_mm, leakQ, leakE, closed` (Npts,); `atom_{Z,sym,Ehf,Emm,offset,npts}`; `R_c, n_out, X_window` |
| `holes/` | `training_sets.npz` | **nested FPS training sets**: valid-filtered (leakage ≤10% AND ρ≥0.1 uncapped, 4523/10350 kept; mix of closed+open-shell refs) + farthest-point-sampled in the kernel metric. Prefixes of one ordering = increasing-density sets | `order` (M,) GLOBAL idx into `hole_refs_full`; `sizes` (S,) [16…512,M]; `keep_mask`; `seed, leakage_cutoff, leakage_mode, rho_floor, fp_l0, fp_l1`; `fill_dist` (S,) |
| `regen/` | python scripts | regenerators (see below) | — |

### Conventions
- All HF/PSP solves: `all_electron_flag=False`, `xc_functional="HF"`. `Ehf` = `energy_components.hf_exchange`
  = the **exact exchange of the HF density** (an exact-hole reconstruction from the cached orbitals
  reproduces it to ≤2.7 mHa for the closed-shell set).
- Orbitals: `g_sorted` is `(nr, n_orb)`, `occ`/`l_values` are `(n_orb,)`; consumed by
  `atom.xc.orbital_hole` (general-l exact hole).
- Hole refs: `rt` are the moment-matched hole coefficients on the adaptive n_out=10 unit frame
  (pinned to charge=−1, on-top=−ρ/2, Coulomb=exact ε_x); `eps_sel` = the per-r0 energy density
  (reproduces HF exchange). Built at R_c=6, n_out=10, X=8.
- **Exact-atom rule**: a reference is only built where the spin-restricted addition-theorem hole is
  *exact* — every occupied subshell full, `occ_i == 2(2l_i+1)`. `build_refs_full.py` auto-detects this
  (it picks up **Pd**, which the older hardcoded list missed). Open-(sub)shell atoms need the
  spin-resolved 1-RDM (not yet implemented) and are excluded — a reference must be an exact point on
  the universal manifold, not an approximation.
- **Leakage** (`hole_refs_full.npz` only): `leakQ = 1+Q_window` (hole charge beyond the R_c window;
  true total = −1) and `leakE = (eps_full−eps_win)/eps_full` (energy beyond the window). Diagnostic of
  where the hard `R_c=6` cap bites — ~0 in valence (the scale-free `8/k_F` window captures the hole),
  but up to ~99% at diffuse alkaline-earth/semicore tails where `X/k_F > 6` caps the window. The
  moment-match still reproduces the exact energy there by moving only the low-order modes; use leakage
  to weight/screen distorted points when selecting a training set. Energy-reconstruction MAE 0.63 mHa.

## Regeneration (`regen/`)
- `hf_fleet.py` — HF/PSP fleet over Z=1–57,72–83 (retry ladder), → `hf_Z*.npz`.
- `baselines.py` — PBE + rSCAN exchange over the same set → `base_Z*.npz`.
- `build_refs_from_cache.py` — moment-matched hole refs from the cached HF orbitals → `hole_refs.npz`.
- `build_refs_full.py` — full flat (atom×r0) **spin-resolved** exact-hole table with leakage →
  `hole_refs_full.npz` (all 69 atoms; `orbital_hole.exchange_hole_spin` + Hund per-spin occupations;
  reduces to the restricted hole for closed subshells). Load via `loader.load_hole_refs_full()`.
- `build_closed_shell_functional.py` — pins the best CLOSED-SHELL-ONLY functional →
  `atom/xc/data/kernel_fp_refs_closed_n512.npz` (512 FPS refs, closed-valid pool only). Load cleanly:
  `SIMPLEHOLEKERNELFPParameters(fp_l0=0.7, fp_l1=0.5, refs_path=<file>)` (also via
  `AtomicDFTSolver(xc_params=...)`). Non-SCF MAE in 64 / out 50 mHa. **SCF caveat:** converges for
  He (FA-dominated) but NOT Ne at 512 nodes (slow, non-converged) — benchmark non-SCF for now.
- `benchmark_vs_baselines.py` — categorized SIMPLE-vs-PBE/rSCAN. **Currently INVALID**: the cached
  PBE/rSCAN exchange is inconsistent with `Ehf` (0.7–13 Ha gap, a PSP/setup mismatch, not spin/density)
  — needs PBE/rSCAN recomputed non-SCF on the HF densities. See `simple-hole-baseline-cache` memory.
- `build_training_sets.py` — VALID-filter (leakage ≤ `--leakage-cutoff`/`--mode` AND ρ ≥ `--rho-floor`,
  default the uncapping density ≈0.08 where `R_ad=X/k_F` uncaps and σ=hole/(−ρ/2) is scale-free) then
  farthest-point-sample → `training_sets.npz`. Distance = the functional's own kernel metric (`_Kmat`,
  s² bounded as in `_kernel_eps`) with **tunable** scales (`--l0`,`--l1`; default kernel `_fp_l0=0.5`,
  LO `_fp_l1≈10.6`). `--plot`. Load via `loader.load_training_set(size)`; prefixes of `order` are nested.
- `build_kernel_refs.py` — convert a training set → kernel reference nodes
  `atom/xc/data/kernel_fp_refs_n{16,64,256}.npz` (`X`=[cn[1:], bounded s²], `DELTA`=σ_ref−σ_LDA).
  Writes NAMED files; does NOT touch the canonical `kernel_fp_refs.npz` (baseline stays reference-free).
- `benchmark_refs.py` — non-SCF E_x vs HF for in-/out-of-domain atoms across reference-free + n16/…/512
  (`--l0`/`--l1`) → `reports/hole_expansion/benchmark_refs.txt`. Result at the balanced optimum
  (isotropic l0=0.7, l1=0.5, N=512): MAE in 64 / out 50 mHa (~5× better than reference-free 302/204),
  monotone improving from n64. **Ceiling ≈ 60 mHa balanced** with this dataset — N>512 and ARD-SE
  (per-dim length scales, supported in `_Kmat` via `_fp_ell`) give no further gain; in-domain is
  pinned ~64 mHa (the ρ≥0.1 floor excludes the inhomogeneous region → adaptive R_c is the real lever).

These write to a scratchpad `refs/` by default; re-point the output paths to this `cache/refs/`
to refresh in place.

## Provenance
Generated during the kernel-hole campaign (branch `simple-hole-kernel-map`). The HF/PBE/rSCAN data
is trustworthy and method-independent; it underpins all benchmarks. (The earlier GEA-mode / δF
"kernel" *results* were retired — see `reports/hole_expansion/` — but this reference data stands.)
