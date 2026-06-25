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

## Regeneration (`regen/`)
- `hf_fleet.py` — HF/PSP fleet over Z=1–57,72–83 (retry ladder), → `hf_Z*.npz`.
- `baselines.py` — PBE + rSCAN exchange over the same set → `base_Z*.npz`.
- `build_refs_from_cache.py` — moment-matched hole refs from the cached HF orbitals → `hole_refs.npz`.

These write to a scratchpad `refs/` by default; re-point the output paths to this `cache/refs/`
to refresh in place.

## Provenance
Generated during the kernel-hole campaign (branch `simple-hole-kernel-map`). The HF/PBE/rSCAN data
is trustworthy and method-independent; it underpins all benchmarks. (The earlier GEA-mode / δF
"kernel" *results* were retired — see `reports/hole_expansion/` — but this reference data stands.)
