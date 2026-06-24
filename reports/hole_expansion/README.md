# Direct-expansion SIMPLE exchange-hole functional — prototype summary

Prototype of the idea in `SIMPLE_hole_expansion.txt`: expand the spherically-averaged
**exchange hole** directly in the scale-free SIMPLE monopole basis,
`n_x(r0,u) = Σ_n ϱ̃_{n00}(r0) R_{n0}(u)`, with the coefficients produced by a map from the
local density that is constrained to hit known limits exactly. Branch `simple-hole-expansion`;
production `SIMPLE_HOLE`/`SIMPLE_HOLE_GGA` untouched. 39/39 tests green
(`tests/simple/test_simple_hole_expansion.py` + existing `test_simple_hole.py`).

## What was built (phases A–F)
| Phase | Deliverable | Headline validation |
|---|---|---|
| A | Representation: basis moments `a_n,b_n`, projection, `eps_x = ½·4π Σ ϱ̃_n b_n` | HEG hole → LDA (ratio saturates 0.998 once `N ≳ k_F R_c/π`) |
| B | Parameter-free map: HEG ⊕ one-electron anchors, enclosed-charge switch, 2-constraint projection | HEG→LDA & 1e→SIC exact; sum rule/on-top exact to 1e-6 |
| C | Exact orbital-based exchange hole (`orbital_hole.py`, s-only) — ground truth | integrated E_x == solver `oep_exchange` to <1 mHa (He, Be) |
| D | Self-consistent functional `SIMPLE_HOLE_EXPANSION` + discrete adjoint | adjoint == FD `dE/dρ` to 5.5e-8; SCF converges |
| E | Parameter-free GEA2 gradient correction `SIMPLE_HOLE_EXPANSION_GGA` | slope 10/81 recovered; adjoint 4.3e-9; improves He 16%→7.5% |
| F | Learnable residual layer, limits exact by construction (mechanism) | residual=0 at both anchors; charge/on-top-neutral, any weights |

## Atom exchange energies (all-electron SCF)
| atom | EXPANSION (LDA-level) | + GEA2 | exact (oep) |
|------|-----------------------|--------|-------------|
| He   | −0.863                | −0.949 | −1.0258 |
| Be   | −2.477                | −2.812 | −2.6658 |

## Key results & honest limitations
- The **representation is sound and not the bottleneck**: the exact hole projects to the SIMPLE
  basis reproducing the energy to 0.05% (He) / 2% (Be). The HEG and one-electron limits are
  exact; the discrete-adjoint potential is exact (FD-verified to ~1e-8).
- The **frontier is the density→coefficient map**. The simple HEG⊕1e enclosed-charge blend is
  LDA-level; the parameter-free GEA2 correction takes it to GGA-level (over-corrects sharp cores,
  as the bare 10/81 always does). A feature-based learnable map (Phase F mechanism, limits-safe)
  is the route to OEP accuracy.
- **Two understood fixed-R_c effects** (both absent in the scale-free frame): the diffuse-tail
  exchange hole is long-ranged (poorly windowed, but ρ-negligible so energy is unaffected), and
  high-density cores need `N ≳ k_F R_c/π` channels.

## Next build steps (documented)
1. **p-channel (l>0) orbital hole** (spherical-harmonic addition theorem; the machinery is in
   `atom/xc/hf.py`) → unlocks Ne/Ar and open-shell N/P references, and the Phase-F fit.
2. **Scale-free projection frame** (project the hole at fixed ξ* via the adaptive radius) →
   removes the fixed-R_c resolution/tail effects.
3. A density-/feature-dependent (saturated) enhancement instead of the bare GEA2 coefficient.

## Files
- `atom/xc/simple_hole_expansion_explicit.py` — operator-free reference + map + learnable layer
- `atom/xc/orbital_hole.py` — exact orbital exchange hole (ground truth, s-only)
- `atom/xc/simple_hole_expansion.py` — `SIMPLE_HOLE_EXPANSION`, `SIMPLE_HOLE_EXPANSION_GGA`
- `tests/simple/test_simple_hole_expansion.py` — all gates; `tests/simple/data/orbital_hole_{He,Be}.npz`
- `reports/hole_expansion/gen_orbital_hole_refs.py` — regenerates the reference data
- `reports/hole_expansion/phase{A..F}_report.md` — per-phase provenance + findings
