# Phase A — Direct-expansion exchange hole: representation foundation

**Provenance:** R_c = 6 bohr, n_channels = 16 (sweeps to 64), nu = 512–2048 Gauss–Legendre
(4 panels). numpy 2.0.2, scipy 1.13.1. Module `atom/xc/simple_hole_expansion_explicit.py`;
gates `tests/simple/test_simple_hole_expansion.py` (Phase A block). All 8 gates green.

## What was built
Operator-free reference for expanding the spherically-averaged exchange hole directly in the
SIMPLE monopole basis `ñ(u) = Σ_n ϱ̃_n R_{n0}(u)`:
- `radial_basis`, `radial_basis_at_origin` — the monopole basis (identical to `simple_hole._radial_basis`).
- `charge_moments` `a_n` and `coulomb_moments` `b_n`, both **closed-form** and quadrature.
- `project_hole`, `eps_from_coeffs`, `enclosed_charge`, `on_top` — the representation primitives.
- `heg_hole`, `heg_envelope`, `lda_exchange_per_particle` — the HEG anchor + LDA reference.

## Verified
- **A3 (moments):** closed-form vs quadrature agree to <1e-10 for both `a_n` and `b_n`;
  odd-n `b_n` vanish exactly (`b_n ∝ 1 + (−1)ⁿ`). Closed forms:
  `a_n = (−1)ⁿ √2 R_c^{3/2}/((n+1)π)`, `b_n = √(2/R_c)(1+(−1)ⁿ)/k_n`.
- **A1 (HEG → LDA):** `eps_x = ½·4π·Σ_n ϱ̃_n b_n` reproduces LDA exchange. The ratio
  `eps_x/eps_lda` **saturates** in N once `N ≳ k_F R_c/π`; the saturated value approaches 1
  as the dimensionless window `x_c = k_F R_c` grows. At ρ=2.0, R_c=6 (well-resolved):
  ratio(N=16) = 0.99820, identical to N=32/64 (saturated).
- **A2 (constraints):** on-top `ñ(0) = Σ_n ϱ̃_n R_{n0}(0)` reconstructs `−ρ/2` to machine
  precision once resolved (−1.00005 vs −1.0 at ρ=2, N=16). The sum rule
  `4π Σ_n ϱ̃_n a_n = −0.960` carries a ~4% deficit at R_c=6 — the hole **tail past R_c**.

## Key finding (corrects the planning estimate)
The planning note guessed the HEG→LDA ratio improves with density. It does **not**. At fixed
(R_c, N) there are two competing truncations:

| source | mechanism | scaling |
|---|---|---|
| resolution | finite N cannot resolve a hole of width ~1/k_F | need `N ≳ k_F R_c/π` |
| tail | finite R_c truncates the hole's `1/u`-weighted tail | shrinks as `x_c = k_F R_c` grows |

So high density (narrow hole) needs *more* channels; low density (wide hole) needs a *larger
window*. Convergence at ρ=2, R_c=6:

```
N=  8  ratio=0.88347  sumrule=-0.7228   (under-resolved)
N= 12  ratio=0.98995  sumrule=-0.9239
N= 16  ratio=0.99822  sumrule=-0.9597   (saturated; residual = R_c tail)
N>=24  ratio=0.99820  sumrule=-0.9594   (no further change)
```

Scale-free frame (fix `x_c=k_F R_c`, scale N): ratio 0.9976 (x_c=20) → 0.9997 (x_c=60),
sum rule −0.953 → −0.984 — both approach the exact limit.

**Implication for the functional:** the production functional must project the hole in the
**scale-free frame** (fixed `ξ*` via the adaptive radius), exactly as the SIMPLE descriptors
do — then resolution is density-independent. The residual `−0.96` sum-rule deficit is a
modeling truncation that Phase B's 2-constraint least-norm projection restores to exactly
`−1` (and on-top to `−ρ/2`). This is the motivation for the constraint-projection step, now
empirically justified rather than assumed.
