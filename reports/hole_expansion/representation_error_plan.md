# Deconvolving representation error from model-fitting error (fit-free)

**Problem.** The KRR transfer test (richer_invariant_transfer.py) conflates two things: whether the
reduced invariant vector *determines* the exchange hole across atoms (representation error), and how
well KRR *fits* that map (fitting error). We want the first directly, without a model.

**Data.** Each reference point is a tuple
`(V_full, X_inv, Y)` where
- `V_full` = the most complete local descriptor we can build (signed axial coefficients `d_{nl}`,
  `l = 0..L_max` with `L_max` large, all radial `n`, monopole-normalized = scale-free);
- `X_inv` = a *nested* family of reduced invariants: `{cn+s^2 (current), +l=1 vector, +l<=2, +l<=3,
  power spectrum, +CG bispectrum}` (subsets/contractions of `V_full`);
- `Y` = the target. Two choices, used in parallel:
  - `eps` = the exchange energy **density** `2 pi R_ad^2 (-rho/2 . Cmom)` (scalar, in Ha) -- decision-
    relevant and the right linear projection of the hole (energy is one Coulomb moment of the hole);
  - `sigma` = the full hole-shape vector (n_out) -- completeness of the shape, not just the energy.

**Pooling = the cross-atom test.** Pool all points across all reference atoms (closed first, then
+open). Nearest neighbors are then drawn from *any* atom, so every fit-free statistic below measures
**cross-atom universality** -- the leave-atom-out question, with no model.

---

## Principle: the Bayes/representation floor

For representation `R` and target `Y`, the minimum MSE of *any* function `f(R)` is the conditional
variance `E[Var(Y|R)]`. It is > 0 **iff** `R` collides (same `R`, different `Y`). This is the
**representation error**; it caps every model (KRR or otherwise). A specific model achieves
`MSE = E[Var(Y|R)] + (fitting/approximation/estimation error)`. So

    fitting_error(model, R) = MSE(model, R) - representation_floor(R).

Estimating `representation_floor(R)` fit-free, for each nested `R`, is the whole task.

---

## Methods (in order of directness), all fit-free

### A. Collision / degeneracy analysis  (Pozdnyakov-Ceriotti 2020; Nigam et al. 2024)
The sharpest and most metric-robust test of *incompleteness*.
- Compute pairwise distances in `R`-space and in `Y` (use `|Delta eps|` for the energy version).
- **Doppelgaenger scan:** sort pairs by `d_R`; look at the distribution of `d_Y` as `d_R -> 0`. If
  `d_Y` stays bounded away from 0 (pairs with near-identical invariants but different holes), `R` is
  incomplete -- a hard floor no resolution/model removes.
- **Report the worst offenders:** pairs with tiny `d_R`, large `|Delta eps|`, *especially cross-atom*
  -- concrete degenerate counterexamples (the PRL methodology).
- Run with `R = V_full` first: **does the full density vector determine the hole?** (the PI's "maybe
  worth testing".) If `V_full` collides with different `eps`, there is a *fundamental* gap -- the
  exchange hole is a functional of the occupied orbitals (1-RDM), not the density alone, so no
  density-only descriptor can be complete. Knowing whether this bites in practice is decisive: it
  sets the ultimate floor for the entire SIMPLE-on-density approach.

### B. Gamma test + Delta test  (Evans-Jones; Liitiaeinen-Lendasse)  -- the quantitative floor
- For target `eps`: for each point find its `M`-th nearest neighbors in `R`; let
  `delta_M = mean ||Delta R||^2` and `gamma_M = 1/2 mean (Delta eps)^2`. Regress `gamma_M` on
  `delta_M` for `M = 1..p`; the **intercept `Gamma` (delta -> 0) is the irreducible noise variance** =
  `representation_floor(R)^2`. The slope is an effective Lipschitz/complexity of the map.
- **Delta test** (`M = 1`) as the simplest cross-check.
- Convert: `sqrt(Gamma)` is an irreducible **mHa** energy floor per representation. Tabulate it for the
  nested family: `V_full < ... < +bispectrum < +l<=2 < +l=1 vec < cn+s^2`. The drop from `cn+s^2` to
  `+l=1 vec` should mirror -- without the KRR blow-up -- the 6x transfer gain we saw; the `V_full`
  value is the ultimate density-descriptor floor.

### C. Information content  (KSG 2004; conditional MI, Frenzel-Pompe 2007)
- `I(R; eps)` (KSG, k-NN) per representation -> diminishing returns as `l_max` grows (completeness).
- **`I(eps; V_full | X_inv)` (conditional MI):** the energy-relevant information the rotational
  reduction *discards*. `~0` => `X_inv` is a sufficient statistic for the energy (lossless reduction);
  large => the invariants throw away energy-relevant info. This is the representation error in nats.

### D. Dependence / sufficiency cross-checks  (Szekely-Rizzo 2007; Gretton 2007)
- Distance correlation `dCor(R, eps)` and a conditional-independence test `eps _|_ V_full | X_inv`
  (HSIC-based). Sufficiency <=> conditional independence. Corroborates C without density estimation
  (more robust in moderate dimension).

---

## Deconvolution deliverable

A single table, nested representations as rows:

| representation R | collisions? (A) | floor sqrt(Gamma) [mHa] (B) | I(R;eps) (C) | I(eps;V_full|R) (C) | KRR error | fitting = KRR - floor |
|---|---|---|---|---|---|---|

This splits each KRR number (e.g. the 6761, and the in-domain 88) into "irreducible given these
invariants" vs "KRR's slack," and shows whether higher `l`/bispectrum lower the *floor* (real
information gain) independent of any fitting artifact.

---

## Practical notes / pitfalls
- **Metric dependence.** NN-floor estimates assume smoothness in the input metric, so the `R`-metric
  matters. Standardize each `R` (z-score; drop ~zero-variance dims) and report sensitivity; lean on
  (A) collisions and (D) dCor/conditional-MI, which are more metric-robust, to confirm (B).
- **Scale-free features.** Already monopole-normalized, so cross-atom neighbors are physically
  comparable -- a prerequisite for the pooled (cross-atom) floor to be meaningful.
- **Energy vs shape.** Lead with `eps` (scalar, mHa, decision-relevant); repeat key steps with the
  full `sigma` vector to separate "energy-sufficient" from "shape-complete."
- **Compute.** Only the descriptor build (`d_{nl}` per point, already implemented in
  richer_invariant_transfer.py) is non-trivial; all NN statistics on a few thousand pooled points are
  cheap. Reuse the verified descriptor pipeline; extend `L_max` for `V_full`.
- **No fitting** anywhere except importing the prior KRR numbers for the final deconvolution column.

## Expected payoffs
1. A definitive answer to "does the density `V_full` determine the exchange hole?" -- i.e., is there a
   1-RDM gap that caps SIMPLE regardless of features.
2. The representation floor of the *current* `cn+s^2`, and how much each addition (l=1 vector, l>=2,
   bispectrum) lowers it -- the information gain, model-free.
3. A principled choice of which invariants to add to the functional, justified by floor reduction
   rather than KRR behavior.
