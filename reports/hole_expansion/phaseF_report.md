# Phase F — Learnable residual layer with exact limits (mechanism; stretch)

**Status:** mechanism demonstrated and gated; the data-driven *fit* is deferred (needs a larger
reference set — see below). Function `learnable_residual` in
`atom/xc/simple_hole_expansion_explicit.py`; gates Phase-F block. 35/35 hole-expansion gates green.

## The architectural guarantee
A learnable correction to the hole coefficients is added as

    rhotilde = rhotilde_paramfree + lambda(1-lambda) * project_neutral(W @ features) ;
    then the usual 2-constraint projection.

Two properties make it **impossible** for any fitted weights `W` to break the exact limits:
1. **Anchor gate** `g = lambda(1-lambda)` vanishes at `lambda = 0` (HEG) and `lambda = 1`
   (one electron), so the learned term is active only in the intermediate regime where the
   parameter-free map is weakest.
2. **Charge/on-top-neutral projection** of the residual removes its overlap with the sum-rule
   and on-top constraint rows, so it changes only the energy channel — the sum rule (−1) and
   on-top stay exact regardless of `W`.

Verified (F gates): for 20 random `W`, the residual is exactly zero at both anchors
(<1e-12) and is charge- and on-top-neutral at intermediate `lambda` (<1e-10), while retaining
a nonzero energy channel. Features here are rotation-invariant scalars (enclosed charge `Q`,
reduced gradient `s`); the SIMPLE power spectrum / bispectrum slot into `features` unchanged.

## Why the fit is deferred (honest)
Fitting `W` to reference data needs per-point exact hole coefficients across a *range* of
chemistry. The current exact-hole reference (`orbital_hole.py`, Phase C) is **s-only**, so the
only clean closed-shell training atoms are He and Be — far too few to fit and validate a
multi-feature map without overfitting. The prerequisite is the **p-channel (l>0) orbital hole**
(the spherical-harmonic addition-theorem extension flagged in Phase C), which unlocks Ne, Ar,
and the open-shell N/P set the CODEMAP lists as the open-shell benchmark. With that reference
in hand, the fit is a well-conditioned regularized least squares with leave-one-atom-out
cross-validation, and — by the guarantee above — the HEG and one-electron limits remain exact
by construction throughout.

**Deliverable:** the learnable layer is implemented and proven limit-safe; the fit awaits the
p-channel reference (the clear next build step).
