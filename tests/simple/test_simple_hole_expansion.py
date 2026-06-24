"""Gate tests for the direct-expansion SIMPLE exchange hole.

Phase A (this file, first block): the representation primitives and the HEG -> LDA limit.
Later phases append the parameter-free map, the orbital-hole reference, and the production
functional gates.

Provenance for every numeric gate: R_c, n_channels, xi*, nu (quadrature). Production basis
settings are R_c = 6 bohr, n_out = 10, l_max <= 3 (CODEMAP).
"""
import numpy as np
import pytest

from atom.xc import simple_hole_expansion_explicit as ex

R_C = 6.0          # bohr (production)
N_CHAN = 16        # monopole channels for the explicit reference
NU = 512           # quadrature nodes


# --------------------------------------------------------------------------- #
# A3: closed-form vs quadrature for the per-basis-function moments
# --------------------------------------------------------------------------- #
def test_A3_charge_moments_closed_vs_quad():
    a_closed = ex.charge_moments(N_CHAN, R_C)
    a_quad = ex.charge_moments_quad(N_CHAN, R_C, nu=NU)
    assert np.allclose(a_closed, a_quad, atol=1e-10, rtol=0), \
        f"max |a_closed - a_quad| = {np.max(np.abs(a_closed - a_quad)):.2e}"


def test_A3_coulomb_moments_closed_vs_quad():
    b_closed = ex.coulomb_moments(N_CHAN, R_C)
    b_quad = ex.coulomb_moments_quad(N_CHAN, R_C, nu=NU)
    assert np.allclose(b_closed, b_quad, atol=1e-10, rtol=0), \
        f"max |b_closed - b_quad| = {np.max(np.abs(b_closed - b_quad)):.2e}"


def test_A3_odd_coulomb_moments_vanish():
    b = ex.coulomb_moments(N_CHAN, R_C)
    assert np.allclose(b[1::2], 0.0, atol=1e-14)
    assert np.all(np.abs(b[0::2]) > 1e-8)


# --------------------------------------------------------------------------- #
# A1: HEG hole projected onto the basis reproduces LDA exchange
#
# Two distinct truncation sources, disentangled here:
#   (resolution) finite N: need N >~ k_F R_c / pi to resolve the hole's oscillations.
#       Once satisfied, the HEG->LDA ratio SATURATES (adding channels does nothing).
#   (tail) finite R_c, i.e. finite dimensionless window x_c = k_F R_c: the hole tail
#       spills past R_c, so the saturated ratio sits slightly below 1 and the sum rule
#       below -1. Shrinks as x_c grows. (Phase B's constraint projection restores -1 exactly.)
# --------------------------------------------------------------------------- #
def test_A1_heg_ratio_saturates_in_N():
    """At a well-resolved density the ratio saturates by N=16 and is within ~0.2% of LDA."""
    rho = 2.0  # k_F R_c/pi ~ 7.4, so N=16 channels resolve the hole
    b = ex.coulomb_moments(N_CHAN, R_C)
    ratio16 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, N_CHAN, nu=1024), b) \
        / float(ex.lda_exchange_per_particle(rho))
    b32 = ex.coulomb_moments(32, R_C)
    ratio32 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, 32, nu=2048), b32) \
        / float(ex.lda_exchange_per_particle(rho))
    assert ratio16 == pytest.approx(ratio32, abs=1e-4), \
        f"not saturated: ratio(16)={ratio16:.5f} ratio(32)={ratio32:.5f}"
    assert 0.995 < ratio16 <= 1.0 + 1e-6, f"ratio={ratio16:.5f}"


def test_A1_resolution_floor():
    """Under-resolved (N below k_F R_c/pi) the ratio is poor; resolving it fixes it."""
    rho = 2.0
    ratio8 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, 8, nu=1024),
                                ex.coulomb_moments(8, R_C)) / float(ex.lda_exchange_per_particle(rho))
    ratio16 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, 16, nu=1024),
                                 ex.coulomb_moments(16, R_C)) / float(ex.lda_exchange_per_particle(rho))
    assert ratio8 < 0.95, f"expected poor resolution at N=8, got {ratio8:.5f}"
    assert ratio16 > 0.995 > ratio8


def test_A1_tail_capture_scale_free():
    """In the scale-free frame (fix x_c = k_F R_c, scale N with it), the ratio and sum rule
    approach 1 / -1 as the dimensionless window x_c grows."""
    rho = 2.0
    k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    last_ratio, last_q = None, None
    for xc in (20.0, 60.0):
        rc = xc / k_f
        n = int(2 * xc / np.pi) + 4
        c = ex.project_hole(ex.heg_hole(rho), rc, n, nu=2048)
        ratio = ex.eps_from_coeffs(c, ex.coulomb_moments(n, rc)) / float(ex.lda_exchange_per_particle(rho))
        q = ex.enclosed_charge(c, ex.charge_moments(n, rc))
        if last_ratio is not None:
            assert ratio > last_ratio and ratio > 0.999, f"x_c={xc}: ratio={ratio:.5f}"
            assert abs(q + 1.0) < abs(last_q + 1.0), f"x_c={xc}: sum rule {q:.4f} not closer to -1"
        last_ratio, last_q = ratio, q


# --------------------------------------------------------------------------- #
# A2: on-top is reconstructed exactly; sum rule carries the (documented) R_c tail deficit
# --------------------------------------------------------------------------- #
def test_A2_on_top_exact():
    rho = 2.0
    r0 = ex.radial_basis_at_origin(N_CHAN, R_C)
    coeffs = ex.project_hole(ex.heg_hole(rho), R_C, N_CHAN, nu=1024)
    ot = ex.on_top(coeffs, r0)
    assert ot == pytest.approx(-0.5 * rho, rel=1e-3), f"on-top = {ot:.5f} vs -rho/2 = {-0.5*rho:.5f}"


def test_A2_sum_rule_tail_deficit():
    """At R_c=6 the projected HEG hole encloses ~-0.96 (4% tail spills past R_c). This is the
    modeling deficit the Phase-B constraint projection enforces back to exactly -1."""
    rho = 2.0
    a = ex.charge_moments(N_CHAN, R_C)
    coeffs = ex.project_hole(ex.heg_hole(rho), R_C, N_CHAN, nu=1024)
    q = ex.enclosed_charge(coeffs, a)
    assert -1.0 < q < -0.90, f"sum rule {q:.4f} outside expected tail band"
