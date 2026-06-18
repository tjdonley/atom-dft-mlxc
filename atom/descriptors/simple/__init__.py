"""SIMPLE: Scale-Invariant Multipole Local Expansion of the electron density.

Reference implementation of the SIMPLE descriptors documented in
../SIMPLE-Xhole-writeup/ (Theory in main.tex, derivations in appendix.tex).
CODEMAP.md (repo root) maps every equation label of the document to the
functions below. The validation suite lives in tests/simple/.

Modules
-------
params      pinned working parameters (R_C, XI_TARGET, N_OUT, RHO_MIN, L_MAX)
bessel      Dirichlet spherical Bessel basis, monopole integrals, quadrature
rotations   real spherical harmonics and real Wigner-D blocks (any l, O(3))
pipeline    the SIMPLE pipeline: window projection, adaptive radius,
            mean-split scale transform, non-dimensionalization
            (radial and 3D Cartesian evaluation paths)
invariants  power spectrum and real-basis Clebsch-Gordan bispectrum

This subpackage is self-contained (it depends only on numpy/scipy, not on
other descriptor frameworks in this repository).
"""

from .bessel import (
    RadialBesselBasis,
    RadialQuadrature,
    a_n_closed_form,
    radial_gauss_grid,
    spherical_jn_zeros,
)
from .features import get_simple_features, get_simple_invariants
from .invariants import (
    bispectrum_components,
    channel_magnitudes,
    clebsch_gordan,
    coupling_tensor,
    coupling_tensor_from_cg,
    flatten_bispectrum,
    parity,
    power_spectrum,
    self_check_couplings,
    triangle_triples,
    u_real_to_complex,
)
from .params import BOHR_PER_ANGSTROM, L_MAX, N_OUT, R_C, RHO_MIN, XI_TARGET
from .pipeline import (
    M_TARGET,
    LatticeStencil,
    adaptive_radius,
    ball_average,
    fixed_window_descriptors,
    grid_descriptors,
    grid_nodes_weights,
    n_in_for,
    project_window,
    simple_descriptors,
    simple_from_window,
    transfer_matrix,
    window_basis,
)
from .rotations import (
    mirror_matrix,
    random_rotation_matrix,
    real_sh_rotation_matrix,
    real_sph_harm,
    real_wigner_d,
    rotation_matrix_3d,
    rotation_z_to,
)

__all__ = [
    "BOHR_PER_ANGSTROM", "L_MAX", "M_TARGET", "N_OUT", "R_C", "RHO_MIN",
    "XI_TARGET", "adaptive_radius",
    "LatticeStencil", "RadialBesselBasis", "RadialQuadrature",
    "a_n_closed_form", "ball_average", "bispectrum_components",
    "channel_magnitudes", "clebsch_gordan", "coupling_tensor",
    "coupling_tensor_from_cg", "fixed_window_descriptors",
    "flatten_bispectrum", "get_simple_features", "get_simple_invariants",
    "grid_descriptors", "grid_nodes_weights",
    "mirror_matrix", "n_in_for", "parity", "power_spectrum",
    "project_window", "radial_gauss_grid", "random_rotation_matrix",
    "real_sh_rotation_matrix", "real_sph_harm", "real_wigner_d",
    "rotation_matrix_3d", "rotation_z_to", "self_check_couplings",
    "simple_descriptors", "simple_from_window", "spherical_jn_zeros",
    "transfer_matrix", "triangle_triples", "u_real_to_complex",
    "window_basis",
]
