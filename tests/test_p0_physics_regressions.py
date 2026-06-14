"""Regression tests for high-priority silent-physics bugs."""

from __future__ import annotations

import numpy as np

from atom.descriptors.mcsh import MCSHBasis
from atom.scf.driver import SCFDriver, SwitchesFlags
from atom.solver import VALID_XC_FUNCTIONAL_LIST
from atom.utils.occupation_states import OccupationInfo
from atom.xc.evaluator import DensityData, create_xc_evaluator
from atom.xc.functional_requirements import get_functional_requirements
from atom.xc.lda import LDA_PZ, LDA_SVWN


def test_mcsh_l2_components_are_traceless_solid_quadrupoles():
    basis = MCSHBasis()
    radius = 2.0
    points = np.array(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ]
    )
    dx, dy, dz = points.T

    trace = (
        basis.evaluate_component(dx, dy, dz, 2, "200")
        + basis.evaluate_component(dx, dy, dz, 2, "020")
        + basis.evaluate_component(dx, dy, dz, 2, "002")
    )
    np.testing.assert_allclose(trace, np.zeros_like(trace), atol=1e-14)

    components = []
    for label, weight in basis.component_specs(2):
        zeta = float(np.sum(basis.evaluate_component(dx, dy, dz, 2, label)))
        components.append((weight, zeta))
    assert basis.combine_invariant(2, components) == 0.0


def test_reorder_eigenstates_uses_actual_l_labels_for_cutoff_lists():
    info = OccupationInfo(
        z_nuclear=26,
        z_valence=8,
        all_electron_flag=False,
    )
    assert info.occ_l.tolist() == [2, 0]
    assert info.unique_l_values.tolist() == [0, 2]

    driver = type("DriverStub", (), {})()
    driver.occupation_info = info
    driver._l_values_for_channel_lists = (
        SCFDriver._l_values_for_channel_lists.__get__(driver, type(driver))
    )

    eigenvalues_by_l = [
        np.array([0.0]),   # l=0
        np.array([]),      # l=1, no occupied states
        np.array([20.0]),  # l=2
    ]
    eigenvectors_by_l = [
        np.full((2, 1), 0.0),
        np.empty((2, 0)),
        np.full((2, 1), 20.0),
    ]

    eigenvalues, eigenvectors = SCFDriver._reorder_eigenstates_by_occupation(
        driver,
        eigenvalues_by_l,
        eigenvectors_by_l,
        l_values_for_lists=[0, 1, 2],
    )

    np.testing.assert_array_equal(eigenvalues, np.array([20.0, 0.0]))
    np.testing.assert_array_equal(eigenvectors[0], np.array([20.0, 0.0]))


def test_full_l_terms_use_actual_l_labels_for_skipped_channels():
    info = OccupationInfo(
        z_nuclear=26,
        z_valence=8,
        all_electron_flag=False,
    )

    driver = type("DriverStub", (), {})()
    driver.occupation_info = info
    driver.hamiltonian_builder = type("HamiltonianStub", (), {})()
    driver.hamiltonian_builder.ops_builder = type("OpsStub", (), {})()
    driver.hamiltonian_builder.ops_builder.physical_nodes = np.zeros(5)
    driver._l_values_for_channel_lists = (
        SCFDriver._l_values_for_channel_lists.__get__(driver, type(driver))
    )
    driver._check_occ_and_unocc_eigenvalues_and_eigenvectors_lists = (
        SCFDriver._check_occ_and_unocc_eigenvalues_and_eigenvectors_lists.__get__(
            driver,
            type(driver),
        )
    )
    driver._reorder_eigenstates_by_occupation = (
        SCFDriver._reorder_eigenstates_by_occupation.__get__(driver, type(driver))
    )

    full_eigenvalues, _, full_l_terms = (
        SCFDriver._construct_full_eigenvalues_and_eigenvectors_and_l_terms(
            driver,
            occ_eigenvalues_list=[np.array([0.0]), np.array([20.0])],
            occ_eigenvectors_list=[np.zeros((3, 1)), np.zeros((3, 1))],
            unocc_eigenvalues_list=[np.array([1.0, 2.0]), np.array([21.0])],
            unocc_eigenvectors_list=[np.zeros((3, 2)), np.zeros((3, 1))],
        )
    )

    np.testing.assert_array_equal(
        full_eigenvalues,
        np.array([20.0, 0.0, 1.0, 2.0, 21.0]),
    )
    np.testing.assert_array_equal(full_l_terms, np.array([2, 0, 0, 0, 2]))


def test_full_spectrum_preconditioner_lists_include_skipped_l_blocks():
    info = OccupationInfo(
        z_nuclear=26,
        z_valence=8,
        all_electron_flag=False,
    )

    driver = type("DriverStub", (), {})()
    driver.occupation_info = info
    driver.hamiltonian_builder = type("HamiltonianStub", (), {})()
    driver.hamiltonian_builder.ops_builder = type("OpsStub", (), {})()
    driver.hamiltonian_builder.ops_builder.physical_nodes = np.zeros(6)
    driver._l_values_for_channel_lists = (
        SCFDriver._l_values_for_channel_lists.__get__(driver, type(driver))
    )
    driver._check_occ_and_unocc_eigenvalues_and_eigenvectors_lists = (
        SCFDriver._check_occ_and_unocc_eigenvalues_and_eigenvectors_lists.__get__(
            driver,
            type(driver),
        )
    )
    driver._reorder_eigenstates_by_occupation = (
        SCFDriver._reorder_eigenstates_by_occupation.__get__(driver, type(driver))
    )

    l_values_for_lists = driver._l_values_for_channel_lists(include_skipped=True)
    assert l_values_for_lists == [0, 1, 2]

    full_eigenvalues, _, full_l_terms = (
        SCFDriver._construct_full_eigenvalues_and_eigenvectors_and_l_terms(
            driver,
            occ_eigenvalues_list=[np.array([0.0]), np.array([]), np.array([20.0])],
            occ_eigenvectors_list=[
                np.zeros((4, 1)),
                np.zeros((4, 0)),
                np.zeros((4, 1)),
            ],
            unocc_eigenvalues_list=[
                np.array([1.0, 2.0, 3.0]),
                np.array([10.0, 11.0, 12.0, 13.0]),
                np.array([21.0, 22.0, 23.0]),
            ],
            unocc_eigenvectors_list=[
                np.zeros((4, 3)),
                np.zeros((4, 4)),
                np.zeros((4, 3)),
            ],
            l_values_for_lists=l_values_for_lists,
        )
    )

    np.testing.assert_array_equal(
        full_eigenvalues,
        np.array([20.0, 0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 21.0, 22.0, 23.0]),
    )
    np.testing.assert_array_equal(
        np.bincount(full_l_terms, minlength=3),
        np.array([4, 4, 4]),
    )


def _pz81_unpolarized_correlation(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    rs = np.asarray(rs, dtype=float)
    sqrt_rs = np.sqrt(rs)
    log_rs = np.log(rs)
    high_density = rs < 1.0

    e_c_high = A * log_rs + B + C * rs * log_rs + D * rs
    v_c_high = (
        A * log_rs
        + (B - A / 3.0)
        + (2.0 / 3.0) * C * rs * log_rs
        + ((2.0 * D - C) / 3.0) * rs
    )

    denominator = 1.0 + beta1 * sqrt_rs + beta2 * rs
    e_c_low = gamma / denominator
    v_c_low = gamma * (
        1.0
        + (7.0 / 6.0) * beta1 * sqrt_rs
        + (4.0 / 3.0) * beta2 * rs
    ) / (denominator**2)

    return (
        np.where(high_density, e_c_high, e_c_low),
        np.where(high_density, v_c_high, v_c_low),
    )


def test_lda_pz_factory_returns_perdew_zunger_not_vwn():
    evaluator = create_xc_evaluator("LDA_PZ")
    assert isinstance(evaluator, LDA_PZ)
    assert not isinstance(evaluator, LDA_SVWN)

    rs = np.array([0.5, 2.0])
    rho = 3.0 / (4.0 * np.pi * rs**3)
    expected_e_c, expected_v_c = _pz81_unpolarized_correlation(rs)

    correlation = evaluator.compute_correlation_generic(DensityData(rho=rho))

    np.testing.assert_allclose(correlation.e_generic, expected_e_c, atol=1e-14)
    np.testing.assert_allclose(correlation.v_generic, expected_v_c, atol=1e-14)


def test_lda_svwn_is_publicly_reachable_after_lda_pz_remap():
    evaluator = create_xc_evaluator("LDA_SVWN")
    assert isinstance(evaluator, LDA_SVWN)
    assert evaluator.params.functional_name == "LDA_SVWN"

    requirements = get_functional_requirements("LDA_SVWN")
    assert requirements.is_lda
    assert not requirements.needs_gradient
    assert not requirements.needs_tau

    switches = SwitchesFlags("LDA_SVWN")
    assert not switches.use_metagga
    assert not switches.use_oep
    assert "LDA_SVWN" in VALID_XC_FUNCTIONAL_LIST
