"""Golden parity checks for the MCSH case in the generalized API.

These tests use small golden reference artifacts for the corrected MCSH
implementation. The generalized multipole API must reproduce those results
when `angular_basis="mcsh"`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator


DATA_DIR = Path(__file__).resolve().parent / "data" / "master_parity"
SYNTHETIC_REF = DATA_DIR / "synthetic_hydrogen_mcsh_refs.npz"
SOLVER_REF = DATA_DIR / "solver_h_o_mcsh_refs.npz"
SUMMARY_REF = DATA_DIR / "mcsh_validation_summary.json"

SCF_FIELD_ATOL = 5e-9
SCF_DESCRIPTOR_ATOL = 1e-9
SCF_ENERGY_ATOL = 5e-10
SUMMARY_ENERGY_ATOL = 1e-9
SUMMARY_DESCRIPTOR_ATOL = 5e-10
SUMMARY_KERNEL_DIFF_ATOL = 1e-8


@pytest.fixture(scope="module")
def synthetic_reference():
    return np.load(SYNTHETIC_REF)


@pytest.fixture(scope="module")
def solver_reference():
    return np.load(SOLVER_REF)


@pytest.fixture(scope="module")
def summary_reference():
    return json.loads(SUMMARY_REF.read_text(encoding="utf-8"))


def test_synthetic_master_parity_all_kernels(synthetic_reference):
    r = synthetic_reference["r"]
    rho = synthetic_reference["rho"]

    base_kwargs = dict(
        angular_basis="mcsh",
        rcuts=synthetic_reference["rcuts"].tolist(),
        l_max=int(synthetic_reference["l_max"]),
        box_size=float(synthetic_reference["box_size"]),
        spacing=float(synthetic_reference["spacing"]),
    )

    refs = {
        "descriptors_heaviside": MultipoleCalculator(
            **base_kwargs,
            radial_basis="heaviside",
        ).compute_from_radial(r, rho),
        "descriptors_legendre0": MultipoleCalculator(
            **base_kwargs,
            radial_basis="legendre",
            radial_order=0,
        ).compute_from_radial(r, rho),
        "descriptors_legendre1": MultipoleCalculator(
            **base_kwargs,
            radial_basis="legendre",
            radial_order=1,
        ).compute_from_radial(r, rho),
        "descriptors_legendre2": MultipoleCalculator(
            **base_kwargs,
            radial_basis="legendre",
            radial_order=2,
        ).compute_from_radial(r, rho),
    }

    np.testing.assert_array_equal(
        refs["descriptors_heaviside"].grid_indices,
        synthetic_reference["grid_indices"],
    )
    np.testing.assert_array_equal(
        refs["descriptors_heaviside"].grid_positions,
        synthetic_reference["grid_positions"],
    )

    for key, result in refs.items():
        np.testing.assert_allclose(
            result.descriptors,
            synthetic_reference[key],
            atol=1e-14,
            rtol=0.0,
        )


def test_solver_master_parity_hydrogen_and_oxygen(solver_reference):
    solver_kwargs = dict(
        xc_functional="LDA_PZ",
        domain_size=15.0,
        finite_element_number=10,
        polynomial_order=20,
        quadrature_point_number=43,
        verbose=False,
    )
    calc = MultipoleCalculator(
        angular_basis="mcsh",
        rcuts=solver_reference["rcuts"].tolist(),
        l_max=int(solver_reference["l_max"]),
        box_size=float(solver_reference["box_size"]),
        spacing=float(solver_reference["spacing"]),
    )

    for symbol, atomic_number in [("H", 1), ("O", 8)]:
        result = AtomicDFTSolver(atomic_number=atomic_number, **solver_kwargs).solve()
        desc = calc.compute_from_solver_result(result)

        np.testing.assert_allclose(
            result["quadrature_nodes"],
            solver_reference[f"{symbol}_quadrature_nodes"],
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result["quadrature_weights"],
            solver_reference[f"{symbol}_quadrature_weights"],
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result["rho"],
            solver_reference[f"{symbol}_rho"],
            atol=SCF_FIELD_ATOL,
            rtol=0.0,
        )
        assert result["energy"] == pytest.approx(
            float(solver_reference[f"{symbol}_energy"]),
            abs=SCF_ENERGY_ATOL,
        )
        np.testing.assert_array_equal(
            desc.grid_indices, solver_reference[f"{symbol}_grid_indices"]
        )
        np.testing.assert_array_equal(
            desc.grid_positions, solver_reference[f"{symbol}_grid_positions"]
        )
        np.testing.assert_allclose(
            desc.descriptors,
            solver_reference[f"{symbol}_descriptors"],
            atol=SCF_DESCRIPTOR_ATOL,
            rtol=0.0,
        )


@pytest.fixture(scope="module")
def current_scientific_summary():
    atoms = {
        "H": {"Z": 1, "Ne": 1},
        "He": {"Z": 2, "Ne": 2},
        "Li": {"Z": 3, "Ne": 3},
        "Be": {"Z": 4, "Ne": 4},
        "C": {"Z": 6, "Ne": 4},
        "N": {"Z": 7, "Ne": 5},
        "O": {"Z": 8, "Ne": 6},
    }
    solver_kwargs = dict(
        xc_functional="GGA_PBE",
        domain_size=20.0,
        finite_element_number=17,
        polynomial_order=31,
        quadrature_point_number=95,
        verbose=False,
    )
    rcuts = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    base_kwargs = dict(
        angular_basis="mcsh",
        rcuts=rcuts,
        l_max=2,
        box_size=16.0,
        spacing=0.4,
    )

    summary = {"atoms": {}, "global": {"ordered_atoms": list(atoms.keys())}}
    rcut4_values = {}

    for atom, info in atoms.items():
        sr = AtomicDFTSolver(atomic_number=info["Z"], **solver_kwargs).solve()
        r_nodes = sr["quadrature_nodes"]
        rho = sr["rho"]
        weights = sr["quadrature_weights"]

        h_calc = MultipoleCalculator(**base_kwargs, radial_basis="heaviside")
        l0_calc = MultipoleCalculator(
            **base_kwargs, radial_basis="legendre", radial_order=0
        )
        l1_calc = MultipoleCalculator(
            **base_kwargs, radial_basis="legendre", radial_order=1
        )
        l2_calc = MultipoleCalculator(
            **base_kwargs, radial_basis="legendre", radial_order=2
        )

        h = h_calc.compute_from_radial(r_nodes, rho)
        l0 = l0_calc.compute_from_radial(r_nodes, rho)
        l1 = l1_calc.compute_from_radial(r_nodes, rho)
        l2 = l2_calc.compute_from_radial(r_nodes, rho)

        profile = h_calc.extract_radial_profile(h)
        ci = int(np.argmin(profile["r"]))
        electron_count = float(np.sum(4 * np.pi * r_nodes**2 * rho * weights))

        atom_summary = {
            "energy": float(sr["energy"]),
            "electron_count": electron_count,
            "expected_valence_electrons": info["Ne"],
            "heaviside_l0_center": np.abs(profile["descriptors"][ci, :, 0]).tolist(),
            "heaviside_l1_center": profile["descriptors"][ci, :, 1].tolist(),
            "lp0_vs_heaviside_max_abs_diff": float(
                np.max(np.abs(l0.descriptors - h.descriptors))
            ),
            "lp1_vs_heaviside_max_abs_diff": float(
                np.max(np.abs(l1.descriptors - h.descriptors))
            ),
            "lp2_vs_heaviside_max_abs_diff": float(
                np.max(np.abs(l2.descriptors - h.descriptors))
            ),
        }
        summary["atoms"][atom] = atom_summary
        rcut4_values[atom] = atom_summary["heaviside_l0_center"][4]

    summary["global"]["heaviside_l0_center_rcut_3_index_4"] = rcut4_values
    return summary


def test_scientific_summary_matches_master(
    summary_reference, current_scientific_summary
):
    for atom, atom_ref in summary_reference["atoms"].items():
        atom_cur = current_scientific_summary["atoms"][atom]
        assert atom_cur["energy"] == pytest.approx(
            atom_ref["energy"], abs=SUMMARY_ENERGY_ATOL
        )
        assert atom_cur["electron_count"] == pytest.approx(
            atom_ref["electron_count"], abs=1e-12
        )
        assert (
            atom_cur["expected_valence_electrons"]
            == atom_ref["expected_valence_electrons"]
        )
        np.testing.assert_allclose(
            atom_cur["heaviside_l0_center"],
            atom_ref["heaviside_l0_center"],
            atol=SUMMARY_DESCRIPTOR_ATOL,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            atom_cur["heaviside_l1_center"],
            atom_ref["heaviside_l1_center"],
            atol=1e-12,
            rtol=0.0,
        )
        assert atom_cur["lp0_vs_heaviside_max_abs_diff"] == pytest.approx(
            atom_ref["lp0_vs_heaviside_max_abs_diff"], abs=1e-14
        )
        assert atom_cur["lp1_vs_heaviside_max_abs_diff"] == pytest.approx(
            atom_ref["lp1_vs_heaviside_max_abs_diff"], abs=SUMMARY_KERNEL_DIFF_ATOL
        )
        assert atom_cur["lp2_vs_heaviside_max_abs_diff"] == pytest.approx(
            atom_ref["lp2_vs_heaviside_max_abs_diff"], abs=SUMMARY_KERNEL_DIFF_ATOL
        )

    assert (
        current_scientific_summary["global"]["ordered_atoms"]
        == summary_reference["global"]["ordered_atoms"]
    )
    for atom, ref_value in summary_reference["global"][
        "heaviside_l0_center_rcut_3_index_4"
    ].items():
        assert current_scientific_summary["global"][
            "heaviside_l0_center_rcut_3_index_4"
        ][atom] == pytest.approx(
            ref_value,
            abs=SUMMARY_DESCRIPTOR_ATOL,
        )
