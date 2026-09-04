"""OEP forward targets must not change when optional spectrum storage changes."""

import json
import shutil
import sys

import numpy as np
import pytest

from atom.data import AtomicDataManager
from atom.data.data_generation import DataGenerator

SPECTRUM_FILES = ("full_eigen_energies.txt", "full_orbitals.txt", "full_l_terms.txt")

GENERATION = {
    "atomic_number_list": [1],
    "save_energy_density": True,
    "save_derivative_matrix": False,
    "domain_size": 4.0,
    "finite_elements_number": 2,
    "polynomial_order": 2,
    "quadrature_point_number": 7,
    "oep_basis_number": 1,
    "mesh_type": "uniform",
    "use_pulay_mixing": False,
    "use_preconditioner": False,
    "max_scf_iterations": 120,
    "scf_tolerance": 1e-10,
    "verbose": False,
    "overwrite": True,
}
FORWARD = {
    "atomic_number": 1,
    "n_electrons": 1,
    "configuration_index": 1,
    "compute_energy_density": True,
    "save_full_spectrum": False,
    "domain_size": 4.0,
    "finite_elements_number": 2,
    "polynomial_order": 2,
    "quadrature_point_number": 7,
    "oep_basis_number": 1,
    "mesh_type": "uniform",
    "mesh_concentration": 2.0,
    "mesh_spacing": 0.1,
    "verbose": False,
}


def generate(root, source="GGA_PBE", target="PBE0", **options):
    root.mkdir()
    manager = AtomicDataManager(str(root), source, [target], auto_confirm=True)
    manager.generate_data(**dict(GENERATION, **options))
    config = root / "configuration_001"
    assert json.loads((config / "meta.json").read_text())["converged"]
    assert (config / target.lower() / "v_x.txt").is_file()
    return config


@pytest.fixture(scope="module")
def paired_targets(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-oep-targets")
    return tuple(
        generate(root / str(save), save_full_spectrum=save) for save in (False, True)
    )


def test_storage_option_preserves_public_pbe0_target(paired_targets):
    no_optional, with_optional = paired_targets
    # Equal source state, equal nontrivial OEP targets, independent of storage.
    for name in ("rho.txt", "orbitals.txt"):
        np.testing.assert_array_equal(
            np.loadtxt(no_optional / "gga_pbe" / name),
            np.loadtxt(with_optional / "gga_pbe" / name),
        )
    for name in ("v_x.txt", "v_c.txt", "e_x.txt", "e_c.txt"):
        np.testing.assert_allclose(
            np.loadtxt(no_optional / "pbe0" / name),
            np.loadtxt(with_optional / "pbe0" / name),
            atol=1e-12,
            rtol=0,
        )
    pbe = np.loadtxt(no_optional / "gga_pbe/v_x.txt")
    pbe0 = np.loadtxt(no_optional / "pbe0/v_x.txt")
    assert np.max(np.abs(pbe0 - 0.75 * pbe)) > 0.1
    for config in paired_targets:
        for name in SPECTRUM_FILES:
            assert (config / "gga_pbe" / name).is_file()
        provenance = json.loads((config / "pbe0/provenance.json").read_text())
        assert provenance == {
            "xc_functional": "PBE0",
            "use_oep": True,
            "source_folder": "../gga_pbe",
        }
        assert "Disabling OEP" not in (config / "pbe0/out.txt").read_text()
    for name in SPECTRUM_FILES:
        assert not (no_optional / "pbe0" / name).exists()
        assert (with_optional / "pbe0" / name).is_file()


def test_semilocal_target_does_not_require_source_spectrum(tmp_path):
    config = generate(tmp_path / "semilocal", target="LDA_PZ", save_full_spectrum=False)
    for folder in ("gga_pbe", "lda_pz"):
        for name in SPECTRUM_FILES:
            assert not (config / folder / name).exists()
    assert (
        json.loads((config / "lda_pz/provenance.json").read_text())["use_oep"] is False
    )


@pytest.mark.parametrize("functional", ["PBE0", "EXX", "RPA"])
@pytest.mark.parametrize("missing", SPECTRUM_FILES)
def test_missing_spectrum_rejects_target_without_fallback(
    tmp_path, functional, missing
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    np.savetxt(source / "orbitals.txt", np.array([[1.0], [0.0]]))
    for name in SPECTRUM_FILES:
        if name != missing:
            np.savetxt(
                source / name,
                np.ones((2, 2)) if name == "full_orbitals.txt" else np.ones(2),
            )
    stdout = sys.stdout
    with pytest.raises(ValueError, match=missing):
        DataGenerator._forward_pass_single_folder(
            atomic_number=1,
            n_electrons=1,
            xc_functional=functional,
            read_folder_path=str(source),
            write_folder_path=str(target),
            verbose=False,
        )
    assert sys.stdout is stdout
    assert not (target / "v_x.txt").exists()
    assert not (target / "provenance.json").exists()
    assert "require the full source spectrum" in (target / "out.txt").read_text()


def test_missing_intermediate_spectrum_fails_before_main_target(
    tmp_path, paired_targets
):
    config = tmp_path / "configuration_001"
    source = config / "gga_pbe"
    shutil.copytree(paired_targets[0] / "gga_pbe", source)
    intermediate = source / "outer_iter_01"
    intermediate.mkdir()
    for name in ("orbitals.txt", *SPECTRUM_FILES):
        shutil.copy(source / name, intermediate / name)
    (intermediate / "full_l_terms.txt").unlink()
    stdout = sys.stdout
    with pytest.raises(ValueError, match="outer_iter_01"):
        DataGenerator.forward_pass_single_atom_data(
            **FORWARD,
            directory_path=str(tmp_path),
            scf_xc_functional="GGA_PBE",
            forward_pass_xc_functional="PBE0",
            process_intermediate=True,
        )
    assert sys.stdout is stdout
    assert not (config / "pbe0/v_x.txt").exists()
    assert not (config / "pbe0/provenance.json").exists()


def test_complete_intermediate_target_keeps_oep_mode(tmp_path, paired_targets):
    # A saved intermediate state uses the same disk contract as the final state.
    config = tmp_path / "configuration_001"
    source = config / "gga_pbe"
    shutil.copytree(paired_targets[0] / "gga_pbe", source)
    intermediate = source / "outer_iter_01"
    intermediate.mkdir()
    for name in ("orbitals.txt", *SPECTRUM_FILES):
        shutil.copy(source / name, intermediate / name)
    DataGenerator.forward_pass_single_atom_data(
        **FORWARD,
        directory_path=str(tmp_path),
        scf_xc_functional="GGA_PBE",
        forward_pass_xc_functional="PBE0",
        process_intermediate=True,
    )
    target = config / "pbe0"
    assert (
        json.loads((target / "outer_iter_01/provenance.json").read_text())["use_oep"]
        is True
    )
    np.testing.assert_allclose(
        np.loadtxt(target / "v_x.txt"),
        np.loadtxt(target / "outer_iter_01/v_x.txt"),
        atol=1e-12,
        rtol=0,
    )
    for name in SPECTRUM_FILES:
        assert not (target / "outer_iter_01" / name).exists()
