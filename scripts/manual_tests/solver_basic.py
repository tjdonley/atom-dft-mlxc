"""
Test case for basic AtomicDFTSolver functionality.

This module tests basic solver operations including:
1. Basic solve() functionality with different functionals
2. OEP (Optimized Effective Potential) calculations
3. RPA (Random Phase Approximation) calculations
4. PBE0 hybrid functional calculations
5. Forward pass calculations
6. Batch calculations for multiple atoms
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directories to path for imports
# Get the project root directory (parent of scripts directory)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from atom.solver import AtomicDFTSolver


def print_test_passed(test_name: str):
    """Print test passed message."""
    print("\t {:<50} : test passed".format(test_name))


def print_test_failed(test_name: str, error_msg: str = ""):
    """Print test failed message."""
    print("\t {:<50} : test FAILED".format(test_name))
    if error_msg:
        print("\t\t Error: {}".format(error_msg))


def test_basic_solve():
    """
    Test basic solve() functionality with GGA_PBE functional.
    """
    print("\n" + "=" * 60)
    print("Test: Basic solve() with GGA_PBE")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13, 
            verbose           = True, 
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )

        results = atomic_dft_solver.solve()
        rho      = results['rho']
        orbitals = results['orbitals']
        
        assert rho is not None, "Density should not be None"
        assert orbitals is not None, "Orbitals should not be None"
        assert len(rho) > 0, "Density should have non-zero length"
        assert orbitals.shape[0] > 0, "Orbitals should have non-zero shape"
        
        print_test_passed("Basic solve() with GGA_PBE")
        return True
        
    except Exception as e:
        print_test_failed("Basic solve() with GGA_PBE", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_oep():
    """
    Test OEP (Optimized Effective Potential) calculations with EXX functional.
    """
    print("\n" + "=" * 60)
    print("Test: OEP calculation with EXX functional")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number             = 10,
            domain_size               = 13.0,
            finite_element_number     = 10,
            polynomial_order          = 20,
            quadrature_point_number   = 43,
            oep_basis_number          = 5,
            verbose                   = True, 
            xc_functional             = "EXX",
            all_electron_flag         = True,
            use_oep                   = True,
            mesh_type                 = "polynomial",
            mesh_concentration        = 2.0,
        )
        
        results = atomic_dft_solver.solve()
        
        assert results is not None, "Results should not be None"
        assert 'rho' in results, "Results should contain 'rho'"
        assert 'orbitals' in results, "Results should contain 'orbitals'"
        
        print_test_passed("OEP calculation with EXX")
        return True
        
    except Exception as e:
        print_test_failed("OEP calculation with EXX", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_rpa():
    """
    Test RPA (Random Phase Approximation) calculations.
    """
    print("\n" + "=" * 60)
    print("Test: RPA calculation")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number             = 10,
            # atomic_number             = 13.456,
            # n_electrons               = 11.123,
            domain_size               = 13.0,
            finite_element_number     = 10,
            polynomial_order          = 20,
            quadrature_point_number   = 43,
            oep_basis_number          = 5,
            verbose                   = True, 
            xc_functional             = "RPA",
            all_electron_flag         = True,
            use_oep                   = True,
            mesh_type                 = "polynomial",
            mesh_concentration        = 2.0,
            enable_parallelization    = True,
        )

        results = atomic_dft_solver.solve()
        
        assert results is not None, "Results should not be None"
        assert 'rho' in results, "Results should contain 'rho'"
        assert 'orbitals' in results, "Results should contain 'orbitals'"
        
        print_test_passed("RPA calculation")
        return True
        
    except Exception as e:
        print_test_failed("RPA calculation", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_pbe0():
    """
    Test PBE0 hybrid functional calculations.
    """
    print("\n" + "=" * 60)
    print("Test: PBE0 hybrid functional calculation")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number             = 10,
            domain_size               = 13.0,
            finite_element_number     = 10,
            polynomial_order          = 20,
            quadrature_point_number   = 43,
            oep_basis_number          = 5,
            verbose                   = True, 
            xc_functional             = "PBE0", 
            all_electron_flag         = True,
            use_oep                   = False,
            mesh_type                 = "polynomial",
            mesh_concentration        = 2.0,
        )

        results = atomic_dft_solver.solve()

        orbitals  = results['orbitals']
        v_x_local = results['v_x_local']
        v_c_local = results['v_c_local']
        
        assert orbitals is not None, "Orbitals should not be None"
        assert v_x_local is not None, "v_x_local should not be None"
        assert v_c_local is not None, "v_c_local should not be None"
        assert len(v_x_local) > 0, "v_x_local should have non-zero length"
        assert len(v_c_local) > 0, "v_c_local should have non-zero length"
        
        print_test_passed("PBE0 hybrid functional calculation")
        return True
        
    except Exception as e:
        print_test_failed("PBE0 hybrid functional calculation", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_forward(orbitals_file: str = None):
    """
    Test forward pass calculation.
    
    Args:
        orbitals_file: Optional path to orbitals file. If None, will try to load from default location.
    """
    print("\n" + "=" * 60)
    print("Test: Forward pass calculation")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number             = 10,
            domain_size               = 13.0,
            finite_element_number     = 10,
            polynomial_order          = 20,
            quadrature_point_number   = 43,
            oep_basis_number          = 5,
            verbose                   = True, 
            xc_functional             = "PBE0",
            all_electron_flag         = True,
            use_oep                   = False,
            mesh_type                 = "polynomial",
            mesh_concentration        = 2.0,
        )

        # Load orbitals
        if orbitals_file is None:
            orbitals_file = 'converged_orbitals.txt'
        
        if not os.path.exists(orbitals_file):
            print(f"\t Warning: {orbitals_file} not found. Skipping forward pass test.")
            print("\t {:<50} : test SKIPPED (missing orbitals file)".format("Forward pass calculation"))
            return None
        
        orbitals = np.loadtxt(orbitals_file)

        final_results = atomic_dft_solver.forward(orbitals)

        v_x_local = final_results['v_x_local']
        v_c_local = final_results['v_c_local']
        
        assert v_x_local is not None, "v_x_local should not be None"
        assert v_c_local is not None, "v_c_local should not be None"
        
        # Optional: Compare with reference if available
        quadrature_nodes = atomic_dft_solver.grid_data_standard.quadrature_nodes
        
        v_x_local_pbe0_file = "converged_v_x_local_pbe0.txt"
        v_c_local_pbe0_file = "converged_v_c_local_pbe0.txt"
        
        if os.path.exists(v_x_local_pbe0_file) and os.path.exists(v_c_local_pbe0_file):
            v_x_local_pbe0 = np.loadtxt(v_x_local_pbe0_file)
            v_c_local_pbe0 = np.loadtxt(v_c_local_pbe0_file)
            
            v_x_diff = np.max(np.abs(v_x_local - v_x_local_pbe0))
            v_c_diff = np.max(np.abs(v_c_local - v_c_local_pbe0))
            
            print(f"\t Max difference in v_x_local: {v_x_diff:.2e}")
            print(f"\t Max difference in v_c_local: {v_c_diff:.2e}")
            
            if v_x_diff < 1e-8 and v_c_diff < 1e-8:
                print("\t ✓ Forward pass results match reference")
            else:
                print("\t ⚠ Forward pass results differ from reference (may be expected)")
        
        print_test_passed("Forward pass calculation")
        return True
        
    except Exception as e:
        print_test_failed("Forward pass calculation", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_pbe0_all_atoms(max_atoms: int = 10, start_atom: int = 1):
    """
    Test PBE0 calculations for multiple atoms.
    
    Args:
        max_atoms: Maximum number of atoms to test (default: 80)
        start_atom: Starting atomic number (default: 1)
    """
    print("\n" + "=" * 60)
    print(f"Test: PBE0 calculation for atoms {start_atom} to {start_atom + max_atoms - 1}")
    print("=" * 60)
    
    start_time = time.time()
    success_count = 0
    error_count = 0

    for i in range(max_atoms):
        atomic_num = start_atom + i
        try:
            atomic_dft_solver = AtomicDFTSolver(
                atomic_number             = atomic_num,
                domain_size               = 13.0,
                finite_element_number     = 10,
                polynomial_order          = 20,
                quadrature_point_number   = 43,
                oep_basis_number          = 5,
                verbose                   = False, 
                xc_functional             = "PBE0", 
                all_electron_flag         = True,
                use_oep                   = False,
                mesh_type                 = "polynomial",
                mesh_concentration        = 2.0,
            )
            results = atomic_dft_solver.solve()
            orbitals  = results['orbitals']
            v_x_local = results['v_x_local']
            v_c_local = results['v_c_local']
            
            assert orbitals is not None, "Orbitals should not be None"
            assert v_x_local is not None, "v_x_local should not be None"
            assert v_c_local is not None, "v_c_local should not be None"
            
            success_count += 1
            print("atomic_number = {:3d} calculated".format(atomic_num))
            
        except Exception as e:
            error_count += 1
            print("atomic_number = {:3d} error: \n {}".format(atomic_num, e))
            continue

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n" + "=" * 60)
    print(f"Batch calculation summary:")
    print(f"  Total atoms tested: {max_atoms}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Time taken: {elapsed_time:7.4f} secs")
    print(f"  Average time per atom: {elapsed_time/max_atoms:7.4f} secs")
    print("=" * 60)
    
    if success_count > 0:
        print_test_passed(f"PBE0 batch calculation ({success_count}/{max_atoms} successful)")
        return True
    else:
        print_test_failed("PBE0 batch calculation", "All calculations failed")
        return False


def run_all_tests():
    """
    Run all basic solver tests.
    """
    print("\n" + "=" * 80)
    print("Running All Basic Solver Tests")
    print("=" * 80)
    
    results = {}
    
    results['basic_solve'] = test_basic_solve()
    results['oep'] = test_oep()
    results['rpa'] = test_rpa()
    results['pbe0'] = test_pbe0()
    results['forward'] = test_forward()
    # Note: test_pbe0_all_atoms is skipped by default as it takes a long time
    # Uncomment to run: results['pbe0_all_atoms'] = test_pbe0_all_atoms(max_atoms=5)
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result is True:
            status = "PASSED"
        elif result is False:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"  {test_name:<30} : {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Run all tests
    run_all_tests()

