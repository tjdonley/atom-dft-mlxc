"""Public multipole calculator API."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .base import DescriptorCalculator, DescriptorContext
from .basis import resolve_angular_basis
from .multipole import (
    MultipoleResult,
    compute_descriptors,
    compute_descriptors_from_radial,
    normalize_spacing,
)


class MultipoleCalculator(DescriptorCalculator):
    """Compute multipole descriptors from radial or 3D electron density.

    Parameters
    ----------
    angular_basis : str
        Angular basis family. Currently supports ``"mcsh"``.
    radial_basis : str
        Radial basis family. Currently supports ``"heaviside"`` and
        ``"legendre"``.
    rcuts : sequence of float
        Cutoff radii in Bohr.
    l_max : int
        Maximum angular order to evaluate.
    box_size : float
        Side length of the radial-projection cube in Bohr.
    spacing : float or (3,) tuple
        Grid spacing in Bohr.
    periodic : bool
        Whether periodic boundary conditions are used on the 3D descriptor grid.
    radial_order : int
        Order used by the Legendre radial basis.
    name : str
        Result key under ``descriptor_results`` when called by the solver.
    """

    def __init__(
        self,
        *,
        angular_basis: str = "mcsh",
        radial_basis: str = "heaviside",
        rcuts: Sequence[float],
        l_max: int = 2,
        box_size: float = 20.0,
        spacing: float | tuple[float, float, float] = 0.3,
        periodic: bool = True,
        radial_order: int = 0,
        name: str = "multipole",
    ):
        resolve_angular_basis(angular_basis)
        self.spacing = normalize_spacing(spacing)

        self.angular_basis = angular_basis
        self.radial_basis = radial_basis
        self.rcuts = tuple(rcuts)
        self.l_max = l_max
        self.box_size = float(box_size)
        self.periodic = periodic
        self.radial_order = radial_order
        self._name = name

        self._validate()

    def _validate(self) -> None:
        if not self.rcuts:
            raise ValueError("rcuts must be a non-empty sequence")
        if any(rcut <= 0 for rcut in self.rcuts):
            raise ValueError("All rcut values must be strictly positive")
        if self.l_max < 0:
            raise ValueError(f"l_max must be non-negative, got {self.l_max}")
        if self.box_size <= 0:
            raise ValueError(f"box_size must be positive, got {self.box_size}")
        if max(self.rcuts) > self.box_size / 2:
            raise ValueError(
                f"Largest rcut ({max(self.rcuts)}) exceeds half box_size ({self.box_size / 2}). "
                f"Increase box_size or reduce rcuts."
            )
        if not isinstance(self.periodic, bool):
            raise TypeError(f"periodic must be a bool, got {type(self.periodic)!r}")
        if not isinstance(self.radial_order, int):
            raise TypeError(
                f"radial_order must be an int, got {type(self.radial_order)!r}"
            )
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a string, got {type(self.name)!r}")
        if not self.name:
            raise ValueError("name must be a non-empty string")

        # Validate radial basis by attempting to build a kernel for one cutoff.
        from .kernels import build_radial_kernel

        build_radial_kernel(
            self.radial_basis, radius=self.rcuts[0], order=self.radial_order
        )

    @property
    def name(self) -> str:
        return self._name

    def compute(self, context: DescriptorContext) -> MultipoleResult:
        return self.compute_from_radial(
            r_quad=context.quadrature_nodes,
            rho=context.density,
        )

    def compute_from_radial(
        self,
        r_quad: np.ndarray,
        rho: np.ndarray,
        eval_indices: Optional[np.ndarray] = None,
    ) -> MultipoleResult:
        center = (self.box_size / 2, self.box_size / 2, self.box_size / 2)
        return compute_descriptors_from_radial(
            r_radial=r_quad,
            rho_radial=rho,
            box_size=self.box_size,
            spacing=self.spacing,
            atom_center=center,
            rcuts=self.rcuts,
            angular_basis=self.angular_basis,
            l_max=self.l_max,
            eval_indices=eval_indices,
            periodic=self.periodic,
            radial_basis=self.radial_basis,
            radial_order=self.radial_order,
        )

    def compute_from_solver_result(
        self,
        result: Dict,
        eval_indices: Optional[np.ndarray] = None,
    ) -> MultipoleResult:
        return self.compute_from_radial(
            r_quad=result["quadrature_nodes"],
            rho=result["rho"],
            eval_indices=eval_indices,
        )

    def compute_from_3d(
        self,
        rho_3d: np.ndarray,
        spacing: float | tuple[float, float, float] | None = None,
        eval_indices: Optional[np.ndarray] = None,
        center: tuple[float, float, float] | None = None,
    ) -> MultipoleResult:
        spacing = self.spacing if spacing is None else normalize_spacing(spacing)
        return compute_descriptors(
            rho_3d=rho_3d,
            spacing=spacing,
            rcuts=self.rcuts,
            angular_basis=self.angular_basis,
            l_max=self.l_max,
            eval_indices=eval_indices,
            periodic=self.periodic,
            radial_basis=self.radial_basis,
            radial_order=self.radial_order,
            center=center,
        )

    def extract_radial_profile(
        self,
        result: MultipoleResult,
        center: Optional[tuple[float, float, float]] = None,
    ) -> Dict:
        center_to_use = result.center if center is None else center
        center_array = np.array(center_to_use)
        r = np.linalg.norm(result.grid_positions - center_array, axis=1)
        return {
            "r": r,
            "descriptors": result.descriptors,
            "rcuts": result.rcuts,
            "l_max": result.l_max,
        }
