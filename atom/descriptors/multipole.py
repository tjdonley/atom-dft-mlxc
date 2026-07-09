"""Generic multipole descriptor engine.

The multipole framework is defined by three independent choices:
- an angular basis (e.g. ``"mcsh"``)
- a radial basis (e.g. ``"heaviside"`` or ``"legendre"``)
- an evaluation source (radial density projected to 3D or direct 3D density)

MCSH is one concrete angular basis within this broader framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .basis import resolve_angular_basis
from .kernels import build_radial_kernel


def normalize_spacing(
    spacing: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    """Normalize scalar or tuple spacing input to a 3-tuple."""
    if isinstance(spacing, (int, float)):
        spacing = float(spacing)
        if spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        return (spacing, spacing, spacing)

    if len(spacing) != 3:
        raise ValueError(f"spacing must have length 3, got {spacing!r}")

    spacing_tuple = tuple(float(value) for value in spacing)
    if any(value <= 0.0 for value in spacing_tuple):
        raise ValueError(f"spacing values must be positive, got {spacing_tuple!r}")
    return spacing_tuple


@dataclass
class MultipoleResult:
    """Result of generic multipole descriptor computation."""

    grid_indices: np.ndarray
    grid_positions: np.ndarray
    descriptors: np.ndarray
    rcuts: list[float]
    l_max: int
    spacing: tuple[float, float, float]
    angular_basis: str
    radial_basis: str
    radial_order: int
    center: tuple[float, float, float]

    def to_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            grid_indices=self.grid_indices,
            grid_positions=self.grid_positions,
            descriptors=self.descriptors,
            rcuts=np.array(self.rcuts),
            l_max=self.l_max,
            spacing=np.array(self.spacing),
            angular_basis=self.angular_basis,
            radial_basis=self.radial_basis,
            radial_order=self.radial_order,
            center=np.array(self.center),
        )


def _build_stencil(
    rcut: float,
    spacing: tuple[float, float, float],
    radial_basis: str,
    radial_order: int = 0,
):
    """Precompute stencil displacements and radial weights for a given cutoff."""
    hx, hy, hz = spacing
    ri = int(np.ceil(rcut / hx))
    rj = int(np.ceil(rcut / hy))
    rk = int(np.ceil(rcut / hz))

    di = np.arange(-ri, ri + 1)
    dj = np.arange(-rj, rj + 1)
    dk = np.arange(-rk, rk + 1)
    DI, DJ, DK = np.meshgrid(di, dj, dk, indexing="ij")
    dx = DI * hx
    dy = DJ * hy
    dz = DK * hz
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    kernel = build_radial_kernel(radial_basis, radius=rcut, order=radial_order)
    weights = kernel.evaluate(r)

    return DI, DJ, DK, dx, dy, dz, weights


def compute_descriptors(
    rho_3d: np.ndarray,
    spacing: float | tuple[float, float, float],
    rcuts: Sequence[float],
    angular_basis: str = "mcsh",
    l_max: int = 2,
    eval_indices: np.ndarray | None = None,
    periodic: bool = True,
    radial_basis: str = "heaviside",
    radial_order: int = 0,
    center: tuple[float, float, float] | None = None,
) -> MultipoleResult:
    """Compute multipole descriptors at grid points on a 3D density grid."""
    hx, hy, hz = normalize_spacing(spacing)
    basis = resolve_angular_basis(angular_basis)
    nx, ny, nz = rho_3d.shape
    rcuts = list(rcuts)
    if not rcuts:
        raise ValueError("rcuts must be a non-empty sequence")
    if any(rcut <= 0.0 for rcut in rcuts):
        raise ValueError("All rcut values must be strictly positive")
    if l_max < 0:
        raise ValueError(f"l_max must be non-negative, got {l_max}")

    if center is None:
        center = (
            (nx - 1) * hx / 2.0,
            (ny - 1) * hy / 2.0,
            (nz - 1) * hz / 2.0,
        )
    else:
        if len(center) != 3:
            raise ValueError(f"center must have length 3, got {center!r}")
        center = tuple(float(value) for value in center)

    if eval_indices is None:
        center_j = int(np.rint(center[1] / hy))
        center_k = int(np.rint(center[2] / hz))
        if not (0 <= center_j < ny and 0 <= center_k < nz):
            raise ValueError(
                f"center {center!r} lies outside the 3D grid for default sampling"
            )
        eval_indices = np.column_stack(
            [
                np.arange(nx),
                np.full(nx, center_j),
                np.full(nx, center_k),
            ]
        )

    eval_indices = np.asarray(eval_indices, dtype=int)
    if eval_indices.ndim != 2 or eval_indices.shape[1] != 3:
        raise ValueError("eval_indices must have shape (n_eval, 3)")

    positions = eval_indices.astype(float) * np.array([hx, hy, hz])

    descriptors = np.zeros((eval_indices.shape[0], len(rcuts), l_max + 1))
    stencils = [
        _build_stencil(
            rcut=rcut,
            spacing=(hx, hy, hz),
            radial_basis=radial_basis,
            radial_order=radial_order,
        )
        for rcut in rcuts
    ]

    dV = hx * hy * hz

    for idx in range(eval_indices.shape[0]):
        i0, j0, k0 = (int(eval_indices[idx, axis]) for axis in range(3))

        for rcut_index, (DI, DJ, DK, dx, dy, dz, weights) in enumerate(stencils):
            if periodic:
                ii = (i0 + DI) % nx
                jj = (j0 + DJ) % ny
                kk = (k0 + DK) % nz
                local_weights = weights
            else:
                ii_raw = i0 + DI
                jj_raw = j0 + DJ
                kk_raw = k0 + DK
                valid = (
                    (ii_raw >= 0)
                    & (ii_raw < nx)
                    & (jj_raw >= 0)
                    & (jj_raw < ny)
                    & (kk_raw >= 0)
                    & (kk_raw < nz)
                )
                ii = np.where(valid, ii_raw, 0)
                jj = np.where(valid, jj_raw, 0)
                kk = np.where(valid, kk_raw, 0)
                local_weights = weights * valid

            rho_vals = rho_3d[ii, jj, kk]
            weighted_rho_dV = local_weights * rho_vals * dV

            for l in range(l_max + 1):
                components = []
                for label, weight in basis.component_specs(l):
                    harmonic = basis.evaluate_component(dx, dy, dz, l, label)
                    zeta = float(np.sum(harmonic * weighted_rho_dV))
                    components.append((weight, zeta))
                descriptors[idx, rcut_index, l] = basis.combine_invariant(l, components)

    return MultipoleResult(
        grid_indices=eval_indices,
        grid_positions=positions,
        descriptors=descriptors,
        rcuts=rcuts,
        l_max=l_max,
        spacing=(hx, hy, hz),
        angular_basis=basis.name,
        radial_basis=radial_basis,
        radial_order=radial_order,
        center=center,
    )


def compute_descriptors_from_radial(
    r_radial: np.ndarray,
    rho_radial: np.ndarray,
    box_size: float,
    spacing: float | tuple[float, float, float],
    atom_center: tuple[float, float, float],
    rcuts: Sequence[float],
    angular_basis: str = "mcsh",
    l_max: int = 2,
    eval_indices: np.ndarray | None = None,
    periodic: bool = True,
    radial_basis: str = "heaviside",
    radial_order: int = 0,
) -> MultipoleResult:
    """Project a radial density onto a 3D grid and compute multipole descriptors."""
    from .grid3d import grid_radial_distances, project_radial_to_3d

    if box_size <= 0.0:
        raise ValueError(f"box_size must be positive, got {box_size}")
    hx, hy, hz = normalize_spacing(spacing)

    def _axis(requested_spacing: float) -> tuple[np.ndarray, float]:
        n_points = int(round(box_size / requested_spacing)) + 1
        axis = np.linspace(0.0, box_size, n_points)
        actual_spacing = (
            float(axis[1] - axis[0]) if axis.size > 1 else requested_spacing
        )
        return axis, actual_spacing

    x_1d, hx_actual = _axis(hx)
    y_1d, hy_actual = _axis(hy)
    z_1d, hz_actual = _axis(hz)
    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
    R_3d = grid_radial_distances(X, Y, Z, atom_center)
    rho_3d = project_radial_to_3d(r_radial, rho_radial, R_3d)

    return compute_descriptors(
        rho_3d=rho_3d,
        spacing=(hx_actual, hy_actual, hz_actual),
        rcuts=rcuts,
        angular_basis=angular_basis,
        l_max=l_max,
        eval_indices=eval_indices,
        periodic=periodic,
        radial_basis=radial_basis,
        radial_order=radial_order,
        center=atom_center,
    )
