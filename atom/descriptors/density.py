from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class GaussianComponent:
    amplitude: float
    center: float
    width: float


def centered_single_gaussian_density(
    r: np.ndarray,
    sigma: float,
    n_electrons: float,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("Radial grid must be non-negative.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if n_electrons <= 0.0:
        raise ValueError("n_electrons must be positive.")

    prefactor = n_electrons / ((2.0 * np.pi * sigma * sigma) ** 1.5)
    return prefactor * np.exp(-0.5 * (r / sigma) ** 2)


def centered_single_gaussian_derivatives(
    r: np.ndarray,
    sigma: float,
    n_electrons: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    rho = centered_single_gaussian_density(r, sigma=sigma, n_electrons=n_electrons)
    sigma2 = sigma * sigma
    sigma4 = sigma2 * sigma2
    drho = -(r / sigma2) * rho
    d2rho = ((r * r) / sigma4 - 1.0 / sigma2) * rho
    return rho, drho, d2rho


def gaussian_mixture_density(
    r: np.ndarray,
    components: Iterable[GaussianComponent],
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("Radial grid must be non-negative.")

    rho = np.zeros_like(r)
    for component in components:
        if component.width <= 0.0:
            raise ValueError("Gaussian width must be positive.")
        arg = (r - component.center) / component.width
        rho += component.amplitude * np.exp(-0.5 * arg * arg)

    return np.maximum(rho, 0.0)


def normalize_density(r: np.ndarray, rho: np.ndarray, n_electrons: float = 1.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if r.ndim != 1 or rho.ndim != 1 or r.size != rho.size:
        raise ValueError("r and rho must be one-dimensional arrays with matching shape.")
    if np.any(r < 0.0):
        raise ValueError("Radial grid must be non-negative.")
    if n_electrons <= 0.0:
        raise ValueError("n_electrons must be positive.")

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    current = 4.0 * np.pi * trapezoid(r * r * rho, r)
    if current <= 0.0:
        raise ValueError("Density integral is non-positive; cannot normalize.")
    return rho * (n_electrons / current)


def make_linear_interpolator(r: np.ndarray, rho: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if r.ndim != 1 or rho.ndim != 1 or r.size != rho.size:
        raise ValueError("r and rho must be one-dimensional arrays with matching shape.")
    if np.any(np.diff(r) <= 0.0):
        raise ValueError("Radial grid must be strictly increasing.")

    def interp(query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=float)
        out = np.interp(q, r, rho, left=0.0, right=0.0)
        out[q < 0.0] = 0.0
        return out

    return interp


def finite_difference_derivatives(r: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    if r.ndim != 1 or f.ndim != 1 or r.size != f.size:
        raise ValueError("r and f must be one-dimensional arrays with matching shape.")
    first = np.gradient(f, r, edge_order=2)
    second = np.gradient(first, r, edge_order=2)
    return first, second
