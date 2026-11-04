from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from motor.medicion.materiales import Material


@dataclass(frozen=True)
class Capa:
    material: Material
    d: float

    def __post_init__(self):
        if not (np.isfinite(self.d) and self.d > 0):
            raise ValueError(f"espesor debe ser > 0, recibido {self.d}")


def g_conf(n, alpha):
    if not (0.0 <= alpha < np.pi / 2):
        raise ValueError(f"apertura alpha debe estar en [0, pi/2), recibido {alpha}")
    n = np.asarray(n, dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n <= 0):
        raise ValueError("indice confocal debe ser finito y positivo")
    s = np.sin(alpha)
    rad = n**2 - s**2
    if np.any(rad <= 0):
        raise ValueError("n^2 - sin^2(alpha) <= 0: apertura incompatible con el indice")
    return np.cos(alpha) / np.sqrt(rad)


def camino_optico_grupo(capas: Sequence[Capa], lam0: float) -> np.ndarray:
    return np.array([float(c.material.n_grupo(lam0)) * c.d for c in capas])


def camino_optico_fase(capas: Sequence[Capa], lam0: float) -> np.ndarray:
    return np.array([float(c.material.n(lam0)) * c.d for c in capas])


def confocal(capas: Sequence[Capa], lam0: float, alpha: float) -> np.ndarray:
    return np.array([c.d * float(g_conf(c.material.n(lam0), alpha)) for c in capas])


def observables(capas: Sequence[Capa], lam0: float, alpha: float = 0.0):
    return camino_optico_grupo(capas, lam0), confocal(capas, lam0, alpha)


def indices_apilado(capas: Sequence[Capa], lam0: float, n_ext: float = 1.0) -> np.ndarray:
    if not np.isfinite(n_ext) or n_ext <= 0:
        raise ValueError("indice exterior debe ser finito y positivo")
    return np.array([n_ext] + [float(c.material.n(lam0)) for c in capas] + [n_ext])


def r_interfaces(capas: Sequence[Capa], lam0: float, n_ext: float = 1.0) -> np.ndarray:
    n = indices_apilado(capas, lam0, n_ext)
    return (n[:-1] - n[1:]) / (n[:-1] + n[1:])


def amplitudes_pico(capas: Sequence[Capa], lam0: float, n_ext: float = 1.0) -> np.ndarray:
    r = r_interfaces(capas, lam0, n_ext)
    R = r**2
    perdida = np.concatenate([[1.0], np.cumprod(1.0 - R)[:-1]])
    return r * perdida


def jacobiano(f, p: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or not np.isfinite(rel) or rel <= 0:
        raise ValueError("jacobiano requiere parametros finitos 1-D y paso relativo > 0")
    f0 = np.atleast_1d(f(p))
    if not np.all(np.isfinite(f0)):
        raise ValueError("observable no finito al evaluar Jacobiano")
    J = np.zeros((f0.size, p.size))
    for i in range(p.size):
        h = np.zeros_like(p)
        h[i] = rel * max(abs(p[i]), 1.0)
        J[:, i] = (np.atleast_1d(f(p + h)) - np.atleast_1d(f(p - h))) / (2.0 * h[i])
    if not np.all(np.isfinite(J)):
        raise ValueError("Jacobian no finito: revisar dominio y diferenciabilidad del modelo")
    return J
