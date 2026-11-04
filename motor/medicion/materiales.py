from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _lambda_valida(lam):
    lam = np.asarray(lam, dtype=float)
    if lam.size == 0 or not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("lambda debe ser finita y > 0 (um)")
    return lam


def _indice_valido(n):
    if not np.all(np.isfinite(n)) or np.any(n <= 0):
        raise ValueError("el modelo de material no da un indice real finito > 0 en esta lambda")
    return n


class Material:

    nombre: str = "material"

    def n(self, lam):
        raise NotImplementedError

    def n_grupo(self, lam):
        raise NotImplementedError

    def delta_g(self, lam):
        return self.n_grupo(lam) - self.n(lam)


@dataclass(frozen=True)
class Cauchy(Material):

    A: float
    B: float = 0.0
    C: float = 0.0
    nombre: str = "cauchy"

    def __post_init__(self):
        if not np.all(np.isfinite([self.A, self.B, self.C])):
            raise ValueError("coeficientes Cauchy deben ser finitos")

    def n(self, lam):
        lam = _lambda_valida(lam)
        return _indice_valido(self.A + self.B / lam**2 + self.C / lam**4)

    def n_grupo(self, lam):
        lam = _lambda_valida(lam)
        return _indice_valido(self.A + 3.0 * self.B / lam**2 + 5.0 * self.C / lam**4)


@dataclass(frozen=True)
class Sellmeier(Material):

    Bs: Tuple[float, ...]
    Cs: Tuple[float, ...]
    nombre: str = "sellmeier"

    def __post_init__(self):
        if (
            len(self.Bs) == 0
            or len(self.Bs) != len(self.Cs)
            or not np.all(np.isfinite(self.Bs))
            or not np.all(np.isfinite(self.Cs))
        ):
            raise ValueError("Sellmeier requiere Bs y Cs finitos, no vacios, de igual longitud")

    def _n2(self, lam):
        l2 = _lambda_valida(lam) ** 2
        if any(np.any(np.isclose(l2, c, rtol=1e-12, atol=0)) for c in self.Cs):
            raise ValueError("lambda coincide con un polo de Sellmeier")
        return _indice_valido(1.0 + sum(b * l2 / (l2 - c) for b, c in zip(self.Bs, self.Cs)))

    def n(self, lam):
        return np.sqrt(self._n2(lam))

    def n_grupo(self, lam):
        lam = _lambda_valida(lam)
        l2 = lam**2
        n = self.n(lam)
        dn2 = sum(-2.0 * b * c * lam / (l2 - c) ** 2 for b, c in zip(self.Bs, self.Cs))
        dn = dn2 / (2.0 * n)
        return _indice_valido(n - lam * dn)


@dataclass(frozen=True)
class Desplazado(Material):

    base: Material
    dn: float
    nombre: str = "desplazado"

    def __post_init__(self):
        if not np.isfinite(self.dn):
            raise ValueError("desplazamiento de indice debe ser finito")

    def n(self, lam):
        return _indice_valido(self.base.n(lam) + self.dn)

    def n_grupo(self, lam):
        return _indice_valido(self.base.n_grupo(lam) + self.dn)


BK7 = Sellmeier(
    Bs=(1.03961212, 0.231792344, 1.01046945),
    Cs=(0.00600069867, 0.0200179144, 103.560653),
    nombre="BK7",
)
PVB_SUPUESTO = Cauchy(A=1.4703, B=0.0051, nombre="PVB_supuesto")
SODOCALCICO_SUPUESTO = Cauchy(A=1.5111, B=0.00425, nombre="sodocalcico_supuesto")
