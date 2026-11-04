from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm

from motor.geometria.superficies import ProcesoGaussiano2D
from motor.medicion.lectura import ErrorDeEntrada


@dataclass
class Cuna:
    w: np.ndarray
    sigma_w: np.ndarray
    direccion: np.ndarray
    grad: np.ndarray
    cov_grad: np.ndarray
    direccion_estimada: bool = False


def cuna_pvb(
    gp_d_pvb: ProcesoGaussiano2D, Xs: np.ndarray, direccion: Optional[Sequence[float]] = None
) -> Cuna:
    _, _, g, cov = gp_d_pvb.predecir(Xs, gradiente=True)
    if direccion is None:
        gm = g.mean(axis=0)
        nrm = np.linalg.norm(gm)
        if not np.isfinite(nrm) or nrm < 1e-12:
            raise ErrorDeEntrada("gradiente medio nulo: no hay cuna que orientar")
        s = gm / nrm
    else:
        s = np.asarray(direccion, float)
        if s.shape != (2,) or not np.all(np.isfinite(s)) or np.linalg.norm(s) == 0:
            raise ErrorDeEntrada("direccion debe ser un vector 2-D no nulo")
        s = s / np.linalg.norm(s)
    w = g @ s
    var = np.einsum("i,mij,j->m", s, cov, s)
    return Cuna(
        w=w,
        sigma_w=np.sqrt(np.maximum(var, 0.0)),
        direccion=s,
        grad=g,
        cov_grad=cov,
        direccion_estimada=direccion is None,
    )


Tol = Union[float, Tuple[float, float]]


def probabilidad_fuera_de_tolerancia(
    mu: np.ndarray,
    sigma: np.ndarray,
    nominal: np.ndarray,
    tol: Tol,
    sigma_sistematica: Union[float, np.ndarray] = 0.0,
) -> np.ndarray:
    try:
        mu, sigma, nominal, sis = np.broadcast_arrays(
            np.asarray(mu, float),
            np.asarray(sigma, float),
            np.asarray(nominal, float),
            np.asarray(sigma_sistematica, float),
        )
        tl, tu = (float(tol), float(tol)) if np.isscalar(tol) else map(float, tol)
    except (ValueError, TypeError) as exc:
        raise ErrorDeEntrada("calidad: formas incompatibles o tolerancia invalida") from exc
    if not all(np.all(np.isfinite(a)) for a in (mu, sigma, nominal, sis)):
        raise ErrorDeEntrada("calidad: mediciones e incertidumbres deben ser finitas")
    if np.any(sigma < 0) or np.any(sis < 0):
        raise ErrorDeEntrada("las incertidumbres no pueden ser negativas")
    if not np.isfinite(tl) or not np.isfinite(tu) or tl < 0 or tu < 0:
        raise ErrorDeEntrada("la tolerancia debe ser finita y >= 0")
    st = np.hypot(sigma, sis)
    lo, hi = nominal - tl, nominal + tu
    with np.errstate(divide="ignore", invalid="ignore"):
        p_out = norm.cdf((lo - mu) / st) + norm.sf((hi - mu) / st)
        p_out = np.where(st == 0, ((mu < lo) | (mu > hi)).astype(float), p_out)
    return np.clip(p_out, 0.0, 1.0)


@dataclass
class Decision:
    P_fuera: np.ndarray
    falla: np.ndarray
    fraccion_falla: float


def decidir(P_fuera: np.ndarray, umbral: float = 0.05) -> Decision:
    if not (0 < umbral < 1):
        raise ErrorDeEntrada("umbral debe estar en (0,1)")
    P_fuera = np.asarray(P_fuera, float)
    if (
        not P_fuera.size
        or not np.all(np.isfinite(P_fuera))
        or np.any((P_fuera < 0) | (P_fuera > 1))
    ):
        raise ErrorDeEntrada("P_fuera debe contener probabilidades finitas en [0,1]")
    f = P_fuera > umbral
    return Decision(P_fuera=P_fuera, falla=f, fraccion_falla=float(np.mean(f)))
