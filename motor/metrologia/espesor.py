from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.modelo_directo import g_conf
from motor.metrologia.calibracion import Calibracion, _covarianza


@dataclass
class EspesorCapas:
    d: np.ndarray
    sigma_aleatoria: np.ndarray
    sigma_sistematica: np.ndarray
    cov_aleatoria: Optional[np.ndarray] = None
    cov_sistematica: Optional[np.ndarray] = None

    @property
    def sigma_total(self) -> np.ndarray:
        return np.sqrt(self.sigma_aleatoria**2 + self.sigma_sistematica**2)


def espesor_calibrado(
    a: np.ndarray, sigma_a: np.ndarray, cal: Calibracion, cov_a: Optional[np.ndarray] = None
) -> EspesorCapas:
    a, sigma_a = np.asarray(a, float), np.asarray(sigma_a, float)
    if a.ndim < 1 or a.shape[-1] != cal.ng.size or sigma_a.shape != a.shape:
        raise ErrorDeEntrada("a y sigma_a deben tener una entrada por capa (misma forma)")
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ErrorDeEntrada("camino optico a_j debe ser positivo y finito")
    if not np.all(np.isfinite(sigma_a)) or np.any(sigma_a < 0):
        raise ErrorDeEntrada("sigma_a debe ser finito y >= 0")
    n = cal.ng.size
    if cov_a is None:
        escalas = np.outer(cal.sigma_a, cal.sigma_a)
        corr = np.divide(cal.cov_a, escalas, out=np.zeros((n, n)), where=escalas > 0)
        np.fill_diagonal(corr, 1.0)
        Ca = corr * sigma_a[..., :, None] * sigma_a[..., None, :]
    else:
        Ca = np.asarray(cov_a, float)
        if Ca.shape != a.shape[:-1] + (n, n):
            raise ErrorDeEntrada("cov_a debe tener forma (..., capas, capas)")
        for C in Ca.reshape((-1, n, n)):
            _covarianza(C, n, "cov_a")
        if not np.allclose(np.diagonal(Ca, axis1=-2, axis2=-1), sigma_a**2, rtol=1e-6, atol=1e-15):
            raise ErrorDeEntrada("sigma_a no coincide con la diagonal de cov_a")
    d = a / cal.ng
    C_al = Ca / np.outer(cal.ng, cal.ng)
    escala_ng = d / cal.ng
    C_sis = cal.cov_ng * escala_ng[..., :, None] * escala_ng[..., None, :]
    return EspesorCapas(
        d=d,
        sigma_aleatoria=np.sqrt(np.maximum(np.diagonal(C_al, axis1=-2, axis2=-1), 0)),
        sigma_sistematica=np.sqrt(np.maximum(np.diagonal(C_sis, axis1=-2, axis2=-1), 0)),
        cov_aleatoria=C_al,
        cov_sistematica=C_sis,
    )


@dataclass
class SeparacionConfocal:
    d: float
    n: float
    cov: np.ndarray


def separar_confocal(
    a: float,
    c: float,
    delta_g: float,
    alpha: float,
    sigma_a: float = 0.0,
    sigma_c: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 60,
) -> SeparacionConfocal:
    if not np.all(np.isfinite([a, c, delta_g, alpha, sigma_a, sigma_c, tol])):
        raise ErrorDeEntrada("las entradas confocales deben ser finitas")
    if a <= 0 or c <= 0 or sigma_a < 0 or sigma_c < 0 or tol <= 0:
        raise ErrorDeEntrada("a,c,tol deben ser > 0 y sigma_a/sigma_c >= 0")
    if not 0 <= alpha < np.pi / 2:
        raise ErrorDeEntrada("alpha debe estar en [0, pi/2) rad")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ErrorDeEntrada("max_iter debe ser un entero >= 1")
    seno, coseno = np.sin(alpha), np.cos(alpha)
    inferior = max(seno, -delta_g, 0.0)

    def f(n):
        return (n + delta_g) * np.sqrt(max(n * n - seno * seno, 0.0)) / coseno - a / c

    superior = max(2.0, inferior + 1.0, np.sqrt(a / c) * 2)
    for _ in range(100):
        if f(superior) > 0:
            break
        superior *= 2
    else:
        raise ErrorDeEntrada("no se pudo acotar el indice confocal")
    try:
        n = brentq(f, inferior, superior, xtol=tol, maxiter=max_iter)
    except (ValueError, RuntimeError) as exc:
        raise ErrorDeEntrada("separacion confocal no convergio: revisa a, c, alpha") from exc
    gn = float(g_conf(n, alpha))
    d = c / gn
    dg = -coseno * n / (n * n - seno * seno) ** 1.5
    J = np.array([[n + delta_g, d], [gn, d * dg]])
    Jinv = np.linalg.inv(J)
    cov = (Jinv * np.array([sigma_a**2, sigma_c**2])) @ Jinv.T
    return SeparacionConfocal(d=float(d), n=float(n), cov=cov)
