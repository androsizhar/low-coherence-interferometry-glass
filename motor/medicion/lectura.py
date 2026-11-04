from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class ErrorDeEntrada(ValueError):
    pass


def validar_espectro(k: np.ndarray, I: np.ndarray, minimo: int = 1024):
    k, I = np.asarray(k, float), np.asarray(I, float)
    if k.ndim != 1 or k.shape != I.shape or k.size < minimo:
        raise ErrorDeEntrada(f"k e I deben ser 1-D, igual tamano y al menos {minimo} muestras")
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(I))):
        raise ErrorDeEntrada("k o I contienen NaN/inf")
    if np.any(k <= 0):
        raise ErrorDeEntrada("k debe ser positivo (um^-1)")
    dk = np.diff(k)
    if np.any(dk <= 0):
        raise ErrorDeEntrada("k debe ser estrictamente creciente")
    if np.ptp(dk) / dk.mean() > 1e-6:
        raise ErrorDeEntrada("k no es uniforme: usa remuestrear_a_k_uniforme()")
    return k, I


@dataclass
class Instrumento:
    lam0: float
    offset: float
    sigma_ruido: float = 0.0
    alpha: float = 0.0
    limites_saturacion: Optional[tuple] = None

    def validar(self):
        if self.limites_saturacion is not None:
            limites = np.asarray(self.limites_saturacion, float)
            if (
                limites.shape != (2,)
                or not np.all(np.isfinite(limites))
                or limites[0] >= limites[1]
            ):
                raise ErrorDeEntrada(
                    "limites_saturacion debe ser [mínimo,máximo] ADC finito creciente"
                )
        if not np.isfinite(self.lam0) or not (0.3 < self.lam0 < 3.0):
            raise ErrorDeEntrada(f"lam0={self.lam0} um fuera de 0.3-3.0 um: revisa unidades (um)")
        if not np.isfinite(self.offset) or self.offset <= 0:
            raise ErrorDeEntrada("offset debe ser finito y > 0")
        if not np.isfinite(self.sigma_ruido) or self.sigma_ruido < 0:
            raise ErrorDeEntrada("sigma_ruido debe ser finito y >= 0")
        if not np.isfinite(self.alpha) or not 0 <= self.alpha < np.pi / 2:
            raise ErrorDeEntrada("alpha debe estar en [0, pi/2) rad")


@dataclass
class Medicion:
    k: np.ndarray
    I: np.ndarray
    posicion: np.ndarray
    normal: np.ndarray
    id: int = 0
    cov_normal: Optional[np.ndarray] = None

    def validar(self):
        validar_espectro(self.k, self.I)
        n = np.asarray(self.normal, float)
        if n.shape != (3,) or not np.all(np.isfinite(n)) or abs(np.linalg.norm(n) - 1) > 1e-6:
            raise ErrorDeEntrada("normal debe ser un vector unitario finito de 3 componentes")
        if self.cov_normal is not None:
            from motor.metrologia.diagnosticos import covarianza_normal

            covarianza_normal(n, self.cov_normal)
        p = np.asarray(self.posicion, float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ErrorDeEntrada("posicion debe tener 3 componentes finitas (um)")


def remuestrear_a_k_uniforme(lam: np.ndarray, I: np.ndarray, N: Optional[int] = None):
    lam, I = np.asarray(lam, float), np.asarray(I, float)
    if lam.shape != I.shape or lam.ndim != 1 or lam.size < 4:
        raise ErrorDeEntrada("lam e I deben ser 1-D del mismo tamano y tener >= 4 muestras")
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)) or not np.all(np.isfinite(I)):
        raise ErrorDeEntrada("lambda debe ser positiva y lam/I deben ser finitas (lambda en um)")
    if np.all(np.diff(lam) > 0):
        pass
    elif np.all(np.diff(lam) < 0):
        lam, I = lam[::-1], I[::-1]
    else:
        raise ErrorDeEntrada("lambda debe ser estrictamente monotona")
    if N is None:
        N = lam.size
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 4:
        raise ErrorDeEntrada("N debe ser un entero >= 4")
    k_src = (2 * np.pi / lam)[::-1]
    k = np.linspace(k_src[0], k_src[-1], N)
    from scipy.interpolate import CubicSpline

    return k, CubicSpline(k_src, I[::-1])(k)


def preparar_espectros_archivo(datos, remuestrear=False):
    datos = dict(datos)
    if "I" not in datos:
        return datos
    I = np.asarray(datos["I"], float)
    if I.ndim != 2 or not np.all(np.isfinite(I)):
        raise ErrorDeEntrada("I debe ser matriz (adquisiciones,muestras) sin NaN ni infinito")
    if "k" in datos:
        k = np.asarray(datos["k"], float)
        if (
            k.ndim != 1
            or len(k) < 4
            or I.shape[1] != len(k)
            or not np.all(np.isfinite(k))
            or np.any(k <= 0)
            or np.any(np.diff(k) <= 0)
        ):
            raise ErrorDeEntrada(
                "k debe ser vector finito, positivo y creciente, con una entrada por muestra de I"
            )
        dk = np.diff(k)
        uniforme = len(dk) > 0 and np.ptp(dk) / dk.mean() <= 1e-6
        if uniforme:
            return datos
        if not remuestrear:
            raise ErrorDeEntrada(
                "Eje k no uniforme: activa remuestrear_espectro en config.json tras verificar la calibración del espectrómetro"
            )
        from scipy.interpolate import CubicSpline

        nuevo = np.linspace(k[0], k[-1], len(k))
        datos["I"] = CubicSpline(k, I, axis=1)(nuevo)
        datos["k"] = nuevo
    elif "lambda_um" in datos:
        if not remuestrear:
            raise ErrorDeEntrada(
                "NPZ contiene lambda_um: activa remuestrear_espectro para convertirlo a k uniforme; la interpolación correlaciona ruido"
            )
        filas = [remuestrear_a_k_uniforme(datos["lambda_um"], fila) for fila in I]
        if not filas:
            raise ErrorDeEntrada("NPZ no contiene adquisiciones espectrales")
        datos["k"] = filas[0][0]
        datos["I"] = np.array([fila[1] for fila in filas])
    return datos
