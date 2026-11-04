from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from motor.medicion.modelo_directo import Capa


def _k_valida(k, minimo=1):
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size < minimo or not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError(f"k debe ser vector finito positivo de al menos {minimo} elementos")
    return k


def rejilla_k(lam_min: float = 1.5, lam_max: float = 1.6, N: int = 2**14) -> np.ndarray:
    if not (np.all(np.isfinite([lam_min, lam_max])) and 0 < lam_min < lam_max):
        raise ValueError("se requiere 0 < lam_min < lam_max")
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N debe ser entero >= 2")
    return np.linspace(2 * np.pi / lam_max, 2 * np.pi / lam_min, N)


def fuente_gauss(k: np.ndarray, lam0: float = 1.55) -> np.ndarray:
    k = _k_valida(k, minimo=2)
    if np.any(np.diff(k) <= 0) or not np.isfinite(lam0) or lam0 <= 0:
        raise ValueError("fuente requiere k creciente y lam0 finita > 0")
    k0 = 2 * np.pi / lam0
    semiancho = 0.5 * (k[-1] - k[0])
    S = np.exp(-2.0 * ((k - k0) / semiancho) ** 2)
    if S.max() == 0:
        raise ValueError("fuente sin potencia numericamente representable en la banda k")
    return S / S.max()


def reflexion_tmm(
    capas: Sequence[Capa], k: np.ndarray, theta0: float = 0.0, n_ext: float = 1.0
) -> np.ndarray:
    k = _k_valida(k)
    if not np.all(np.isfinite([theta0, n_ext])) or not 0 <= theta0 < np.pi / 2 or n_ext <= 0:
        raise ValueError("TMM requiere 0 <= theta0 < pi/2 e indice exterior > 0, finitos")
    lam = 2 * np.pi / k
    sin0 = n_ext * np.sin(theta0)
    cos0 = np.cos(theta0)
    eta0 = n_ext * cos0
    M = np.zeros((k.size, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    for capa in capas:
        n = capa.material.n(lam)
        if np.any((sin0 / n) ** 2 >= 1):
            raise ValueError("el simulador sin absorcion no soporta ondas evanescentes")
        cos_t = np.sqrt(1.0 - (sin0 / n) ** 2)
        eta = n * cos_t
        delta = k * n * capa.d * cos_t
        c, s = np.cos(delta), np.sin(delta)
        Mj = np.empty_like(M)
        Mj[:, 0, 0] = c
        Mj[:, 0, 1] = 1j * s / eta
        Mj[:, 1, 0] = 1j * s * eta
        Mj[:, 1, 1] = c
        M = np.einsum("kij,kjl->kil", M, Mj)
    etas = n_ext * cos0
    B = M[:, 0, 0] + M[:, 0, 1] * etas
    Cc = M[:, 1, 0] + M[:, 1, 1] * etas
    return (eta0 * B - Cc) / (eta0 * B + Cc)


def interferograma(
    capas: Sequence[Capa],
    k: np.ndarray,
    S: np.ndarray,
    offset: float,
    rho: float = 0.5,
    ruido: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    theta0: float = 0.0,
    incluir_autotermino: bool = True,
) -> np.ndarray:
    k = _k_valida(k)
    S = np.asarray(S, dtype=float)
    if S.shape != k.shape or not np.all(np.isfinite(S)) or np.any(S < 0):
        raise ValueError("espectro S debe tener forma de k y valores finitos no negativos")
    if not np.all(np.isfinite([offset, rho, ruido])) or rho < 0 or ruido < 0:
        raise ValueError("offset debe ser finito; rho y ruido finitos no negativos")
    r = reflexion_tmm(capas, k, theta0)
    Es = r * np.exp(-2j * k * offset)
    Er = rho * np.ones_like(k, dtype=complex)
    cruz = 2.0 * np.real(np.conj(Er) * Es)
    I = rho**2 + cruz
    if incluir_autotermino:
        I = I + np.abs(r) ** 2
    I = S * I
    if ruido > 0:
        rng = rng or np.random.default_rng()
        I = I + ruido * rng.standard_normal(I.size)
    return I
