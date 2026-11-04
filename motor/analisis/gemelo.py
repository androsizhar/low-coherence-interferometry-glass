from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.special import ndtr

from motor.analisis.imagen_doble import cuna_optima_laminado, separacion_laminado
from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.materiales import Desplazado
from motor.medicion.modelo_directo import (
    Capa,
    amplitudes_pico,
    camino_optico_grupo,
    confocal,
    jacobiano,
)


@dataclass
class ConfigGemelo:
    lam0: float = 1.55
    sigma_a: float = 0.3
    sigma_lnA: Optional[float] = None
    alpha: Optional[float] = None
    sigma_c: float = 1.0
    covarianza: Optional[np.ndarray] = None
    separacion_minima_a: Optional[float] = None
    amplitud_minima: Optional[float] = None

    def __post_init__(self):
        if not np.isfinite(self.lam0) or self.lam0 <= 0:
            raise ErrorDeEntrada("lam0 debe ser finita y positiva (um)")
        for nombre, valor in (
            ("sigma_a", self.sigma_a),
            ("sigma_c", self.sigma_c),
            ("sigma_lnA", self.sigma_lnA),
            ("separacion_minima_a", self.separacion_minima_a),
            ("amplitud_minima", self.amplitud_minima),
        ):
            if valor is None and nombre in ("sigma_a", "sigma_c"):
                raise ErrorDeEntrada(f"{nombre} debe ser finita y > 0")
            if valor is not None and (not np.isfinite(valor) or valor <= 0):
                raise ErrorDeEntrada(f"{nombre} debe ser finita y > 0")
        if self.alpha is not None and (
            not np.isfinite(self.alpha) or not 0 <= self.alpha < np.pi / 2
        ):
            raise ErrorDeEntrada("alpha debe estar en [0, pi/2)")


def _con_indices(capas: Sequence[Capa], dn: np.ndarray) -> List[Capa]:
    return [Capa(Desplazado(c.material, float(x)), c.d) for c, x in zip(capas, dn)]


def vector_observables(capas: Sequence[Capa], cfg: ConfigGemelo) -> np.ndarray:
    A = np.abs(amplitudes_pico(capas, cfg.lam0))
    if np.any(A <= 1e-12):
        raise ErrorDeEntrada(
            "interfaz sin contraste a lam0: sus caminos por capa no son observables "
            "en este modelo; se requiere analisis espectral"
        )
    a = camino_optico_grupo(capas, cfg.lam0)
    if cfg.separacion_minima_a is not None and np.any(a < cfg.separacion_minima_a):
        raise ErrorDeEntrada("picos no resueltos segun separacion_minima_a del instrumento")
    if cfg.amplitud_minima is not None and np.any(A < cfg.amplitud_minima):
        raise ErrorDeEntrada("interfaz por debajo de amplitud_minima del instrumento")
    y = [a]
    if cfg.alpha is not None:
        y.append(confocal(capas, cfg.lam0, cfg.alpha))
    if cfg.sigma_lnA is not None:
        y.append(np.log(A))
    return np.concatenate(y)


def sigma_observables(n_capas: int, cfg: ConfigGemelo) -> np.ndarray:
    s = [np.full(n_capas, cfg.sigma_a)]
    if cfg.alpha is not None:
        s.append(np.full(n_capas, cfg.sigma_c))
    if cfg.sigma_lnA is not None:
        s.append(np.full(n_capas + 1, cfg.sigma_lnA))
    return np.concatenate(s)


def _covarianza_valida(cov, dimension: int, nombre: str) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (dimension, dimension) or not np.all(np.isfinite(cov)):
        raise ErrorDeEntrada(f"{nombre}: se requiere matriz finita {dimension}x{dimension}")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=0):
        raise ErrorDeEntrada(f"{nombre} debe ser simetrica")
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ErrorDeEntrada(f"{nombre} debe ser definida positiva") from exc
    return (cov + cov.T) / 2


def covarianza_observables(n_capas: int, cfg: ConfigGemelo) -> np.ndarray:
    s = sigma_observables(n_capas, cfg)
    if cfg.covarianza is None:
        return np.diag(s**2)
    return _covarianza_valida(cfg.covarianza, s.size, "covarianza de observables")


@dataclass
class Distinguibilidad:
    delta: float
    distinguibles: bool
    contribuciones: np.ndarray
    error_clasificacion: float
    limites_observabilidad_especificados: bool


def distinguibilidad(
    capas1: Sequence[Capa],
    capas2: Sequence[Capa],
    cfg: ConfigGemelo,
    umbral: float = 3.0,
    *,
    dos_mediciones: bool = False,
) -> Distinguibilidad:
    if len(capas1) != len(capas2):
        raise ErrorDeEntrada("las recetas deben tener el mismo numero de capas")
    if not capas1 or not np.isfinite(umbral) or umbral <= 0:
        raise ErrorDeEntrada("se requieren capas y umbral finito > 0")
    y1, y2 = vector_observables(capas1, cfg), vector_observables(capas2, cfg)
    cov = covarianza_observables(len(capas1), cfg) * (2.0 if dos_mediciones else 1.0)
    z = np.linalg.solve(np.linalg.cholesky(cov), y1 - y2)
    d = float(np.linalg.norm(z))
    return Distinguibilidad(
        delta=d,
        distinguibles=d > umbral,
        contribuciones=z,
        error_clasificacion=float(ndtr(-d / 2.0)),
        limites_observabilidad_especificados=(
            cfg.separacion_minima_a is not None and cfg.amplitud_minima is not None
        ),
    )


@dataclass
class Sensibilidad:
    nombres: List[str]
    J: np.ndarray
    fisher: np.ndarray
    rango: int
    identificables: np.ndarray
    sigma: np.ndarray
    delta_min: np.ndarray


def sensibilidad(
    capas: Sequence[Capa],
    cfg: ConfigGemelo,
    grupos: Optional[Sequence[int]] = None,
    sigma_prior_dn: Optional[Sequence[float]] = None,
    umbral: float = 3.0,
    tol_rango: float = 1e-9,
    *,
    cov_prior_dn: Optional[np.ndarray] = None,
) -> Sensibilidad:
    n = len(capas)
    if n == 0 or not np.isfinite(umbral) or umbral <= 0 or not 0 < tol_rango < 1:
        raise ErrorDeEntrada("se requieren capas, umbral > 0 y 0 < tol_rango < 1")
    grupos = list(range(n)) if grupos is None else list(grupos)
    if len(grupos) != n:
        raise ErrorDeEntrada("grupos debe tener una entrada por capa")
    ug = sorted(set(grupos))
    G = len(ug)
    gi = np.array([ug.index(g) for g in grupos])
    d0 = np.array([c.d for c in capas])

    def f(p):
        d, dn = p[:n], p[n:]
        cs = [
            Capa(Desplazado(c.material, float(dn[gi[j]])), float(d[j])) for j, c in enumerate(capas)
        ]
        return vector_observables(cs, cfg)

    p0 = np.concatenate([d0, np.zeros(G)])
    J = jacobiano(f, p0)
    L = np.linalg.cholesky(covarianza_observables(n, cfg))
    A = np.linalg.solve(L, J)
    if sigma_prior_dn is not None and cov_prior_dn is not None:
        raise ErrorDeEntrada("usar sigma_prior_dn o cov_prior_dn, no ambos")
    if sigma_prior_dn is not None:
        sp = np.asarray(sigma_prior_dn, float)
        if sp.shape != (G,) or np.any(~np.isfinite(sp)) or np.any(sp <= 0):
            raise ErrorDeEntrada("sigma_prior_dn: una entrada > 0 por grupo")
        cov_prior_dn = np.diag(sp**2)
    if cov_prior_dn is not None:
        cp = _covarianza_valida(cov_prior_dn, G, "cov_prior_dn")
        prior = np.zeros((G, n + G))
        prior[:, n:] = np.linalg.solve(np.linalg.cholesky(cp), np.eye(G))
        A = np.vstack([A, prior])
    F = A.T @ A
    esc = np.concatenate([d0, np.ones(G)])
    _, sv, Vt = np.linalg.svd(A * esc[None, :], full_matrices=True)
    rango = int(np.sum(sv > np.sqrt(tol_rango) * sv[0])) if sv.size else 0
    nombres = [f"d{j + 1}" for j in range(n)] + [f"dn_g{g}" for g in ug]
    nulo = Vt[rango:].T
    ident = np.sum(nulo**2, axis=1) < 1e-10
    sigma = np.full(F.shape[0], np.inf)
    if rango:
        Vr = Vt[:rango].T
        cov = ((Vr / sv[:rango]) @ (Vr / sv[:rango]).T) * np.outer(esc, esc)
        sigma[ident] = np.sqrt(np.maximum(np.diag(cov)[ident], 0.0))
    return Sensibilidad(
        nombres=nombres,
        J=J,
        fisher=F,
        rango=rango,
        identificables=ident,
        sigma=sigma,
        delta_min=umbral * sigma,
    )


@dataclass
class ParametrosHUD:
    theta_i: float
    Rp: float
    Rv: float
    lam: float = 0.55

    def __post_init__(self):
        if (
            not np.all(np.isfinite([self.theta_i, self.Rp, self.Rv, self.lam]))
            or not 0 < self.theta_i < np.pi / 2
            or self.Rp <= 0
            or self.Rv <= 0
            or self.lam <= 0
        ):
            raise ErrorDeEntrada("HUD requiere 0 < theta_i < pi/2, Rp,Rv,lam > 0, finitos")

    @staticmethod
    def ejemplo() -> "ParametrosHUD":
        return ParametrosHUD(theta_i=np.deg2rad(62.0), Rp=600e3, Rv=900e3)


def n_efectivo(capas: Sequence[Capa], lam: float = 0.55) -> float:
    if not capas:
        raise ErrorDeEntrada("se requiere al menos una capa")
    d = np.array([c.d for c in capas])
    n = np.array([float(c.material.n(lam)) for c in capas])
    return float((n * d).sum() / d.sum())


def imagen_doble_receta(capas: Sequence[Capa], w: float, hud: ParametrosHUD) -> float:
    rad = separacion_laminado(
        [c.d for c in capas],
        [float(c.material.n(hud.lam)) for c in capas],
        w,
        hud.theta_i,
        hud.Rp,
        hud.Rv,
    )
    return float(np.degrees(rad) * 60.0)


def cuna_optima_receta(capas: Sequence[Capa], hud: ParametrosHUD) -> float:
    return cuna_optima_laminado(
        [c.d for c in capas],
        [float(c.material.n(hud.lam)) for c in capas],
        hud.theta_i,
        hud.Rp,
        hud.Rv,
    )


@dataclass
class PrediccionHUD:
    separacion_arcmin: float
    sigma_parametros_arcmin: Optional[float]
    jacobiano: Optional[np.ndarray]
    incertidumbre_modelo_incluida: bool = False


def predecir_imagen_doble(
    capas: Sequence[Capa], w: float, hud: ParametrosHUD, cov_parametros: Optional[np.ndarray] = None
) -> PrediccionHUD:
    media = imagen_doble_receta(capas, w, hud)
    if cov_parametros is None:
        return PrediccionHUD(media, None, None)
    C = np.asarray(cov_parametros, dtype=float)
    if (
        C.shape != (10, 10)
        or not np.all(np.isfinite(C))
        or not np.allclose(C, C.T, rtol=1e-10, atol=0)
        or np.any(np.diag(C) < 0)
    ):
        raise ErrorDeEntrada(
            "cov_parametros HUD: matriz 10x10 simetrica finita semidefinida positiva"
        )
    if np.any(C[np.diag(C) == 0] != 0):
        raise ErrorDeEntrada(
            "cov_parametros HUD: variable de varianza cero no puede tener covarianza"
        )
    C = (C + C.T) / 2
    escala = np.sqrt(np.diag(C))
    escala[escala == 0] = 1.0
    normalizada = C / np.outer(escala, escala)
    if np.linalg.eigvalsh(normalizada).min() < -1e-10:
        raise ErrorDeEntrada("cov_parametros HUD debe ser semidefinida positiva")
    p0 = np.array(
        [c.d for c in capas]
        + [float(c.material.n(hud.lam)) for c in capas]
        + [w, hud.theta_i, hud.Rp, hud.Rv]
    )

    def f(p):
        rad = separacion_laminado(p[:3], p[3:6], p[6], p[7], p[8], p[9])
        return np.array([np.degrees(rad) * 60])

    J = jacobiano(f, p0)[0]
    varianza = float(J @ C @ J)
    return PrediccionHUD(media, float(np.sqrt(max(varianza, 0.0))), J)
