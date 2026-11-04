from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats

from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.picos import ResultadoPicos


def _vector(x, nombre, positivo=False):
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ErrorDeEntrada(f"{nombre} debe ser un vector no vacio y finito")
    if np.any(x <= 0 if positivo else x < 0):
        raise ErrorDeEntrada(f"{nombre} debe ser {'> 0' if positivo else '>= 0'}")
    return x


def _covarianza(C, n, nombre):
    C = np.asarray(C, float)
    if C.shape != (n, n) or not np.all(np.isfinite(C)) or not np.allclose(C, C.T, atol=1e-12):
        raise ErrorDeEntrada(f"{nombre} debe ser una matriz simetrica finita de {n}x{n}")
    C = (C + C.T) / 2
    if np.linalg.eigvalsh(C).min() < -1e-10 * max(np.max(np.abs(C)), np.finfo(float).tiny):
        raise ErrorDeEntrada(f"{nombre} debe ser semidefinida positiva")
    return C


@dataclass
class Calibracion:
    ng: np.ndarray
    u_rel: np.ndarray
    sigma_a: np.ndarray
    grupos: List[str]
    banderas: List[str] = field(default_factory=list)
    m_repeticiones: int = 0
    cov_a: Optional[np.ndarray] = None
    cov_ng: Optional[np.ndarray] = None
    metadatos_referencia: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ng = _vector(self.ng, "ng", positivo=True)
        self.u_rel = _vector(self.u_rel, "u_rel")
        self.sigma_a = _vector(self.sigma_a, "sigma_a")
        n = self.ng.size
        if self.u_rel.size != n or self.sigma_a.size != n or len(self.grupos) != n:
            raise ErrorDeEntrada("calibracion debe tener una entrada por capa")
        self.cov_a = _covarianza(
            np.diag(self.sigma_a**2) if self.cov_a is None else self.cov_a, n, "cov_a"
        )
        self.cov_ng = _covarianza(
            np.diag((self.ng * self.u_rel) ** 2) if self.cov_ng is None else self.cov_ng,
            n,
            "cov_ng",
        )
        if not np.allclose(np.diag(self.cov_a), self.sigma_a**2, rtol=1e-6, atol=1e-15):
            raise ErrorDeEntrada("sigma_a no coincide con la diagonal de cov_a")
        if not np.allclose(
            np.diag(self.cov_ng), (self.ng * self.u_rel) ** 2, rtol=1e-6, atol=1e-15
        ):
            raise ErrorDeEntrada("u_rel no coincide con la diagonal de cov_ng")

    @property
    def ok(self) -> bool:
        return not self.banderas

    @property
    def sigma_ng(self) -> np.ndarray:
        return self.ng * self.u_rel


def _apilar_caminos(resultados: Sequence[ResultadoPicos], n_capas: int) -> np.ndarray:
    if len(resultados) == 0:
        raise ErrorDeEntrada("se necesita al menos una adquisicion de referencia")
    filas = []
    for r in resultados:
        if not r.ok:
            raise ErrorDeEntrada(
                f"medicion de referencia con banderas {r.banderas}: repetir o descartar"
            )
        a = np.asarray(r.caminos_capas(), float)
        if a.shape != (n_capas,) or not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ErrorDeEntrada(f"se esperaban {n_capas} caminos de capa positivos y finitos")
        s = np.asarray(r.sigma_caminos(), float)
        if s.shape != a.shape or not np.all(np.isfinite(s)) or np.any(s < 0):
            raise ErrorDeEntrada("incertidumbres de localizacion de pico invalidas")
        filas.append(a)
    return np.array(filas)


def _pesos_media(C):
    vals, vecs = np.linalg.eigh(C)
    corte = max(float(vals.max()), np.finfo(float).tiny) * np.finfo(float).eps * 100
    positivos = vals > corte
    null = vecs[:, ~positivos]
    unos = np.ones(C.shape[0])
    proy = null @ (null.T @ unos)
    if unos @ proy > 1e-10:
        w = proy / (unos @ proy)
    else:
        inv = (vecs[:, positivos] / vals[positivos]) @ vecs[:, positivos].T
        w = inv @ unos / (unos @ inv @ unos)
    inv = (vecs[:, positivos] / vals[positivos]) @ vecs[:, positivos].T
    return w, inv, null


def calibrar(
    resultados_ref: Sequence[ResultadoPicos],
    d_ref: Sequence[float],
    sigma_d_ref: Sequence[float],
    grupos: Optional[Sequence[str]] = None,
    confianza: float = 0.9973,
    cov_d_ref: Optional[np.ndarray] = None,
) -> Calibracion:
    d_ref = _vector(d_ref, "d_ref", positivo=True)
    sd = _vector(sigma_d_ref, "sigma_d_ref")
    n_capas = d_ref.size
    if sd.size != n_capas:
        raise ErrorDeEntrada("sigma_d_ref debe tener una entrada por capa")
    if not np.isfinite(confianza) or not 0 < confianza < 1:
        raise ErrorDeEntrada("confianza debe estar en (0,1)")
    if len(resultados_ref) < 5:
        raise ErrorDeEntrada("se necesitan >= 5 repeticiones para estimar la repetibilidad")
    grupos = list(grupos) if grupos is not None else [f"capa{j}" for j in range(n_capas)]
    if len(grupos) != n_capas or any(not isinstance(g, str) or not g for g in grupos):
        raise ErrorDeEntrada("grupos debe tener una etiqueta de material no vacia por capa")
    Cd = _covarianza(np.diag(sd**2) if cov_d_ref is None else cov_d_ref, n_capas, "cov_d_ref")
    if not np.allclose(np.diag(Cd), sd**2, rtol=1e-6, atol=1e-15):
        raise ErrorDeEntrada("sigma_d_ref no coincide con la diagonal de cov_d_ref")

    A = _apilar_caminos(resultados_ref, n_capas)
    m = A.shape[0]
    a_med = A.mean(axis=0)
    Ca = np.atleast_2d(np.cov(A, rowvar=False, ddof=1))
    Cdet = np.mean([r.cov_caminos() for r in resultados_ref], axis=0)
    vals, vecs = np.linalg.eigh(Cdet - Ca)
    Ca += (vecs * np.maximum(vals, 0)) @ vecs.T
    ng = a_med / d_ref
    Cng = Ca / m / np.outer(d_ref, d_ref) + Cd * np.outer(ng / d_ref, ng / d_ref)
    P = np.eye(n_capas)
    banderas: List[str] = []
    for g in sorted(set(grupos)):
        idx = np.flatnonzero(np.array(grupos) == g)
        if idx.size < 2:
            continue
        w, inv, null = _pesos_media(Cng[np.ix_(idx, idx)])
        nbar = float(w @ ng[idx])
        residuo = ng[idx] - nbar
        chi2 = float(residuo @ inv @ residuo)
        incompatible_exacto = np.max(np.abs(null.T @ residuo), initial=0) > 1e-10 * max(
            1, abs(nbar)
        )
        if incompatible_exacto or chi2 > stats.chi2.ppf(confianza, df=idx.size - 1):
            banderas.append(f"inconsistencia_grupo:{g}:chi2={chi2:.1f}")
        P[idx] = 0
        P[np.ix_(idx, idx)] = np.tile(w, (idx.size, 1))
    ng_f = P @ ng
    Cng_f = P @ Cng @ P.T
    return Calibracion(
        ng=ng_f,
        u_rel=np.sqrt(np.maximum(np.diag(Cng_f), 0)) / ng_f,
        sigma_a=np.sqrt(np.diag(Ca)),
        grupos=grupos,
        banderas=banderas,
        m_repeticiones=m,
        cov_a=Ca,
        cov_ng=Cng_f,
        metadatos_referencia={
            "d_ref_um": d_ref.tolist(),
            "sigma_d_ref_um": sd.tolist(),
            "cov_d_ref_um2": Cd.tolist(),
            "incluida_en_cov_ng": True,
        },
    )


@dataclass
class ValidacionCruzada:
    z: np.ndarray
    ok: bool
    d_estimado: np.ndarray
    sigma_d: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None


def validar_con_muestra(
    cal: Calibracion,
    resultados: Sequence[ResultadoPicos],
    d_ref2: Sequence[float],
    sigma_d_ref2: Sequence[float],
    z_max: float = 3.0,
) -> ValidacionCruzada:
    if not cal.ok:
        raise ErrorDeEntrada("no se puede validar una calibracion con banderas")
    d_ref2 = _vector(d_ref2, "d_ref2", positivo=True)
    sd2 = _vector(sigma_d_ref2, "sigma_d_ref2")
    if d_ref2.size != cal.ng.size or sd2.shape != d_ref2.shape:
        raise ErrorDeEntrada("la segunda referencia debe tener una entrada por capa calibrada")
    if not np.isfinite(z_max) or z_max <= 0:
        raise ErrorDeEntrada("z_max debe ser positivo y finito")
    A = _apilar_caminos(resultados, d_ref2.size)
    m = len(A)
    var_a = np.maximum(
        cal.sigma_a**2, np.mean([r.sigma_caminos() ** 2 for r in resultados], axis=0)
    )
    if m > 1:
        var_a = np.maximum(var_a, A.var(axis=0, ddof=1))
    a_med = A.mean(axis=0)
    d_est = a_med / cal.ng
    var_d = var_a / m / cal.ng**2 + (d_est * cal.u_rel) ** 2
    combinado = np.sqrt(var_d + sd2**2)
    error = d_est - d_ref2
    z = np.zeros_like(error)
    np.divide(error, combinado, out=z, where=combinado > 0)
    exacto_mal = (combinado == 0) & (np.abs(error) > 1e-12 * np.maximum(d_ref2, 1))
    z[exacto_mal] = np.copysign(np.inf, error[exacto_mal])
    return ValidacionCruzada(
        z=z, ok=bool(np.all(np.abs(z) < z_max)), d_estimado=d_est, sigma_d=combinado, error=error
    )
