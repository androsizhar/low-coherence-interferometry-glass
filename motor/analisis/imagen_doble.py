from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from motor.medicion.lectura import ErrorDeEntrada


def _refractar(d: np.ndarray, n_hacia_incidente: np.ndarray, eta: float) -> np.ndarray:
    c = -float(np.dot(n_hacia_incidente, d))
    rad = 1.0 - eta**2 * (1.0 - c**2)
    if rad < 0:
        raise ErrorDeEntrada("reflexion total interna: geometria no valida")
    return eta * d + (eta * c - np.sqrt(rad)) * n_hacia_incidente


def _reflejar(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    return d - 2.0 * np.dot(d, n) * n


def _cruz(a, b):
    return a[0] * b[1] - a[1] * b[0]


@dataclass
class TrazadoFantasma:
    x_entrada: float
    x_salida: float
    d_salida: np.ndarray
    fallo_en_ojo: float


def _validar(t, n, theta_i, Rp, Rv):
    if not (
        np.all(np.isfinite([t, n, theta_i, Rp, Rv]))
        and t > 0
        and n > 1.0
        and 0 < theta_i < np.pi / 2
        and Rp > 0
        and Rv > 0
    ):
        raise ErrorDeEntrada("imagen doble: t > 0, n > 1, 0 < theta_i < pi/2, Rp > 0, Rv > 0")


def _validar_cuna(w):
    if not np.isfinite(w) or abs(w) >= np.pi / 2:
        raise ErrorDeEntrada("cuna debe ser finita y estar entre -pi/2 y pi/2")


def _geometria(theta_i, Rp, Rv):
    S = np.array([-Rp * np.sin(theta_i), Rp * np.cos(theta_i)])
    E = np.array([Rv * np.sin(theta_i), Rv * np.cos(theta_i)])
    x1 = (S[0] * E[1] + E[0] * S[1]) / (S[1] + E[1])
    d_prim = E - np.array([x1, 0.0])
    d_prim /= np.linalg.norm(d_prim)
    return S, E, x1, d_prim


def _trazar(x_a, S, E, t, w, n):
    tw = np.tan(w)
    d0 = np.array([x_a, 0.0]) - S
    d0 /= np.linalg.norm(d0)
    td = _refractar(d0, np.array([0.0, 1.0]), 1.0 / n)
    if t + x_a * tw <= 0:
        raise ErrorDeEntrada("las caras de la placa se cruzan en la trayectoria")
    den = td[1] + td[0] * tw
    if abs(den) < 1e-14:
        raise ErrorDeEntrada("rayo paralelo a una interfaz")
    s = (-t - x_a * tw) / den
    if s <= 0:
        raise ErrorDeEntrada("interseccion de interfaz detras del rayo")
    H = np.array([x_a, 0.0]) + s * td
    n2 = np.array([tw, 1.0]) / np.sqrt(1.0 + tw**2)
    r = _reflejar(td, n2)
    if r[1] <= 0:
        raise ErrorDeEntrada("rayo reflejado no regresa a la cara interior")
    s2 = -H[1] / r[1]
    Xb = H + s2 * r
    d_out = _refractar(r, np.array([0.0, -1.0]), n)
    return Xb, d_out


def _resolver_hasta_ojo(trazador, S, E, x1, t):
    def f(xa):
        Xb, d_out = trazador(xa)
        if np.dot(d_out, E - Xb) <= 0:
            raise ErrorDeEntrada("el ojo queda detras del rayo de salida")
        return _cruz(d_out, E - Xb)

    for factor in (6.0, 60.0, 600.0):
        xs = np.linspace(x1 - factor * t, x1 + factor * t, 241)
        fs = np.full(xs.size, np.nan)
        for i, x in enumerate(xs):
            try:
                fs[i] = f(x)
            except ErrorDeEntrada:
                pass
        exactos = np.flatnonzero(fs == 0)
        candidatos = [(abs(xs[i] - x1), i, True) for i in exactos]
        cambios = np.flatnonzero(
            np.isfinite(fs[:-1]) & np.isfinite(fs[1:]) & (np.sign(fs[:-1]) * np.sign(fs[1:]) < 0)
        )
        candidatos += [(abs((xs[i] + xs[i + 1]) / 2 - x1), i, False) for i in cambios]
        for _, i, exacto in sorted(candidatos):
            try:
                xa = (
                    xs[i]
                    if exacto
                    else brentq(f, xs[i], xs[i + 1], xtol=1e-10, rtol=1e-14, maxiter=200)
                )
                Xb, d_out = trazador(xa)
                return TrazadoFantasma(
                    float(xa), float(Xb[0]), d_out, abs(float(_cruz(d_out, E - Xb)))
                )
            except ErrorDeEntrada:
                pass
    raise ErrorDeEntrada(
        "no se encontro trayectoria fantasma fisica hacia el ojo (revisa la geometria)"
    )


def trazar_fantasma(
    t: float, w: float, n: float, theta_i: float, Rp: float, Rv: float
) -> TrazadoFantasma:
    _validar(t, n, theta_i, Rp, Rv)
    _validar_cuna(w)
    S, E, x1, _ = _geometria(theta_i, Rp, Rv)
    return _resolver_hasta_ojo(lambda xa: _trazar(xa, S, E, t, w, n), S, E, x1, t)


def separacion_imagen_doble(
    t: float, w: float, n: float, theta_i: float, Rp: float, Rv: float
) -> float:
    _validar(t, n, theta_i, Rp, Rv)
    _, _, _, d_prim = _geometria(theta_i, Rp, Rv)
    tr = trazar_fantasma(t, w, n, theta_i, Rp, Rv)
    return float(np.arctan2(_cruz(d_prim, tr.d_salida), float(np.dot(d_prim, tr.d_salida))))


def separacion_por_cuna_analitica(
    w: float, n: float, theta_i: float, Rp: float = None, Rv: float = None
) -> float:
    _validar(1.0, n, theta_i, 1.0 if Rp is None else Rp, 1.0 if Rv is None else Rv)
    _validar_cuna(w)
    if (Rp is None) != (Rv is None):
        raise ErrorDeEntrada("proporcionar Rp y Rv juntos")
    theta_t = np.arcsin(np.sin(theta_i) / n)
    giro = 2.0 * n * w * np.cos(theta_t) / np.cos(theta_i)
    if Rp is None or Rv is None:
        return giro
    return -giro * Rp / (Rp + Rv)


def cuna_optima(
    t: float, n: float, theta_i: float, Rp: float, Rv: float, w_max: float = 5e-3
) -> float:
    if not np.isfinite(w_max) or not 0 < w_max < np.pi / 2:
        raise ErrorDeEntrada("w_max debe ser finito y estar entre 0 y pi/2")
    f = lambda w: separacion_imagen_doble(t, w, n, theta_i, Rp, Rv)
    a, b = -w_max, w_max
    if f(a) * f(b) > 0:
        raise ErrorDeEntrada("la cuna optima esta fuera de +-w_max")
    return float(brentq(f, a, b, xtol=1e-14, rtol=1e-12))


def _laminado(espesores, indices, w):
    d, n = np.asarray(espesores, dtype=float), np.asarray(indices, dtype=float)
    if (
        d.shape != (3,)
        or n.shape != (3,)
        or not np.all(np.isfinite(d))
        or not np.all(np.isfinite(n))
        or np.any(d <= 0)
        or np.any(n <= 1)
    ):
        raise ErrorDeEntrada("HUD laminado: tres espesores > 0 e indices de fase > 1, finitos")
    _validar_cuna(w)
    profundidades = np.concatenate([[0.0], np.cumsum(d)])
    pendientes = np.array([0.0, 0.0, np.tan(w), np.tan(w)])
    return d, n, profundidades, pendientes


def _interseccion(P, direccion, profundidad, pendiente):
    den = direccion[1] + pendiente * direccion[0]
    if abs(den) < 1e-14:
        raise ErrorDeEntrada("rayo paralelo a una interfaz")
    s = (-profundidad - pendiente * P[0] - P[1]) / den
    if s <= 0:
        raise ErrorDeEntrada("interseccion de interfaz detras del rayo")
    return P + s * direccion


def _trazar_laminado(xa, S, n, profundidad, pendiente):
    P = np.array([xa, 0.0])
    direccion = P - S
    direccion /= np.linalg.norm(direccion)
    normales = np.column_stack([pendiente, np.ones(4)])
    normales /= np.linalg.norm(normales, axis=1)[:, None]

    def comprobar(P):
        locales = profundidad[1:] - profundidad[:-1] + (pendiente[1:] - pendiente[:-1]) * P[0]
        if np.any(locales <= 0):
            raise ErrorDeEntrada("las interfaces se cruzan en la trayectoria")

    comprobar(P)
    direccion = _refractar(direccion, normales[0], 1.0 / n[0])
    for k in (1, 2, 3):
        P = _interseccion(P, direccion, profundidad[k], pendiente[k])
        comprobar(P)
        if k < 3:
            direccion = _refractar(direccion, normales[k], n[k - 1] / n[k])
        else:
            direccion = _reflejar(direccion, normales[k])
    for k in (2, 1, 0):
        P = _interseccion(P, direccion, profundidad[k], pendiente[k])
        comprobar(P)
        n_salida = n[k - 1] if k > 0 else 1.0
        direccion = _refractar(direccion, -normales[k], n[k] / n_salida)
    return P, direccion


def trazar_fantasma_laminado(espesores, indices, w, theta_i, Rp, Rv) -> TrazadoFantasma:
    d, n, profundidad, pendiente = _laminado(espesores, indices, w)
    _validar(float(d.sum()), float(n[0]), theta_i, Rp, Rv)
    S, E, x1, _ = _geometria(theta_i, Rp, Rv)
    return _resolver_hasta_ojo(
        lambda xa: _trazar_laminado(xa, S, n, profundidad, pendiente), S, E, x1, float(d.sum())
    )


def separacion_laminado(espesores, indices, w, theta_i, Rp, Rv) -> float:
    tr = trazar_fantasma_laminado(espesores, indices, w, theta_i, Rp, Rv)
    _, _, _, principal = _geometria(theta_i, Rp, Rv)
    return float(np.arctan2(_cruz(principal, tr.d_salida), np.dot(principal, tr.d_salida)))


def cuna_optima_laminado(espesores, indices, theta_i, Rp, Rv, w_max=5e-3) -> float:
    if not np.isfinite(w_max) or not 0 < w_max < np.pi / 2:
        raise ErrorDeEntrada("w_max debe ser finito y estar entre 0 y pi/2")
    f = lambda w: separacion_laminado(espesores, indices, w, theta_i, Rp, Rv)
    a, b = -w_max, w_max
    if f(a) * f(b) > 0:
        raise ErrorDeEntrada("la cuna optima del laminado esta fuera de +-w_max")
    return float(brentq(f, a, b, xtol=1e-14, rtol=1e-12))
