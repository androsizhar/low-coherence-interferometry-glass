from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from motor.geometria.cad import Malla, Transformacion, _rodrigues, malla_desde_grilla
from motor.medicion.lectura import Instrumento, Medicion
from motor.medicion.materiales import BK7, PVB_SUPUESTO, Cauchy, Desplazado
from motor.medicion.modelo_directo import Capa
from motor.medicion.simulador import fuente_gauss, interferograma, rejilla_k

UM = 1e3
LAM0 = 1.55
INSTR = Instrumento(lam0=LAM0, offset=10000.0, sigma_ruido=0.01)
STANDOFF = 20000.0
D_NOM = np.array([2100.0, 760.0, 1600.0])
GRADIENTE_NOMINAL = np.array([[0.0, 0.0], [0.35, 0.0], [0.0, 0.0]])
VIDRIO_VERDADERO = Desplazado(BK7, 0.006)
PVB_VERDADERO = Cauchy(A=1.4715, B=0.0060, nombre="pvb_verdadero")
MATERIALES_VERDADEROS = [VIDRIO_VERDADERO, PVB_VERDADERO, VIDRIO_VERDADERO]
MATERIALES_APRIORI = [BK7, PVB_SUPUESTO, BK7]
R1, R2 = 1500 * UM, 2500 * UM


def transformacion_montaje():
    return Transformacion(
        _rodrigues(np.array([0.4e-3, -0.3e-3, 0.5e-3])), np.array([40.0, -25.0, 120.0])
    )


def z_cad(x, y):
    return x**2 / (2 * R1) + y**2 / (2 * R2)


def normal_ext(x, y):
    n = np.stack([-x / R1, -y / R2, np.ones_like(x)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def malla_cad(n: int = 61, semi: float = 60 * UM) -> Malla:
    xs = np.linspace(-semi, semi, n)
    X, Y = np.meshgrid(xs, xs)
    return malla_desde_grilla(X, Y, z_cad(X, Y))


def espesores_verdaderos(x_um, y_um) -> np.ndarray:
    x, y = np.asarray(x_um) / UM, np.asarray(y_um) / UM
    d1 = 2100 + 3.0 * np.sin(2 * np.pi * x / 60) * np.cos(2 * np.pi * y / 80)
    d2 = 760 + 0.35 * x + 2.0 * np.cos(2 * np.pi * (x + y) / 90)
    d3 = 1600 + 2.0 * np.cos(2 * np.pi * y / 70)
    return np.stack([d1, d2, d3], axis=-1)


def delta_frente_verdadero(x_um, y_um):
    x, y = np.asarray(x_um) / UM, np.asarray(y_um) / UM
    return 12.0 * np.exp(-(((x - 20) ** 2 + (y + 15) ** 2) / 25.0**2))


def _gauss(x, y, x0, y0, radio):
    return np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / radio**2))


def espesores_conforme(x_um, y_um) -> np.ndarray:
    x, y = np.asarray(x_um) / UM, np.asarray(y_um) / UM
    borde = (np.hypot(x, y) / 45.0) ** 4
    d1 = 2100 + 2.0 * x / 45 - 1.5 * (y / 45) ** 2 + 1.0 * np.sin(2 * np.pi * x / 35 + 0.6)
    d2 = 760 + 0.35 * x + 0.5 * np.cos(2 * np.pi * (x - y) / 70) - 0.6 * borde
    d3 = 1600 - 1.0 * x / 45 + 1.2 * (y / 45) ** 2 + 0.8 * np.sin(2 * np.pi * y / 30)
    return np.stack([d1, d2, d3], axis=-1)


def delta_conforme(x_um, y_um):
    x, y = np.asarray(x_um) / UM, np.asarray(y_um) / UM
    return 6.0 * _gauss(x, y, -15, 10, 30) - 3.0 * _gauss(x, y, 25, -20, 20)


def espesores_defecto(x_um, y_um) -> np.ndarray:
    x, y = np.asarray(x_um) / UM, np.asarray(y_um) / UM
    d = espesores_conforme(x_um, y_um)
    d[..., 0] += 14.0 * _gauss(x, y, -25, 20, 12)
    d[..., 1] += -14.0 * _gauss(x, y, 18, -12, 9) + 0.15 * x * (x > 0)
    return d


def delta_defecto(x_um, y_um):
    x, y = np.asarray(x_um) / UM, np.asarray(y_um) / UM
    return delta_conforme(x_um, y_um) + 18.0 * _gauss(x, y, -25, 20, 10)


VARIANTES = {
    "base": (espesores_verdaderos, delta_frente_verdadero),
    "conforme": (espesores_conforme, delta_conforme),
    "defecto": (espesores_defecto, delta_defecto),
}


def _capas(d):
    return [Capa(m, float(x)) for m, x in zip(MATERIALES_VERDADEROS, d)]


@dataclass
class Escenario:
    mediciones: List[Medicion]
    T_verdadera: Transformacion
    verdad_d: np.ndarray
    verdad_delta: np.ndarray
    xy_cad: np.ndarray
    k: np.ndarray


def _interferograma_punto(d, s, k, S, ruido, rng, theta0):
    return interferograma(_capas(d), k, S, INSTR.offset + s, ruido=ruido, rng=rng, theta0=theta0)


def generar_escenario(
    n_lado: int = 13,
    ruido: float = 0.01,
    semilla: int = 0,
    semi_escaneo: float = 45 * UM,
    N: int = 2**14,
    variante: str = "base",
) -> Escenario:
    perfil_espesores, perfil_delta = VARIANTES[variante]
    rng = np.random.default_rng(semilla)
    k = rejilla_k(N=N)
    S = fuente_gauss(k, LAM0)
    xs = np.linspace(-semi_escaneo, semi_escaneo, n_lado)
    X, Y = np.meshgrid(xs, xs)
    x, y = X.ravel(), Y.ravel()
    Q = np.column_stack([x, y, z_cad(x, y)])
    n_out = normal_ext(x, y)
    d_true = perfil_espesores(x, y)
    delta0 = perfil_delta(x, y)
    T = transformacion_montaje()
    tips_cad = Q + n_out * STANDOFF
    meds: List[Medicion] = []
    D_cum = np.concatenate([np.zeros((len(x), 1)), np.cumsum(d_true, axis=1)], axis=1)
    verdad_delta = np.zeros((4, len(x)))
    for i in range(len(x)):
        s = STANDOFF - delta0[i]
        theta0 = abs(rng.normal(0.0, 0.5e-3))
        I = _interferograma_punto(d_true[i], s, k, S, ruido, rng, theta0)
        pos_s = (tips_cad[i] - T.t) @ T.R
        nin_s = (-n_out[i]) @ T.R
        meds.append(Medicion(k=k, I=I, posicion=pos_s, normal=nin_s / np.linalg.norm(nin_s), id=i))
    for kk in range(4):
        dnom = D_NOM + np.column_stack([x, y]) / UM @ GRADIENTE_NOMINAL.T
        nominal_cum = np.concatenate([np.zeros((len(x), 1)), np.cumsum(dnom, axis=1)], axis=1)
        verdad_delta[kk] = delta0 - (D_cum[:, kk] - nominal_cum[:, kk])
    return Escenario(
        mediciones=meds,
        T_verdadera=T,
        verdad_d=d_true,
        verdad_delta=verdad_delta,
        xy_cad=np.column_stack([x, y]),
        k=k,
    )


def muestra_referencia(d_ref, m: int, ruido: float, semilla: int):
    rng = np.random.default_rng(semilla)
    k = rejilla_k(N=2**14)
    S = fuente_gauss(k, LAM0)
    return k, [
        _interferograma_punto(np.asarray(d_ref), STANDOFF, k, S, ruido, rng, 0.0) for _ in range(m)
    ]


def guardar_escenario(
    carpeta: str, ruido: float = 0.01, semilla: int = 0, n_lado: int = 13, variante: str = "base"
) -> str:
    import json
    import os

    from motor.geometria.cad import escribir_stl

    os.makedirs(carpeta, exist_ok=True)
    esc = generar_escenario(n_lado=n_lado, ruido=ruido, semilla=semilla, variante=variante)
    np.savez(
        os.path.join(carpeta, "mediciones.npz"),
        k=esc.k,
        I=np.array([m.I for m in esc.mediciones]),
        posicion=np.array([m.posicion for m in esc.mediciones]),
        normal=np.array([m.normal for m in esc.mediciones]),
        id=np.array([m.id for m in esc.mediciones]),
    )
    d_ref, sd = np.array([2100.0, 760.0, 1600.0]), np.array([1.0, 1.0, 1.0])
    k, Is = muestra_referencia(d_ref, 30, ruido, semilla + 1000)
    np.savez(
        os.path.join(carpeta, "calibracion.npz"), k=k, I=np.array(Is), d_ref=d_ref, sigma_d_ref=sd
    )
    d2 = np.array([2050.0, 800.0, 1650.0])
    k2, Is2 = muestra_referencia(d2, 10, ruido, semilla + 2000)
    np.savez(
        os.path.join(carpeta, "validacion.npz"), k=k2, I=np.array(Is2), d_ref=d2, sigma_d_ref=sd
    )
    escribir_stl(os.path.join(carpeta, "cad.stl"), malla_cad())
    config = {
        "origen_datos": "simulado",
        "mediciones": "mediciones.npz",
        "calibracion": {
            "archivo": "calibracion.npz",
            "validacion": "validacion.npz",
            "muestra_id": "sintetica-ref-1",
            "validacion_muestra_id": "sintetica-ref-2",
        },
        "cad": "cad.stl",
        "instrumento": {"lam0": LAM0, "offset": INSTR.offset},
        "capas": {
            "d_nom": D_NOM.tolist(),
            "tol_d": [10.0, 25.0, 10.0],
            "ng_prior": [float(m.n_grupo(LAM0)) for m in MATERIALES_APRIORI],
            "grupos": ["vidrio", "pvb", "vidrio"],
            "nombres": ["vidrio_ext", "pvb", "vidrio_int"],
            "idx_pvb": 1,
            "gradiente_nominal": GRADIENTE_NOMINAL.tolist(),
            "origen_nominal_xy": [0, 0],
        },
        "cuna": {"w_nom": 0.35, "tol": 0.10},
        "rango_frontal": [STANDOFF - 1500.0, STANDOFF + 1500.0],
        "prior_montaje": [150.0, 1e-3],
        "registro": {
            "modo": "fijo",
            "R": transformacion_montaje().R.tolist(),
            "t": transformacion_montaje().t.tolist(),
            "covarianza": np.zeros((6, 6)).tolist(),
            "fuente": "pose exacta del escenario sintetico",
        },
        "tol_delta": [15.0] * 4,
        "direccion_cuna": [1, 0],
        "materiales_apriori": ["BK7", "PVB_SUPUESTO", "BK7"],
        "hud": {"theta_i_deg": 62.0, "Rp": 600000.0, "Rv": 900000.0},
        "salida": "salida",
    }
    ruta = os.path.join(carpeta, "config.json")
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return ruta
