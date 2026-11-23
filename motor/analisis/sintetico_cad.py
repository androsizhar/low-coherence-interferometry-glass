"""Pieza sintética sobre la geometría de cualquier CAD importado.

Sirve para validar la cadena completa (CAD → señales → análisis) con la forma, las
capas y la cuña reales de una pieza nueva antes de tener mediciones. La verdad usada
se guarda en verdad.npz para compararla con el resultado.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from motor.geometria.cad import leer_stl
from motor.geometria.importar_cad import cargar_modelo, zonas_por_nombre
from motor.geometria.soporte import dentro_region
from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.materiales import CATALOGO, Desplazado
from motor.medicion.modelo_directo import Capa
from motor.medicion.simulador import fuente_gauss, interferograma, rejilla_k

LAM0, OFFSET, STANDOFF = 1.55, 10000.0, 20000.0


def _gauss(u, v, u0, v0, r):
    return np.exp(-(((u - u0) ** 2 + (v - v0) ** 2) / r**2))


def _perturbacion(u, v, N, idx_cuna, variante):
    dd = np.stack(
        [
            1.5 * np.sin(2 * np.pi * (1.3 * u + 0.2 * j)) * np.cos(2 * np.pi * 0.9 * v)
            for j in range(N)
        ],
        axis=-1,
    )
    delta = 5.0 * _gauss(u, v, -0.15, 0.1, 0.25)
    if variante == "defecto":
        dd[:, 0] += 12.0 * _gauss(u, v, -0.2, 0.15, 0.07)
        dd[:, idx_cuna] += -14.0 * _gauss(u, v, 0.15, -0.1, 0.08)
        delta += 15.0 * _gauss(u, v, -0.2, 0.15, 0.07)
    return dd, delta


def generar_desde_cad(
    ruta_modelo: str,
    carpeta: str,
    n_lado: int = 15,
    ruido: float = 0.01,
    semilla: int = 0,
    materiales: Sequence[str] | None = None,
    variante: str = "conforme",
    zona: str | None = None,
) -> str:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import binary_erosion, distance_transform_edt

    from motor.analisis.pipeline import evaluar_malla_en_xy

    if variante not in ("conforme", "defecto"):
        raise ErrorDeEntrada("variante debe ser conforme o defecto")
    meta = cargar_modelo(ruta_modelo)
    base = Path(meta["_base"])
    capas = meta["capas"]
    N = len(capas)
    sug = meta.get("sugerencias", {})
    idx_cuna = sug.get("idx_cuna")
    if materiales is None:
        if N != 3:
            raise ErrorDeEntrada(
                f"el CAD tiene {N} capas: indica --materiales con un nombre por capa de {list(CATALOGO)}"
            )
        materiales = ["BK7", "PVB_SUPUESTO", "BK7"]
    materiales = list(materiales)
    if len(materiales) != N or any(m not in CATALOGO for m in materiales):
        raise ErrorDeEntrada(f"materiales: uno por capa ({N}) y de {list(CATALOGO)}")
    apriori = [CATALOGO[m] for m in materiales]
    verdaderos = [Desplazado(m, 0.004) for m in apriori]
    with np.load(base / meta["archivos"]["nominal"], allow_pickle=False) as z:
        X, Y, D, mascara = z["X"], z["Y"], z["d_nom"], z["mascara"]
    malla = leer_stl(base / meta["archivos"]["cad"])
    xs, ys = X[0], Y[:, 0]
    util = binary_erosion(mascara, iterations=2)
    if zona is not None:
        zonas = zonas_por_nombre(meta)
        if zona not in zonas:
            raise ErrorDeEntrada(f"zona '{zona}' no existe; hay {sorted(zonas)}")
        util &= dentro_region(np.stack([X, Y], axis=-1), zonas[zona])
    if util.sum() < 20:
        raise ErrorDeEntrada("la región útil es demasiado pequeña para un mapa (≥ 20 nodos)")
    x0, x1 = X[util].min(), X[util].max()
    y0, y1 = Y[util].min(), Y[util].max()
    PX, PY = np.meshgrid(np.linspace(x0, x1, n_lado), np.linspace(y0, y1, n_lado))
    dentro = (
        RegularGridInterpolator((ys, xs), util.astype(float), method="nearest")(
            np.column_stack([PY.ravel(), PX.ravel()])
        )
        > 0.5
    )
    x, y = PX.ravel()[dentro], PY.ravel()[dentro]
    if len(x) < 20:
        raise ErrorDeEntrada(f"solo {len(x)} puntos dentro de la pieza: aumenta n_lado")
    _, (iy, ix) = distance_transform_edt(~mascara, return_indices=True)
    Dl = D[:, iy, ix]
    d_nom = RegularGridInterpolator((ys, xs), np.moveaxis(Dl, 0, -1))(np.column_stack([y, x]))
    Z, n_out = evaluar_malla_en_xy(malla, x, y)
    if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(n_out))):
        raise ErrorDeEntrada("puntos fuera de la superficie frontal")
    L = max(x1 - x0, y1 - y0)
    u, v = (x - 0.5 * (x0 + x1)) / L, (y - 0.5 * (y0 + y1)) / L
    dd, delta = _perturbacion(u, v, N, 0 if idx_cuna is None else int(idx_cuna), variante)
    d_true = d_nom + dd
    rng = np.random.default_rng(semilla)
    k = rejilla_k(N=2**14)
    S = fuente_gauss(k, LAM0)
    Q = np.column_stack([x, y, Z])
    M_R, M_t = np.asarray(meta["marco"]["R"]), np.asarray(meta["marco"]["t"])
    puntas_loc = Q + n_out * STANDOFF
    posicion = (puntas_loc - M_t) @ M_R
    normal = (-n_out) @ M_R
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    I = []
    for i in range(len(x)):
        cs = [Capa(m, float(t)) for m, t in zip(verdaderos, d_true[i])]
        theta0 = abs(rng.normal(0.0, 0.5e-3))
        I.append(
            interferograma(
                cs, k, S, OFFSET + STANDOFF - delta[i], ruido=ruido, rng=rng, theta0=theta0
            )
        )
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    np.savez(
        carpeta / "mediciones.npz",
        k=k,
        I=np.array(I),
        posicion=posicion,
        normal=normal,
        id=np.arange(len(x)),
    )
    np.savez(
        carpeta / "verdad.npz",
        x=x,
        y=y,
        d_true=d_true,
        d_nom=d_nom,
        delta_frontal=delta,
    )
    d_ref = np.round(Dl[:, mascara].mean(axis=1), 1)
    for nombre, dref, m, sem in (
        ("calibracion.npz", d_ref, 30, semilla + 1000),
        ("validacion.npz", np.round(d_ref * 1.03, 1), 10, semilla + 2000),
    ):
        rr = np.random.default_rng(sem)
        cs = [Capa(mt, float(t)) for mt, t in zip(verdaderos, dref)]
        Is = [interferograma(cs, k, S, OFFSET + STANDOFF, ruido=ruido, rng=rr) for _ in range(m)]
        np.savez(
            carpeta / nombre,
            k=k,
            I=np.array(Is),
            d_ref=dref,
            sigma_d_ref=np.ones(N),
        )
    rel = Path(os.path.relpath(base, carpeta))  # pathlib no tiene relpath con ".." antes de 3.12
    tol_d = [10.0] * N
    if idx_cuna is not None:
        tol_d[int(idx_cuna)] = 25.0
    config = {
        "origen_datos": "simulado",
        "mediciones": "mediciones.npz",
        "calibracion": {
            "archivo": "calibracion.npz",
            "validacion": "validacion.npz",
            "muestra_id": "sintetica-ref-1",
            "validacion_muestra_id": "sintetica-ref-2",
        },
        "cad": (rel / meta["archivos"]["cad"]).as_posix(),
        "cad_importado": (rel / "cad_importado.json").as_posix(),
        "instrumento": {"lam0": LAM0, "offset": OFFSET},
        "capas": {
            "nombres": capas,
            "d_nom_mapa": (rel / meta["archivos"]["nominal"]).as_posix(),
            "tol_d": tol_d,
            "ng_prior": [float(m.n_grupo(LAM0)) for m in apriori],
            "grupos": materiales,
            "idx_pvb": 0 if idx_cuna is None else int(idx_cuna),
        },
        "cuna": {"w_nom": "cad" if idx_cuna is not None else 0.0, "tol": 0.10},
        "direccion_cuna": "cad" if idx_cuna is not None else [1, 0],
        "rango_frontal": [STANDOFF - 1500.0, STANDOFF + 1500.0],
        "registro": {
            "modo": "fijo",
            "R": np.eye(3).tolist(),
            "t": [0.0, 0.0, 0.0],
            "covarianza": np.zeros((6, 6)).tolist(),
            "fuente": "pose exacta del escenario sintetico (sensor en el marco del CAD original)",
        },
        "tol_delta": [15.0] * (N + 1),
        "materiales_apriori": materiales,
        "sigma_pos": 3.0,
        "grilla": [31, 31],
        "semilla": int(semilla),
        "salida": "salida",
    }
    ruta = carpeta / "config.json"
    ruta.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(ruta)


def comparar_con_verdad(carpeta: str) -> dict:
    """Error del análisis respecto a la verdad sintética, en los puntos medidos."""
    carpeta = Path(carpeta)
    with np.load(carpeta / "verdad.npz") as v:
        x, y, d_true = v["x"], v["y"], v["d_true"]
    with np.load(carpeta / "salida" / "mapas.npz") as m:
        X, Y, d = m["X"], m["Y"], m["d_media"]
    from scipy.interpolate import RegularGridInterpolator

    out = {}
    for j in range(d.shape[0]):
        f = RegularGridInterpolator((Y[:, 0], X[0]), d[j], bounds_error=False, fill_value=np.nan)
        e = f(np.column_stack([y, x])) - d_true[:, j]
        e = e[np.isfinite(e)]
        out[j] = {"rms_um": float(np.sqrt(np.mean(e**2))), "max_um": float(np.abs(e).max())}
    return out
