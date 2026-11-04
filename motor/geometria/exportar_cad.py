from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from motor.geometria.cad import Malla, escribir_stl
from motor.medicion.lectura import ErrorDeEntrada


def _vertices_grilla(S: np.ndarray) -> np.ndarray:
    return np.asarray(S, float).reshape(-1, 3)


def _caras_grilla(forma, mascara=None):
    ny, nx = forma
    if min(ny, nx) < 2:
        raise ErrorDeEntrada("superficie debe tener ny, nx >= 2")
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    if mascara is not None:
        mascara = np.asarray(mascara)
        if mascara.shape != (ny, nx) or mascara.dtype != np.bool_:
            raise ErrorDeEntrada("mascara debe ser booleana y compartir la grilla (ny,nx)")
        f = f[np.all(mascara.ravel()[f], axis=1)]
    if not len(f):
        raise ErrorDeEntrada("el soporte medido no contiene triangulos completos para CAD")
    return f


def _compactar(v, f):
    usados, inv = np.unique(f, return_inverse=True)
    return Malla(v[usados], inv.reshape(-1, 3))


def malla_superficie(S: np.ndarray, mascara=None) -> Malla:
    S = np.asarray(S, float)
    if S.ndim != 3 or S.shape[2] != 3:
        raise ErrorDeEntrada("superficie debe ser (ny, nx, 3) con ny, nx >= 2")
    f = _caras_grilla(S.shape[:2], mascara)
    return _compactar(_vertices_grilla(S), f)


def solido_capa(S_arriba: np.ndarray, S_abajo: np.ndarray, mascara=None) -> Malla:
    A, B = np.asarray(S_arriba, float), np.asarray(S_abajo, float)
    if A.shape != B.shape or A.ndim != 3 or A.shape[2] != 3:
        raise ErrorDeEntrada("las dos superficies deben tener la misma forma (ny, nx, 3)")
    ny, nx, _ = A.shape
    n = nx * ny
    va, vb = _vertices_grilla(A), _vertices_grilla(B)
    f = _caras_grilla((ny, nx), mascara)
    superior = Malla(va[np.unique(f)], np.searchsorted(np.unique(f), f))
    inferior = Malla(vb[np.unique(f)], np.searchsorted(np.unique(f), f))
    na = superior.normales_cara()
    nb = inferior.normales_cara()
    separacion = np.einsum("tvi,ti->tv", (va - vb)[f], na)
    if np.any(np.einsum("ti,ti->t", na, nb) <= 0):
        raise ErrorDeEntrada("capa: una interfaz esta plegada o invertida respecto de la otra")
    if not (np.all(separacion > 0) or np.all(separacion < 0)):
        raise ErrorDeEntrada("capa: interfaces cruzadas o espesor normal no positivo")
    aristas = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inv, cuentas = np.unique(
        np.sort(aristas, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    p, q = aristas[cuentas[inv] == 1].T
    tris = np.vstack(
        [
            f,
            (f + n)[:, [0, 2, 1]],
            np.column_stack([p, p + n, q + n]),
            np.column_stack([p, q + n, q]),
        ]
    )
    if np.all(separacion < 0):
        tris = tris[:, [0, 2, 1]]
    m = _compactar(np.vstack([va, vb]), tris)
    if not es_cerrada(m) or m.volumen() <= 0:
        raise ErrorDeEntrada("capa: no se pudo formar un solido cerrado con volumen positivo")
    return m


def es_cerrada(m: Malla) -> bool:
    if not len(m.f):
        return False
    e = np.concatenate([m.f[:, [0, 1]], m.f[:, [1, 2]], m.f[:, [2, 0]]])
    dirigidas = {}
    for u, w in e:
        dirigidas[(int(u), int(w))] = dirigidas.get((int(u), int(w)), 0) + 1
    for (u, w), c in dirigidas.items():
        if c != 1 or dirigidas.get((w, u), 0) != 1:
            return False
    enlaces = {}
    for a, b, c in m.f:
        for v, u, w in ((a, b, c), (b, c, a), (c, a, b)):
            grafo = enlaces.setdefault(int(v), {})
            grafo.setdefault(int(u), set()).add(int(w))
            grafo.setdefault(int(w), set()).add(int(u))
    for grafo in enlaces.values():
        if any(len(vecinos) != 2 for vecinos in grafo.values()):
            return False
        vistos, pendientes = set(), [next(iter(grafo))]
        while pendientes:
            v = pendientes.pop()
            if v not in vistos:
                vistos.add(v)
                pendientes.extend(grafo[v] - vistos)
        if len(vistos) != len(grafo):
            return False
    return True


@dataclass
class ReporteExportacion:
    archivos: Dict[str, str]
    volumenes_um3: Dict[str, float]
    cerradas: Dict[str, bool]


def exportar_stl(
    carpeta: str, interfaces: Sequence[np.ndarray], nombres_capas: Sequence[str], mascara=None
) -> ReporteExportacion:
    if len(interfaces) != len(nombres_capas) + 1:
        raise ErrorDeEntrada("se necesitan N+1 interfaces para N capas")
    if not nombres_capas:
        raise ErrorDeEntrada("se necesita al menos una capa")
    for nombre in nombres_capas:
        if (
            not isinstance(nombre, str)
            or not nombre
            or nombre in (".", "..")
            or any(c in nombre for c in ("/", "\\", "\n", "\r", "\x00"))
        ):
            raise ErrorDeEntrada("nombre de capa invalido para archivo CAD")
    superficies = [malla_superficie(S, mascara) for S in interfaces]
    capas = [
        solido_capa(interfaces[j], interfaces[j + 1], mascara) for j in range(len(nombres_capas))
    ]
    for m in superficies + capas:
        with np.errstate(over="ignore", invalid="ignore"):
            Malla(m.v.astype(np.float32).astype(float), m.f)
    os.makedirs(carpeta, exist_ok=True)
    arch, vol, cer = {}, {}, {}
    for k, m in enumerate(superficies):
        ruta = os.path.join(carpeta, f"interfaz_{k}.stl")
        escribir_stl(ruta, m, nombre=f"interfaz_{k}")
        arch[f"interfaz_{k}"] = ruta
    for j, nom in enumerate(nombres_capas):
        m = capas[j]
        ok = es_cerrada(m)
        ruta = os.path.join(carpeta, f"capa_{j + 1}_{nom}.stl")
        escribir_stl(ruta, m, nombre=nom)
        arch[f"capa_{j + 1}_{nom}"] = ruta
        vol[nom + f"_{j + 1}"] = m.volumen()
        cer[nom + f"_{j + 1}"] = ok
    return ReporteExportacion(archivos=arch, volumenes_um3=vol, cerradas=cer)
