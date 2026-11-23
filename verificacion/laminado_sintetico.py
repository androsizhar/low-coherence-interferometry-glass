"""Laminado curvo sintético de geometría conocida para probar el importador de CAD.

La verdad se define como en el motor: cada interfaz es la cara frontal desplazada a lo
largo de su normal por la suma de espesores. Todas las longitudes en µm.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from motor.geometria.cad import Malla, escribir_stl

R1, R2 = 1.5e6, 3.0e6
SEMI_X, SEMI_Y = 150e3, 100e3


def z_frontal(x, y):
    return -(x**2) / (2 * R1) - y**2 / (2 * R2)


def normal_frontal(x, y):
    n = np.stack([x / R1, y / R2, np.ones_like(x)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def espesores(x, y, cuna_mrad=0.5):
    x = np.asarray(x, float)
    return np.stack([2100 + 0 * x, 760 + cuna_mrad * 1e-3 * (x + SEMI_X), 1600 + 0 * x], axis=0)


def superficies(n_x=61, n_y=41, cuna_mrad=0.5):
    xs = np.linspace(-SEMI_X, SEMI_X, n_x)
    ys = np.linspace(-SEMI_Y, SEMI_Y, n_y)
    X, Y = np.meshgrid(xs, ys)
    S0 = np.stack([X, Y, z_frontal(X, Y)], axis=-1)
    n = normal_frontal(X, Y)
    D = np.concatenate([np.zeros((1, *X.shape)), np.cumsum(espesores(X, Y, cuna_mrad), axis=0)])
    return [S0 - D[k][..., None] * n for k in range(4)]


def _solido(arriba, abajo):
    ny, nx, _ = arriba.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tapa = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    off = nx * ny
    base = tapa[:, ::-1] + off
    borde = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    s = []
    for i in range(len(borde)):
        p, q = borde[i], borde[(i + 1) % len(borde)]
        s += [[p, q + off, q], [p, p + off, q + off]]
    V = np.concatenate([arriba.reshape(-1, 3), abajo.reshape(-1, 3)])
    F = np.concatenate([tapa, base, np.array(s)])
    m = Malla(V, F)
    if m.volumen() < 0:
        m = Malla(V, F[:, ::-1])
    return m


def pose():
    return Rotation.from_rotvec(np.array([0.5, 0.2, -0.3])).as_matrix(), np.array(
        [1.2e6, -3.0e5, 9.0e5]
    )


def mallas_capas(cuna_mrad=0.5, con_pose=True):
    S = superficies(cuna_mrad=cuna_mrad)
    R, t = pose() if con_pose else (np.eye(3), np.zeros(3))
    capas = [_solido(S[k], S[k + 1]) for k in range(3)]
    return [Malla(m.v @ R.T + t, m.f) for m in capas]


def zona_rectangular(x0, x1, y0, y1, interfaz, n=12, cuna_mrad=0.5):
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)
    n0 = normal_frontal(X, Y)
    D = np.concatenate([np.zeros((1, *X.shape)), np.cumsum(espesores(X, Y, cuna_mrad), axis=0)])
    P = np.stack([X, Y, z_frontal(X, Y)], axis=-1) - D[interfaz][..., None] * n0
    ny, nx = X.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    F = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    R, t = pose()
    return Malla(P.reshape(-1, 3) @ R.T + t, F)


def escribir(carpeta, cuna_mrad=0.5):
    import os

    rutas = []
    for j, m in enumerate(mallas_capas(cuna_mrad)):
        r = os.path.join(carpeta, f"capa_{['vidrio_ext', 'intercapa', 'vidrio_int'][j]}.stl")
        escribir_stl(r, m, ascii_=True)
        rutas.append(r)
    zona = os.path.join(carpeta, "zona_hud.stl")
    escribir_stl(
        zona, zona_rectangular(20e3, 80e3, -50e3, 10e3, 3, cuna_mrad=cuna_mrad), ascii_=True
    )
    return rutas[::-1], {"HUD": zona}
