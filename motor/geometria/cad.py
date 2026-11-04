from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from motor.medicion.lectura import ErrorDeEntrada


@dataclass
class Malla:
    v: np.ndarray
    f: np.ndarray

    def __post_init__(self):
        self.v = np.array(self.v, float, copy=True)
        caras = np.asarray(self.f)
        if not np.issubdtype(caras.dtype, np.integer):
            if not np.all(np.isfinite(caras)) or np.any(caras != np.floor(caras)):
                raise ErrorDeEntrada("malla: los indices de cara deben ser enteros")
        self.f = np.array(caras, int, copy=True)
        if self.v.ndim != 2 or self.v.shape[1] != 3 or self.f.ndim != 2 or self.f.shape[1] != 3:
            raise ErrorDeEntrada("malla: v debe ser (V,3) y f (T,3)")
        if self.f.size and (self.f.min() < 0 or self.f.max() >= len(self.v)):
            raise ErrorDeEntrada("malla: indice de triangulo fuera de rango")
        if not len(self.v) or not len(self.f) or not np.all(np.isfinite(self.v)):
            raise ErrorDeEntrada("malla: se necesitan vertices finitos y al menos un triangulo")
        self.normales_cara()
        self.v.setflags(write=False)
        self.f.setflags(write=False)
        self._arbol = None

    def tri(self):
        return self.v[self.f[:, 0]], self.v[self.f[:, 1]], self.v[self.f[:, 2]]

    def normales_cara(self) -> np.ndarray:
        a, b, c = self.tri()
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        if np.any(nn == 0):
            raise ErrorDeEntrada("malla con triangulos degenerados (area 0)")
        return n / nn

    def areas(self) -> np.ndarray:
        a, b, c = self.tri()
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def volumen(self) -> float:
        a, b, c = self.tri()
        centro = self.v.mean(axis=0)
        return float(
            np.einsum("ij,ij->i", a - centro, np.cross(b - centro, c - centro)).sum() / 6.0
        )

    def normales_vertice(self) -> np.ndarray:
        a, b, c = self.tri()
        nf = np.cross(b - a, c - a)
        nv = np.zeros_like(self.v)
        for i in range(3):
            np.add.at(nv, self.f[:, i], nf)
        nn = np.linalg.norm(nv, axis=1, keepdims=True)
        if np.any(nn == 0):
            raise ErrorDeEntrada(
                "malla: normal de vertice indefinida (caras opuestas o vertice aislado)"
            )
        return nv / np.where(nn == 0, 1, nn)

    def normales_interpoladas(self, Q: np.ndarray, ti: np.ndarray) -> np.ndarray:
        a, b, c = self.tri()
        a, b, c = a[ti], b[ti], c[ti]
        v0, v1, v2 = b - a, c - a, Q - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        den = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        u = 1.0 - v - w
        nv = self.normales_vertice()
        f = self.f[ti]
        nn = u[:, None] * nv[f[:, 0]] + v[:, None] * nv[f[:, 1]] + w[:, None] * nv[f[:, 2]]
        norma = np.linalg.norm(nn, axis=1, keepdims=True)
        if np.any(norma < 1e-15):
            raise ErrorDeEntrada("malla: normal interpolada indefinida")
        return nn / norma

    def _kdtree(self):
        if self._arbol is None:
            a, b, c = self.tri()
            self._centros = (a + b + c) / 3.0
            self._radios = np.maximum.reduce(
                [
                    np.linalg.norm(a - self._centros, axis=1),
                    np.linalg.norm(b - self._centros, axis=1),
                    np.linalg.norm(c - self._centros, axis=1),
                ]
            )
            self._arbol = cKDTree(self._centros)
        return self._arbol

    def proyectar(self, P: np.ndarray, k: int = 12):
        P = np.atleast_2d(np.asarray(P, float))
        if P.ndim != 2 or P.shape[1] != 3 or not np.all(np.isfinite(P)):
            raise ErrorDeEntrada("proyectar: puntos finitos (N,3)")
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ErrorDeEntrada("proyectar: k debe ser un entero positivo")
        if not len(P):
            return P.copy(), np.empty(0), np.empty(0, dtype=int)
        k = min(k, len(self.f))
        _, cand = self._kdtree().query(P, k=k)
        cand = cand.reshape(len(P), k)
        a, b, c = self.tri()
        mejor_d = np.full(len(P), np.inf)
        mejor_q = np.zeros_like(P)
        mejor_t = np.zeros(len(P), dtype=int)
        for j in range(k):
            t = cand[:, j]
            q = _punto_mas_cercano_triangulo(P, a[t], b[t], c[t])
            d = np.linalg.norm(P - q, axis=1)
            mejor = d < mejor_d
            mejor_d = np.where(mejor, d, mejor_d)
            mejor_q[mejor] = q[mejor]
            mejor_t[mejor] = t[mejor]
        rmax = float(self._radios.max())
        for i, p in enumerate(P):
            candidatos = np.asarray(self._arbol.query_ball_point(p, mejor_d[i] + rmax), dtype=int)
            cotas = np.linalg.norm(self._centros[candidatos] - p, axis=1) - self._radios[candidatos]
            candidatos = candidatos[cotas <= mejor_d[i] + 1e-12]
            if not len(candidatos):
                continue
            q = _punto_mas_cercano_triangulo(
                np.broadcast_to(p, (len(candidatos), 3)),
                a[candidatos],
                b[candidatos],
                c[candidatos],
            )
            d = np.linalg.norm(q - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < mejor_d[i]:
                mejor_d[i], mejor_q[i], mejor_t[i] = d[j], q[j], candidatos[j]
        nf = self.normales_cara()[mejor_t]
        signo = np.sign(np.einsum("ij,ij->i", P - mejor_q, nf))
        signo[signo == 0] = 1.0
        return mejor_q, signo * mejor_d, mejor_t

    def muestrear(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        A = self.areas()
        idx = rng.choice(len(A), size=n, p=A / A.sum())
        u, v = rng.random(n), rng.random(n)
        su = np.sqrt(u)
        a, b, c = self.tri()
        return (
            (1 - su)[:, None] * a[idx]
            + (su * (1 - v))[:, None] * b[idx]
            + (su * v)[:, None] * c[idx]
        )


def _punto_mas_cercano_triangulo(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1.0 / (va + vb + vc)
        q_int = a + ab * (vb * den)[:, None] + ac * (vc * den)[:, None]
        q_ab = a + ab * (d1 / (d1 - d3))[:, None]
        q_ac = a + ac * (d2 / (d2 - d6))[:, None]
        q_bc = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]
    conds = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
    ]
    elecciones = [a, b, q_ab, c, q_ac, q_bc]
    out = q_int.copy()
    for comp in range(3):
        out[:, comp] = np.select(conds, [e[:, comp] for e in elecciones], default=q_int[:, comp])
    return out


def malla_desde_grilla(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Malla:
    X, Y, Z = np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)
    if Z.ndim != 2 or X.shape != Z.shape or Y.shape != Z.shape or min(Z.shape) < 2:
        raise ErrorDeEntrada("grilla: X, Y, Z deben compartir forma (ny,nx), ny,nx >= 2")
    ny, nx = Z.shape
    v = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    return Malla(v, f)


def interfaces_nominales(exterior: Malla, d_nom: Sequence[float]) -> List[Malla]:
    d_nom = np.asarray(d_nom, float)
    if (
        d_nom.ndim not in (1, 2)
        or not len(d_nom)
        or (d_nom.ndim == 2 and d_nom.shape[1] != len(exterior.v))
        or not np.all(np.isfinite(d_nom))
        or np.any(d_nom <= 0)
    ):
        raise ErrorDeEntrada("interfaces nominales: espesores finitos y positivos")
    nv = exterior.normales_vertice()
    mallas = [exterior]
    D = 0.0
    for d in d_nom:
        D = D + d
        desplazamiento = float(D) if np.ndim(D) == 0 else D[:, None]
        mallas.append(Malla(exterior.v - nv * desplazamiento, exterior.f.copy()))
    return mallas


@dataclass
class Transformacion:
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.R, self.t = np.asarray(self.R, float), np.asarray(self.t, float)
        if (
            self.R.shape != (3, 3)
            or self.t.shape != (3,)
            or not np.all(np.isfinite(self.R))
            or not np.all(np.isfinite(self.t))
            or not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(self.R), 1.0, atol=1e-8)
        ):
            raise ErrorDeEntrada(
                "transformacion: R debe ser rotacion propia y t un vector finito (3,)"
            )

    def aplicar(self, P):
        return np.asarray(P) @ self.R.T + self.t


@dataclass
class ResultadoRegistro:
    T: Transformacion
    rms: float
    iteraciones: int
    convergio: bool
    cond: float
    banderas: List[str] = field(default_factory=list)
    covarianza: Optional[np.ndarray] = None


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def registrar_icp(
    puntos: np.ndarray,
    malla: Malla,
    T0: Optional[Transformacion] = None,
    max_iter: int = 60,
    recorte: float = 0.9,
    amortiguamiento: float = 1e-9,
    tol: float = 1e-9,
    cond_max: float = 1e8,
    sigma_prior_traslacion: Optional[float] = None,
    sigma_prior_rotacion: Optional[float] = None,
    sigma_ruido: float = 2.0,
) -> ResultadoRegistro:
    P = np.asarray(puntos, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 6 or not np.all(np.isfinite(P)):
        raise ErrorDeEntrada("registro: se necesitan >= 6 puntos (N,3)")
    if (
        not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
        or not np.isfinite(recorte)
        or not 0 < recorte <= 1
        or not np.isfinite(sigma_ruido)
        or sigma_ruido <= 0
        or not np.isfinite(amortiguamiento)
        or amortiguamiento <= 0
        or not np.isfinite(tol)
        or tol <= 0
        or not np.isfinite(cond_max)
        or cond_max <= 1
    ):
        raise ErrorDeEntrada(
            "registro: iteraciones, recorte, ruido, tolerancia o amortiguamiento invalidos"
        )
    for s in (sigma_prior_traslacion, sigma_prior_rotacion):
        if s is not None and (not np.isfinite(s) or s <= 0):
            raise ErrorDeEntrada(
                "registro: las incertidumbres del montaje deben ser finitas y positivas"
            )
    T = T0 or Transformacion(np.eye(3), np.zeros(3))
    R0, t0 = T.R.copy(), T.t.copy()
    Lc = float(np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, axis=1))))
    if Lc <= 0:
        raise ErrorDeEntrada("registro: puntos coincidentes")
    con_prior = sigma_prior_traslacion is not None or sigma_prior_rotacion is not None
    st = sigma_prior_traslacion if sigma_prior_traslacion is not None else np.inf
    sr = (sigma_prior_rotacion if sigma_prior_rotacion is not None else np.inf) * Lc
    Pinv = np.diag([1.0 / sr**2] * 3 + [1.0 / st**2] * 3)
    banderas: List[str] = []
    convergio = False
    cond = np.inf
    it = 0
    keep_fijo = None
    from scipy.spatial.transform import Rotation

    for it in range(1, max_iter + 1):
        X = T.aplicar(P)
        Q, _, ti = malla.proyectar(X)
        n = malla.normales_interpoladas(Q, ti)
        r = np.einsum("ij,ij->i", n, X - Q)
        if it <= 8 or keep_fijo is None:
            keep = np.argsort(np.abs(r))[: max(6, int(recorte * len(r)))]
            if it == 8:
                keep_fijo = keep
        else:
            keep = keep_fijo
        Xk, nk, rk = X[keep], n[keep], r[keep]
        A = np.hstack([np.cross(Xk, nk) / Lc, nk])
        N = A.T @ A
        ev = np.linalg.eigvalsh(N)
        cond = float(ev[-1] / max(ev[0], 1e-300))
        if con_prior:
            Rd = T.R @ R0.T
            x_cur = np.concatenate([Rotation.from_matrix(Rd).as_rotvec() * Lc, T.t - Rd @ t0])
            w2 = 1.0 / sigma_ruido**2
            x = np.linalg.lstsq(N * w2 + Pinv, -(A.T @ rk) * w2 - Pinv @ x_cur, rcond=None)[0]
        else:
            x = np.linalg.solve(N + amortiguamiento * ev[-1] * np.eye(6), -A.T @ rk)
        w, dt = x[:3] / Lc, x[3:]
        dR = _rodrigues(w)
        T = Transformacion(dR @ T.R, dR @ T.t + dt)
        if np.linalg.norm(x) < (
            max(tol * max(Lc, 1.0) * 1e3, 0.05) if con_prior else tol * max(Lc, 1.0)
        ):
            convergio = True
            break
    X = T.aplicar(P)
    Q, d, ti = malla.proyectar(X)
    rms = float(np.sqrt(np.mean(np.sort(d**2)[: max(6, int(recorte * len(d)))])))
    if not convergio:
        banderas.append("icp_no_convergio")
    if cond > cond_max:
        banderas.append(
            ("registro_sostenido_por_el_prior" if con_prior else "mal_condicionado")
            + f":cond={cond:.2e}"
        )
    nk = malla.normales_interpoladas(Q[keep], ti[keep])
    Af = np.hstack([np.cross(X[keep], nk) / Lc, nk])
    precision = Af.T @ Af / sigma_ruido**2 + Pinv
    evp = np.linalg.eigvalsh(precision)
    cov = None
    if evp[0] > 0 and evp[-1] / evp[0] <= cond_max:
        escala = np.diag([1 / Lc] * 3 + [1.0] * 3)
        cov = escala @ np.linalg.inv(precision) @ escala
    else:
        banderas.append("covarianza_registro_no_identificable")
    return ResultadoRegistro(
        T=T,
        rms=rms,
        iteraciones=it,
        convergio=convergio,
        cond=cond,
        banderas=banderas,
        covarianza=cov,
    )


def escribir_stl(ruta: str, malla: Malla, nombre: str = "vitro", ascii_: bool = False):
    if not isinstance(nombre, str) or any(c in nombre for c in ("\n", "\r", "\x00")):
        raise ErrorDeEntrada("STL: nombre invalido")
    if not ascii_:
        with np.errstate(over="ignore", invalid="ignore"):
            cuantizados = malla.v.astype(np.float32).astype(float)
        Malla(cuantizados, malla.f)
    nf = malla.normales_cara()
    a, b, c = malla.tri()
    if ascii_:
        with open(ruta, "w", encoding="utf-8") as fh:
            fh.write(f"solid {nombre}\n")
            for i in range(len(malla.f)):
                fh.write(
                    f"  facet normal {nf[i,0]:.9e} {nf[i,1]:.9e} {nf[i,2]:.9e}\n    outer loop\n"
                )
                for p in (a[i], b[i], c[i]):
                    fh.write(f"      vertex {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}\n")
                fh.write("    endloop\n  endfacet\n")
            fh.write(f"endsolid {nombre}\n")
        return
    with open(ruta, "wb") as fh:
        fh.write(nombre.encode("ascii", "ignore")[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(malla.f)))
        for i in range(len(malla.f)):
            fh.write(struct.pack("<12fH", *nf[i], *a[i], *b[i], *c[i], 0))


def leer_stl(ruta: str) -> Malla:
    with open(ruta, "rb") as fh:
        datos = fh.read()
    n = struct.unpack("<I", datos[80:84])[0] if len(datos) >= 84 else None
    es_binario = n is not None and len(datos) == 84 + 50 * n
    if not es_binario and datos.lstrip().startswith(b"solid"):
        verts = []
        try:
            for linea in datos.decode("utf-8").splitlines():
                t = linea.split()
                if t and t[0] == "vertex":
                    if len(t) != 4:
                        raise ValueError("vertex incompleto")
                    verts.append([float(t[1]), float(t[2]), float(t[3])])
        except (UnicodeError, ValueError) as exc:
            raise ErrorDeEntrada("STL ascii corrupto") from exc
        V = np.array(verts)
        if not len(V) or len(V) % 3:
            raise ErrorDeEntrada("STL ascii corrupto")
        return _soldar(V.reshape(-1, 3, 3))
    if not es_binario:
        raise ErrorDeEntrada("STL binario corrupto (tamano no coincide)")
    arr = np.frombuffer(
        datos[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n
    )
    return _soldar(arr["v"].astype(float))


def _soldar(tri: np.ndarray) -> Malla:
    flat = tri.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    return Malla(uniq, inv.reshape(-1, 3))
