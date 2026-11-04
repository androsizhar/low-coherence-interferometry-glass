from __future__ import annotations

import numpy as np

from motor.medicion.lectura import ErrorDeEntrada


def dentro_poligono(xy, poligono):
    xy = np.asarray(xy, float)
    p = np.asarray(poligono, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.all(np.isfinite(p)):
        raise ErrorDeEntrada("polígono: se requieren al menos tres vértices [x,y] finitos en µm")
    if np.linalg.matrix_rank(p - p.mean(0)) < 2:
        raise ErrorDeEntrada("polígono: los vértices no pueden ser colineales")
    x, y = xy[..., 0], xy[..., 1]
    interior = np.zeros(x.shape, bool)
    borde = np.zeros(x.shape, bool)
    escala = max(float(np.ptp(p, axis=0).max()), 1.0)
    eps = 1e-10 * escala
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        dx, dy = b - a
        cruz = (x - a[0]) * dy - (y - a[1]) * dx
        borde |= (
            (np.abs(cruz) <= eps * max(float(np.hypot(dx, dy)), 1.0))
            & (x >= min(a[0], b[0]) - eps)
            & (x <= max(a[0], b[0]) + eps)
            & (y >= min(a[1], b[1]) - eps)
            & (y <= max(a[1], b[1]) + eps)
        )
        if dy != 0:
            interior ^= ((a[1] > y) != (b[1] > y)) & (x < a[0] + (y - a[1]) * dx / dy)
    return interior | borde


def cargar_especificacion(especificacion, base):
    if especificacion is None:
        return None
    if not isinstance(especificacion, dict):
        raise ErrorDeEntrada(
            "mascara_adquisicion debe ser un objeto con archivo, incluir y/o excluir"
        )
    out = dict(especificacion)
    if out.get("archivo"):
        from pathlib import Path

        ruta = Path(out["archivo"])
        if not ruta.is_absolute():
            ruta = Path(base) / ruta
        try:
            with np.load(ruta, allow_pickle=False) as z:
                clave = out.get("clave", "mascara")
                if clave not in z:
                    raise ErrorDeEntrada(f"máscara NPZ: falta la clave '{clave}'")
                out["mascara"] = z[clave]
                if "X" in z and "Y" in z:
                    out["X"], out["Y"] = z["X"], z["Y"]
        except (OSError, ValueError) as exc:
            raise ErrorDeEntrada(
                f"No se puede cargar la máscara de adquisición {ruta}: {exc}"
            ) from exc
    return out


def _celdas_cruzadas(X, Y, poligono):
    p = np.asarray(poligono, float)
    xmin = np.minimum(X[:-1, :-1], X[1:, 1:])
    xmax = np.maximum(X[:-1, :-1], X[1:, 1:])
    ymin = np.minimum(Y[:-1, :-1], Y[1:, 1:])
    ymax = np.maximum(Y[:-1, :-1], Y[1:, 1:])
    eps = 1e-10 * max(float(np.ptp(X)), float(np.ptp(Y)), 1.0)
    xmin, xmax, ymin, ymax = xmin + eps, xmax - eps, ymin + eps, ymax - eps
    cruza = np.zeros(xmin.shape, bool)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        t0, t1 = np.zeros(xmin.shape), np.ones(xmin.shape)
        valido = np.ones(xmin.shape, bool)
        for aa, dd, lo, hi in ((a[0], b[0] - a[0], xmin, xmax), (a[1], b[1] - a[1], ymin, ymax)):
            if abs(dd) < 1e-15:
                valido &= (aa >= lo) & (aa <= hi)
            else:
                u, v = (lo - aa) / dd, (hi - aa) / dd
                t0 = np.maximum(t0, np.minimum(u, v))
                t1 = np.minimum(t1, np.maximum(u, v))
        cruza |= valido & (t0 <= t1)
    return cruza


def _quitar_celdas_no_resueltas(mascara, cruzadas):
    completas = mascara[:-1, :-1] & mascara[1:, :-1] & mascara[:-1, 1:] & mascara[1:, 1:]
    veto = cruzadas & completas
    for vista in (mascara[:-1, :-1], mascara[1:, :-1], mascara[:-1, 1:], mascara[1:, 1:]):
        vista[veto] = False
    return mascara


def aplicar_mascara(X, Y, soporte_convexo, especificacion=None):
    xy = np.stack([X, Y], axis=-1)
    mascara = np.asarray(soporte_convexo, bool).copy()
    if especificacion is None:
        return mascara
    if not isinstance(especificacion, dict):
        raise ErrorDeEntrada("mascara_adquisicion debe ser un objeto")
    if "mascara" in especificacion:
        a = np.asarray(especificacion["mascara"])
        if a.dtype.kind != "b" or a.shape != X.shape:
            raise ErrorDeEntrada(
                f"máscara de adquisición: matriz booleana de forma {X.shape}, sin remuestreo implícito"
            )
        if "X" in especificacion or "Y" in especificacion:
            if not all(
                k in especificacion
                and np.shape(especificacion[k]) == X.shape
                and np.allclose(especificacion[k], ref, atol=1e-6, rtol=0)
                for k, ref in (("X", X), ("Y", Y))
            ):
                raise ErrorDeEntrada(
                    "X/Y de la máscara no coinciden con la rejilla CAD en µm; corrige la rejilla o la máscara"
                )
        mascara &= a
    incluir = especificacion.get("incluir", [])
    if incluir:
        mascara &= np.logical_or.reduce([dentro_poligono(xy, p) for p in incluir])
    for p in especificacion.get("excluir", []):
        mascara &= ~dentro_poligono(xy, p)
        mascara = _quitar_celdas_no_resueltas(mascara, _celdas_cruzadas(X, Y, p))
    for p in incluir:
        mascara = _quitar_celdas_no_resueltas(mascara, _celdas_cruzadas(X, Y, p))
    if not any(k in especificacion for k in ("mascara", "incluir", "excluir")):
        raise ErrorDeEntrada("máscara de adquisición vacía: declara archivo NPZ, incluir o excluir")
    return mascara


def mapas_tolerancia(X, Y, tol_d, tol_w, tol_delta=None, zonas=()):
    td = np.broadcast_to(np.asarray(tol_d, float)[:, None, None], (len(tol_d), *X.shape)).copy()
    tw = np.full(X.shape, float(tol_w))
    tdelta = (
        None
        if tol_delta is None
        else np.broadcast_to(
            np.asarray(tol_delta, float)[:, None, None], (len(tol_delta), *X.shape)
        ).copy()
    )
    for i, zona in enumerate(zonas or []):
        if not isinstance(zona, dict) or "poligono_xy_um" not in zona:
            raise ErrorDeEntrada(f"zonas_tolerancia[{i}]: falta poligono_xy_um")
        m = dentro_poligono(np.stack([X, Y], axis=-1), zona["poligono_xy_um"])
        for nombre, destino, n in (
            ("tol_d", td, len(tol_d)),
            ("tol_w", tw, 1),
            ("tol_delta", tdelta, 0 if tdelta is None else len(tdelta)),
        ):
            if nombre not in zona:
                continue
            if destino is None:
                raise ErrorDeEntrada(
                    "tolerancia por zona tol_delta requiere tol_delta base por interfaz"
                )
            v = np.asarray(zona[nombre], float)
            if (
                not np.all(np.isfinite(v))
                or np.any(v < 0)
                or (destino.ndim == 3 and v.shape != (n,))
                or (destino.ndim == 2 and v.ndim)
            ):
                raise ErrorDeEntrada(
                    f"zonas_tolerancia[{i}].{nombre}: tolerancias finitas no negativas, una por capa/interfaz"
                )
            if destino.ndim == 2:
                destino[m] = v
            else:
                destino[:, m] = v[:, None]
    return td, tw, tdelta


def probabilidad_con_mapa(mu, sigma, nominal, tolerancia):
    from motor.analisis.calidad import probabilidad_fuera_de_tolerancia

    mu, sigma, nominal, tolerancia = np.broadcast_arrays(mu, sigma, nominal, tolerancia)
    out = np.empty(mu.shape)
    for t in np.unique(tolerancia):
        m = tolerancia == t
        out[m] = probabilidad_fuera_de_tolerancia(mu[m], sigma[m], nominal[m], float(t))
    return out
