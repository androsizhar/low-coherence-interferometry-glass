"""Importación genérica de CAD de vidrio laminado.

El importador no depende de nombres, armadoras, ejes de vehículo ni número de capas:
identifica las capas por geometría (sólidos laminares apilados que se tocan), arma un
marco local donde la pieza es un campo de altura z(x,y), calcula los espesores
nominales a lo largo de la normal de la cara del lado de la sonda y proyecta las
superficies auxiliares (zonas) a ese marco. Todas las longitudes de salida están en µm.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from motor.geometria.cad import Malla, _punto_mas_cercano_triangulo, escribir_stl
from motor.medicion.lectura import ErrorDeEntrada

UM = 1e3
ESCALAS = {"um": 1.0, "mm": 1e3, "cm": 1e4, "m": 1e6, "in": 25.4e3}


# --------------------------------------------------------------------------- entrada


@dataclass
class Solido:
    nombre: str
    malla: Malla
    volumen: float | None = None


@dataclass
class Superficie:
    nombre: str
    malla: Malla


@dataclass
class PiezaCAD:
    solidos: list[Solido]
    superficies: list[Superficie] = field(default_factory=list)
    fuente: dict[str, object] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


@dataclass
class OpcionesImportacion:
    lado_sonda: str = "convexo"
    tol_contacto_um: float = 50.0
    espesor_max_um: float = 25000.0
    paso_nominal_um: float | None = None
    error_cuerda_objetivo_um: float = 1.0
    max_vertices_stl: int = 300_000
    marco: str = "local"
    vecinos_ajuste: int = 24
    ng_aprox: Sequence[float] | None = None
    sep_min_picos_um: float = 300.0
    distancia_zona_um: float = 1000.0
    cobertura_min: float = 0.6

    def validar(self):
        if self.marco not in ("local", "original"):
            raise ErrorDeEntrada("marco debe ser 'local' u 'original'")
        for nombre in ("tol_contacto_um", "espesor_max_um", "error_cuerda_objetivo_um"):
            v = getattr(self, nombre)
            if not np.isfinite(v) or v <= 0:
                raise ErrorDeEntrada(f"{nombre} debe ser finito y > 0")
        if self.paso_nominal_um is not None and (
            not np.isfinite(self.paso_nominal_um) or self.paso_nominal_um <= 0
        ):
            raise ErrorDeEntrada("paso_nominal_um debe ser > 0")
        if self.max_vertices_stl < 1000 or self.vecinos_ajuste < 8:
            raise ErrorDeEntrada("max_vertices_stl >= 1000 y vecinos_ajuste >= 8")


# --------------------------------------------------------------------------- utilidades


def malla_limpia(v, f, escala: float = 1.0) -> Malla | None:
    """Suelda vértices repetidos y descarta triángulos degenerados."""
    v = np.asarray(v, float) * escala
    f = np.asarray(f, int)
    if not len(v) or not len(f):
        return None
    tam = float(np.ptp(v, axis=0).max()) or 1.0
    clave = np.round(v / (tam * 1e-10)).astype(np.int64)
    _, unico, inv = np.unique(clave, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    v2, f2 = v[unico], inv[f]
    a, b, c = v2[f2[:, 0]], v2[f2[:, 1]], v2[f2[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ok = (
        (area2 > (tam * 1e-9) ** 2)
        & (f2[:, 0] != f2[:, 1])
        & (f2[:, 1] != f2[:, 2])
        & (f2[:, 0] != f2[:, 2])
    )
    if not ok.any():
        return None
    f2 = f2[ok]
    usados, f3 = np.unique(f2, return_inverse=True)
    return Malla(v2[usados], f3.reshape(-1, 3))


def submalla(malla: Malla, seleccion: np.ndarray) -> Malla | None:
    seleccion = np.asarray(seleccion, bool)
    if not seleccion.any():
        return None
    f = malla.f[seleccion]
    usados, inv = np.unique(f, return_inverse=True)
    return Malla(malla.v[usados], inv.reshape(-1, 3))


def distancia_aprox(malla: Malla, P: np.ndarray, k: int = 16):
    """Distancia punto-malla con los k triángulos de centro más cercano (vectorizada)."""
    P = np.atleast_2d(np.asarray(P, float))
    a, b, c = malla.tri()
    arbol = malla._kdtree()
    k = int(min(k, len(malla.f)))
    _, cand = arbol.query(P, k=k)
    cand = cand.reshape(len(P), k)
    mejor = np.full(len(P), np.inf)
    tri = np.zeros(len(P), int)
    for j in range(k):
        t = cand[:, j]
        q = _punto_mas_cercano_triangulo(P, a[t], b[t], c[t])
        d = np.linalg.norm(P - q, axis=1)
        m = d < mejor
        mejor[m], tri[m] = d[m], t[m]
    return mejor, tri


def malla_cerrada(malla: Malla) -> bool:
    aristas = np.sort(
        np.concatenate([malla.f[:, [0, 1]], malla.f[:, [1, 2]], malla.f[:, [2, 0]]]), axis=1
    )
    _, cuenta = np.unique(aristas, axis=0, return_counts=True)
    return bool(np.all(cuenta == 2))


def lazos_borde(malla: Malla) -> list[np.ndarray]:
    aristas = np.concatenate([malla.f[:, [0, 1]], malla.f[:, [1, 2]], malla.f[:, [2, 0]]])
    ordenadas = np.sort(aristas, axis=1)
    _, inv, cuenta = np.unique(ordenadas, axis=0, return_inverse=True, return_counts=True)
    borde = aristas[cuenta[inv.ravel()] == 1]
    siguiente: dict[int, list[int]] = {}
    for a, b in borde:
        siguiente.setdefault(int(a), []).append(int(b))
        siguiente.setdefault(int(b), []).append(int(a))
    vistos = set()
    lazos = []
    for inicio in list(siguiente):
        if inicio in vistos:
            continue
        lazo, previo, actual = [inicio], None, inicio
        vistos.add(inicio)
        while True:
            opciones = [n for n in siguiente[actual] if n != previo and n not in vistos]
            if not opciones:
                break
            previo, actual = actual, opciones[0]
            vistos.add(actual)
            lazo.append(actual)
        if len(lazo) >= 3:
            lazos.append(np.array(lazo))
    return lazos


def simplificar_lazo(P: np.ndarray, tol: float, maximo: int = 400) -> np.ndarray:
    P = np.asarray(P, float)
    if len(P) <= 4:
        return P

    def rdp(Q):
        if len(Q) < 3:
            return Q
        a, b = Q[0], Q[-1]
        ab = b - a
        L = np.linalg.norm(ab)
        if L == 0:
            d = np.linalg.norm(Q - a, axis=1)
        else:
            d = np.abs(ab[0] * (Q[:, 1] - a[1]) - ab[1] * (Q[:, 0] - a[0])) / L
        i = int(np.argmax(d))
        if d[i] <= tol:
            return np.array([a, b])
        return np.vstack([rdp(Q[: i + 1])[:-1], rdp(Q[i:])])

    j = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    out = np.vstack([rdp(P[: j + 1])[:-1], rdp(np.vstack([P[j:], P[:1]]))[:-1]])
    while len(out) > maximo:
        tol *= 1.5
        out = simplificar_lazo(P, tol, maximo)
    return out


class CampoAltura:
    """z(x,y) por mínimos cuadrados cuadráticos locales sobre los vértices de una superficie.

    Los vértices de un teselado de CAD están sobre la superficie exacta; el ajuste
    cuadrático local no arrastra el error de cuerda de los triángulos.
    """

    def __init__(self, xy: np.ndarray, z: np.ndarray, vecinos: int = 24, malla_xy=None):
        self.xy, self.z = np.asarray(xy, float), np.asarray(z, float)
        if len(self.xy) < 6:
            raise ErrorDeEntrada("superficie con muy pocos vértices para un campo de altura")
        self.arbol = cKDTree(self.xy)
        self.k = int(min(vecinos, len(self.xy)))
        dnn = self.arbol.query(self.xy, k=2)[0][:, 1]
        self.h = float(np.median(dnn[dnn > 0])) if np.any(dnn > 0) else 1.0
        self.buscador = None
        if malla_xy is not None:
            try:
                from matplotlib.tri import Triangulation

                tri = Triangulation(malla_xy[0][:, 0], malla_xy[0][:, 1], malla_xy[1])
                self.buscador = tri.get_trifinder()
            except (ValueError, RuntimeError):
                self.buscador = None

    def dentro(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        d = self.arbol.query(xy.reshape(-1, 2), k=1)[0].reshape(xy.shape[:-1])
        cerca = d <= 2.5 * self.h
        if self.buscador is not None:
            return cerca & (self.buscador(xy[..., 0], xy[..., 1]) >= 0)
        return cerca

    def evaluar(self, xy: np.ndarray, gradiente: bool = False):
        xy = np.asarray(xy, float)
        forma = xy.shape[:-1]
        Q = xy.reshape(-1, 2)
        z = np.full(len(Q), np.nan)
        g = np.full((len(Q), 2), np.nan)
        for ini in range(0, len(Q), 20000):
            q = Q[ini : ini + 20000]
            d, idx = self.arbol.query(q, k=self.k)
            dx = (self.xy[idx, 0] - q[:, None, 0]) / self.h
            dy = (self.xy[idx, 1] - q[:, None, 1]) / self.h
            A = np.stack([np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy], axis=-1)
            escala = np.maximum(d[:, -1:], 1e-12) / self.h
            w = np.exp(-(((d / self.h) / escala) ** 2) * 2.0)
            AtW = A * w[..., None]
            N = np.einsum("mki,mkj->mij", AtW, A) + 1e-10 * np.eye(6)
            b = np.einsum("mki,mk->mi", AtW, self.z[idx])
            c = np.linalg.solve(N, b[..., None])[..., 0]
            z[ini : ini + len(q)] = c[:, 0]
            g[ini : ini + len(q)] = c[:, 1:3] / self.h
        if gradiente:
            return z.reshape(forma), g.reshape(*forma, 2)
        return z.reshape(forma)


# --------------------------------------------------------------------------- lectores


def _decodificar_step(s: str) -> str:
    def x2(m):
        h = m.group(1)
        return "".join(chr(int(h[i : i + 4], 16)) for i in range(0, len(h), 4))

    s = re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\", x2, s)
    s = re.sub(
        r"\\X\\([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]).decode("latin-1"), s
    )
    return s.replace("''", "'")


def nombres_step(ruta: str) -> tuple[list[str], list[tuple[str, int]]]:
    """Nombres de sólidos y de conchas abiertas en orden de archivo (solo como etiquetas)."""
    texto = Path(ruta).read_text(encoding="latin-1")
    solidos = [
        _decodificar_step(m.group(2))
        for m in re.finditer(
            r"=\s*(MANIFOLD_SOLID_BREP|BREP_WITH_VOIDS)\s*\(\s*'((?:[^']|'')*)'", texto
        )
    ]
    conchas = [
        (_decodificar_step(m.group(1)), len([x for x in m.group(2).split(",") if x.strip()]))
        for m in re.finditer(
            r"=\s*OPEN_SHELL\s*\(\s*'((?:[^']|'')*)'\s*,\s*\(([^)]*)\)\s*\)", texto, re.DOTALL
        )
    ]
    return solidos, conchas


def leer_step(ruta: str, tam_malla_mm: float | None = None) -> PiezaCAD:
    """Lee STEP/IGES/BREP con el núcleo OpenCASCADE de gmsh (dependencia opcional)."""
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise ErrorDeEntrada(
            "Leer STEP requiere el paquete opcional gmsh (pip install gmsh). "
            "Alternativa sin gmsh: exporta cada capa como STL cerrado y usa --capas."
        ) from exc
    if not Path(ruta).exists():
        raise ErrorDeEntrada(f"no existe el archivo CAD: {ruta}")
    avisos: list[str] = []
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("pieza")
        try:
            gmsh.model.occ.importShapes(ruta, highestDimOnly=False)
        except Exception as exc:  # gmsh lanza Exception genérica
            raise ErrorDeEntrada(f"gmsh no pudo leer {ruta}: {exc}") from exc
        gmsh.model.occ.synchronize()
        volumenes = [t for _, t in gmsh.model.getEntities(3)]
        if not volumenes:
            raise ErrorDeEntrada("el CAD no contiene sólidos: se necesitan las capas como sólidos")
        en_volumen = set()
        info = {}
        for v in volumenes:
            caras = [abs(t) for _, t in gmsh.model.getBoundary([(3, v)], oriented=False)]
            en_volumen.update(caras)
            info[v] = [abs(gmsh.model.occ.getMass(3, v)), None, caras]
        for v in volumenes:
            caras = info[v][2]
            V = info[v][0]
            # área exacta solo si el volumen puede ser laminar (evita miles de llamadas)
            bb = np.asarray(gmsh.model.getBoundingBox(3, v))
            ext = np.sort(bb[3:] - bb[:3])
            if ext[2] <= 0 or 100.0 * ext[2] * ext[1] <= V:
                info[v][1] = 0.0
                continue
            info[v][1] = sum(gmsh.model.occ.getMass(2, c) for c in caras)
        libres = [t for _, t in gmsh.model.getEntities(2) if t not in en_volumen]
        n_sol, conchas = [], []
        with contextlib.suppress(OSError, UnicodeError):
            n_sol, conchas = nombres_step(ruta)
        nombres_vol = {}
        if len(n_sol) == len(volumenes):
            nombres_vol = dict(zip(volumenes, n_sol))
            avisos.append("nombres_de_solidos_asignados_por_orden_de_archivo")
        nombres_sup = {}
        if conchas and sum(n for _, n in conchas) == len(libres):
            it = iter(libres)
            for nombre, n in conchas:
                for _ in range(n):
                    nombres_sup[next(it)] = nombre
            avisos.append("nombres_de_superficies_asignados_por_orden_de_archivo")
        laminares = []
        for v in volumenes:
            V, A, _ = info[v]
            if not A:
                continue
            teq, Lc = 2 * V / max(A, 1e-30), np.sqrt(A / 2)
            if V > 0 and teq < min(25.0, 0.05 * Lc):
                laminares.append(v)
        if laminares:
            # una capa de laminado cubre la pieza; accesorios pequeños se descartan antes de mallar
            Amax = max(info[v][1] for v in laminares)
            pequenos = [v for v in laminares if info[v][1] < 0.3 * Amax]
            laminares = [v for v in laminares if v not in pequenos]
            avisos.extend(
                f"solido_laminar_pequeno_ignorado:{nombres_vol.get(v, f'solido_{v}')}"
                for v in pequenos
            )
        if len(laminares) < 2:
            raise ErrorDeEntrada(
                f"solo {len(laminares)} sólido(s) laminar(es): un laminado necesita al menos 2 capas"
            )
        quitar = [(3, v) for v in volumenes if v not in laminares]
        if quitar:
            # ocultar es mucho más rápido que borrar del modelo OCC; solo se malla lo visible
            gmsh.model.setVisibility(quitar, 0, recursive=True)
            gmsh.model.setVisibility([(3, v) for v in laminares], 1, recursive=True)
            gmsh.model.setVisibility([(2, t) for t in libres], 1, recursive=True)
            gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
        if tam_malla_mm is None:
            Amax = max(info[v][1] for v in laminares)
            tam_malla_mm = float(np.clip(np.sqrt(Amax / 2) / 120, 1.0, 15.0))
        gmsh.option.setNumber("Mesh.MeshSizeMax", tam_malla_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", tam_malla_mm / 20)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.model.mesh.generate(2)
        etiquetas, coords, _ = gmsh.model.mesh.getNodes()
        X = coords.reshape(-1, 3)
        pos = np.full(int(etiquetas.max()) + 1, -1)
        pos[etiquetas.astype(int)] = np.arange(len(etiquetas))

        def triangulos(caras):
            tri = []
            for c in caras:
                tipos, _, nodos = gmsh.model.mesh.getElements(2, c)
                for tp, nd in zip(tipos, nodos):
                    if tp == 2:
                        tri.append(pos[nd.astype(int)].reshape(-1, 3))
            return np.concatenate(tri) if tri else np.empty((0, 3), int)

        solidos = []
        for v in laminares:
            m = malla_limpia(X, triangulos(info[v][2]), UM)
            if m is not None:
                solidos.append(Solido(nombres_vol.get(v, f"solido_{v}"), m, info[v][0] * UM**3))
        grupos: dict[str, list[int]] = {}
        for t in libres:
            grupos.setdefault(nombres_sup.get(t, f"superficie_{t}"), []).append(t)
        superficies = []
        for nombre, caras in grupos.items():
            m = malla_limpia(X, triangulos(caras), UM)
            if m is not None:
                superficies.append(Superficie(nombre, m))
        ignorados = [nombres_vol.get(v, f"solido_{v}") for v in volumenes if v not in laminares]
        fuente = {
            "archivo": Path(ruta).name,
            "sha256": _sha256(ruta),
            "formato": "STEP/IGES/BREP via gmsh-OpenCASCADE",
            "unidades_origen": "convertidas a mm por OpenCASCADE y luego a µm",
            "tam_malla_mm": tam_malla_mm,
            "solidos_no_laminares_ignorados": ignorados,
        }
        return PiezaCAD(solidos, superficies, fuente, avisos)
    finally:
        gmsh.finalize()


def leer_stl_capas(
    rutas: Sequence[str],
    unidades: str = "mm",
    zonas: dict[str, str] | None = None,
) -> PiezaCAD:
    """Cada STL es un sólido cerrado (una capa), en cualquier orden."""
    from motor.geometria.cad import leer_stl

    if unidades not in ESCALAS:
        raise ErrorDeEntrada(f"unidades debe ser una de {list(ESCALAS)}")
    esc = ESCALAS[unidades]
    solidos = []
    for r in rutas:
        m0 = leer_stl(r)
        m = malla_limpia(m0.v, m0.f, esc)
        if m is None:
            raise ErrorDeEntrada(f"{r}: malla vacía")
        if not malla_cerrada(m):
            raise ErrorDeEntrada(f"{r}: la capa debe ser un sólido cerrado (estanco)")
        solidos.append(Solido(Path(r).stem, m, abs(m.volumen())))
    superficies = []
    for nombre, r in (zonas or {}).items():
        m0 = leer_stl(r)
        m = malla_limpia(m0.v, m0.f, esc)
        if m is not None:
            superficies.append(Superficie(nombre, m))
    fuente = {
        "archivo": [Path(r).name for r in rutas],
        "sha256": [_sha256(r) for r in rutas],
        "formato": "STL por capa",
        "unidades_origen": unidades,
    }
    return PiezaCAD(solidos, superficies, fuente)


def _sha256(ruta):
    with Path(ruta).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


# --------------------------------------------------------------------------- núcleo


@dataclass
class ModeloImportado:
    nombres_capas: list[str]
    marco_R: np.ndarray
    marco_t: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    z0: np.ndarray
    d_nom: np.ndarray
    mascara: np.ndarray
    malla_frontal: Malla
    zonas: list[dict]
    diagnostico: dict
    fuente: dict
    avisos: list[str]
    sugerencias: dict


def _apilamiento(solidos: list[Solido], op: OpcionesImportacion, avisos: list[str]):
    datos = []
    for s in solidos:
        A = float(s.malla.areas().sum())
        V = float(s.volumen) if s.volumen is not None else abs(s.malla.volumen())
        teq = 2 * V / A if A > 0 else np.inf
        datos.append((A, teq, np.sqrt(A / 2)))
    lam = [i for i, (A, teq, Lc) in enumerate(datos) if 0 < teq < min(op.espesor_max_um, 0.05 * Lc)]
    avisos.extend(
        f"solido_no_laminar_ignorado:{solidos[i].nombre}"
        for i in sorted(set(range(len(solidos))) - set(lam))
    )
    if len(lam) < 2:
        raise ErrorDeEntrada("se necesitan al menos 2 sólidos laminares que se toquen")
    rng = np.random.default_rng(0)
    muestras = {i: solidos[i].malla.muestrear(2000, rng) for i in lam}
    cajas = {i: (solidos[i].malla.v.min(0), solidos[i].malla.v.max(0)) for i in lam}
    contacto, adheridos = {}, set()
    for a in lam:
        for b in lam:
            if b <= a:
                continue
            (amin, amax), (bmin, bmax) = cajas[a], cajas[b]
            if np.any(amin > bmax + op.tol_contacto_um) or np.any(bmin > amax + op.tol_contacto_um):
                continue
            fa = np.mean(distancia_aprox(solidos[b].malla, muestras[a])[0] < op.tol_contacto_um)
            fb = np.mean(distancia_aprox(solidos[a].malla, muestras[b])[0] < op.tol_contacto_um)
            area = max(fa * datos[a][0], fb * datos[b][0])
            cara_mayor = max(datos[a][0], datos[b][0]) / 2
            if area > op.cobertura_min * cara_mayor:
                contacto[(a, b)] = area
            elif area > 0.2 * min(datos[a][0], datos[b][0]) / 2:
                menor = a if datos[a][0] < datos[b][0] else b
                adheridos.add(solidos[menor].nombre)
    vecinos = {i: [] for i in lam}
    for a, b in contacto:
        vecinos[a].append(b)
        vecinos[b].append(a)
    vistos, componentes = set(), []
    for i in lam:
        if i in vistos:
            continue
        pila, comp = [i], []
        while pila:
            j = pila.pop()
            if j in vistos:
                continue
            vistos.add(j)
            comp.append(j)
            pila.extend(vecinos[j])
        componentes.append(comp)
    componentes = [c for c in componentes if len(c) >= 2]
    if not componentes:
        raise ErrorDeEntrada(
            "ninguna pareja de capas laminares está en contacto: revisa unidades o tol_contacto_um"
        )
    avisos.extend(f"solido_adherido_de_cobertura_parcial_ignorado:{n}" for n in sorted(adheridos))
    comp = max(componentes, key=lambda c: sum(datos[i][0] for i in c))
    avisos.extend(
        "apilado_secundario_ignorado:" + ",".join(solidos[i].nombre for i in c)
        for c in componentes
        if c is not comp
    )
    grados = {i: len(vecinos[i]) for i in comp}
    aristas = sum(grados.values()) // 2
    extremos = [i for i in comp if grados[i] == 1]
    if any(g > 2 for g in grados.values()) or aristas != len(comp) - 1 or len(extremos) != 2:
        raise ErrorDeEntrada(
            "las capas en contacto no forman una pila simple (cada capa con a lo más 2 vecinas); "
            "separa sub-ensambles o indica las capas explícitamente"
        )
    orden, previo, actual = [extremos[0]], None, extremos[0]
    while len(orden) < len(comp):
        siguiente = next(j for j in vecinos[actual] if j != previo)
        previo, actual = actual, siguiente
        orden.append(actual)
    return orden, {i: datos[i] for i in orden}


def _superficies_pila(solidos, orden, datos, op: OpcionesImportacion):
    mallas = [solidos[i].malla for i in orden]
    n = len(mallas)
    contactos = []
    for j in range(n - 1):
        a, b = mallas[j], mallas[j + 1]
        cen = (lambda t: (t[0] + t[1] + t[2]) / 3)(a.tri())
        d, _ = distancia_aprox(b, cen)
        sel = d < op.tol_contacto_um
        m = submalla(a, sel)
        if m is None:
            raise ErrorDeEntrada(f"no se aisló la interfaz entre capas {j} y {j + 1}")
        contactos.append(m)

    def cara_libre(j, interfaz):
        m = mallas[j]
        t = m.tri()
        cen = (t[0] + t[1] + t[2]) / 3
        d, ti = distancia_aprox(interfaz, cen)
        paralelo = np.abs(np.einsum("ij,ij->i", m.normales_cara(), interfaz.normales_cara()[ti]))
        teq = datos[orden[j]][1]
        sel = (d > 0.3 * teq) & (paralelo > np.cos(np.deg2rad(35)))
        cara = submalla(m, sel)
        if cara is None:
            raise ErrorDeEntrada(f"no se aisló la cara libre de la capa {j}")
        return cara

    return [cara_libre(0, contactos[0]), *contactos, cara_libre(n - 1, contactos[-1])]


def _marco(superficies, nombres, op: OpcionesImportacion, avisos):
    medio = superficies[len(superficies) // 2]
    P = medio.v
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    ex, ez = Vt[0], Vt[2]
    lado_a = superficies[0].v.mean(0) - superficies[-1].v.mean(0)
    if np.dot(lado_a, ez) < 0:
        ez = -ez
    ey = np.cross(ez, ex)
    R = np.vstack([ex, ey, ez])
    Q = (P - c) @ R.T
    A = np.column_stack(
        [Q[:, 0] ** 2, Q[:, 0] * Q[:, 1], Q[:, 1] ** 2, Q[:, 0], Q[:, 1], np.ones(len(Q))]
    )
    coef = np.linalg.lstsq(A, Q[:, 2], rcond=None)[0]
    H = coef[0] + coef[2]
    flecha = abs(H) * (np.ptp(Q[:, 0]) ** 2 + np.ptp(Q[:, 1]) ** 2) / 8
    convexo_a = H < 0
    lado = op.lado_sonda
    invertir = False
    if lado in ("convexo", "concavo"):
        if flecha < 10.0:
            avisos.append(
                "pieza_practicamente_plana:lado_sonda_tomado_como_la_primera_capa_encontrada"
            )
        else:
            invertir = (lado == "convexo") != convexo_a
    elif lado in nombres:
        if lado == nombres[-1]:
            invertir = True
        elif lado != nombres[0]:
            raise ErrorDeEntrada("lado_sonda por nombre debe ser una capa de un extremo")
    else:
        raise ErrorDeEntrada("lado_sonda debe ser convexo, concavo o el nombre de una capa extrema")
    if invertir:
        superficies = superficies[::-1]
        nombres = nombres[::-1]
        ez, ey = -ez, -ey
    R = np.vstack([ex, ey, ez])
    frente = superficies[0].v
    origen = frente.mean(0)
    if op.marco == "original":
        if ez[2] < 0.9:
            raise ErrorDeEntrada(
                "marco 'original' requiere que la sonda mire hacia +z del CAD; usa marco 'local'"
            )
        R, origen = np.eye(3), np.zeros(3)
    t = -R @ origen
    return superficies, nombres, R, t, float(flecha)


def analizar_pieza(pieza: PiezaCAD, op: OpcionesImportacion | None = None) -> ModeloImportado:
    op = op or OpcionesImportacion()
    op.validar()
    avisos = list(pieza.avisos)
    orden, datos = _apilamiento(pieza.solidos, op, avisos)
    nombres = [
        re.sub(r"[^\w.-]+", "_", pieza.solidos[i].nombre).strip("._") or f"capa{j + 1}"
        for j, i in enumerate(orden)
    ]
    if len(set(nombres)) != len(nombres):
        nombres = [f"{n}_{j + 1}" for j, n in enumerate(nombres)]
    sup = _superficies_pila(pieza.solidos, orden, datos, op)
    sup, nombres, R, t, flecha = _marco(sup, nombres, op, avisos)
    loc = [Malla(s.v @ R.T + t, s.f) for s in sup]
    campos = [CampoAltura(m.v[:, :2], m.v[:, 2], op.vecinos_ajuste, (m.v[:, :2], m.f)) for m in loc]
    xy = loc[0].v[:, :2]
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    extension = max(x1 - x0, y1 - y0)
    paso = op.paso_nominal_um or float(np.clip(extension / 250, 500.0, 10000.0))
    X, Y, z0, d, mascara, g0 = _rejilla_espesores(campos, x0, x1, y0, y1, paso)
    if mascara.sum() < 4:
        raise ErrorDeEntrada("la rejilla nominal no tiene nodos válidos: revisa el CAD")
    curv = _curvaturas(campos[0], X, Y, mascara, paso)
    radio = 1 / max(float(np.percentile(curv[mascara], 99)), 1e-15)
    h_stl = float(np.sqrt(8 * radio * op.error_cuerda_objetivo_um))
    area_xy = float(mascara.sum()) * paso * paso
    h_min_cupo = float(np.sqrt(area_xy / op.max_vertices_stl))
    if h_stl < h_min_cupo:
        avisos.append(
            f"stl_limitado_por_max_vertices:error_cuerda_estimado_{h_min_cupo**2 / (8 * radio):.2f}um"
        )
    h_stl = float(np.clip(max(h_stl, h_min_cupo), 200.0, paso))
    malla_frontal, err_cuerda = _malla_frontal(campos[0], x0, x1, y0, y1, h_stl, radio)
    zonas, ignoradas = _zonas(pieza.superficies, R, t, campos, mascara, X, Y, z0, op)
    diag = _diagnostico(
        nombres, X, Y, d, mascara, g0, curv, campos, loc, paso, flecha, op, err_cuerda, h_stl
    )
    diag["superficies_ignoradas"] = ignoradas
    avisos.extend(diag.pop("avisos"))
    sug = _sugerencias(d, mascara, paso, diag)
    return ModeloImportado(
        nombres_capas=nombres,
        marco_R=R,
        marco_t=t,
        X=X,
        Y=Y,
        z0=z0,
        d_nom=d,
        mascara=mascara,
        malla_frontal=malla_frontal,
        zonas=zonas,
        diagnostico=diag,
        fuente=pieza.fuente,
        avisos=avisos,
        sugerencias=sug,
    )


def _normal_desde_gradiente(g):
    n = np.concatenate([-g, np.ones((*g.shape[:-1], 1))], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _rejilla_espesores(campos, x0, x1, y0, y1, paso):
    xs = np.arange(x0, x1 + paso / 2, paso)
    ys = np.arange(y0, y1 + paso / 2, paso)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], axis=-1)
    z0, g0 = campos[0].evaluar(P, gradiente=True)
    valido = campos[0].dentro(P) & np.isfinite(z0)
    n = _normal_desde_gradiente(np.nan_to_num(g0))
    p0 = np.stack([X, Y, np.nan_to_num(z0)], axis=-1)
    s_prev = np.zeros(X.shape)
    d = np.full((len(campos) - 1, *X.shape), np.nan)
    for k in range(1, len(campos)):
        s = s_prev.copy()
        for _ in range(6):
            q = p0 - s[..., None] * n
            zk, gk = campos[k].evaluar(q[..., :2], gradiente=True)
            f = q[..., 2] - zk
            df = -n[..., 2] + np.einsum("...i,...i->...", np.nan_to_num(gk), n[..., :2])
            s = s - f / np.where(np.abs(df) < 1e-6, -1.0, df)
        q = p0 - s[..., None] * n
        valido &= campos[k].dentro(q[..., :2]) & np.isfinite(s)
        d[k - 1] = s - s_prev
        s_prev = s
    valido &= np.all(d > 0, axis=0)
    d[:, ~valido] = np.nan
    z0 = np.where(valido, z0, np.nan)
    return X, Y, z0, d, valido, g0


def _curvaturas(campo, X, Y, mascara, paso):
    h = paso / 2
    P = np.stack([X, Y], axis=-1)
    z = campo.evaluar(P)
    zxx = (campo.evaluar(P + [h, 0]) - 2 * z + campo.evaluar(P - [h, 0])) / h**2
    zyy = (campo.evaluar(P + [0, h]) - 2 * z + campo.evaluar(P - [0, h])) / h**2
    zxy = (
        campo.evaluar(P + [h, h])
        - campo.evaluar(P + [h, -h])
        - campo.evaluar(P + [-h, h])
        + campo.evaluar(P + [-h, -h])
    ) / (4 * h * h)
    tr, det = zxx + zyy, zxx * zyy - zxy**2
    disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    k = np.maximum(np.abs(tr / 2 + disc), np.abs(tr / 2 - disc))
    return np.where(mascara & np.isfinite(k), k, 0.0)


def _malla_frontal(campo, x0, x1, y0, y1, h, radio):
    xs = np.arange(x0, x1 + h / 2, h)
    ys = np.arange(y0, y1 + h / 2, h)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], axis=-1)
    Z = campo.evaluar(P)
    valido = campo.dentro(P) & np.isfinite(Z)
    ny, nx = X.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    f = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    v = valido.ravel()
    f = f[v[f].all(axis=1)]
    V = np.column_stack([X.ravel(), Y.ravel(), np.nan_to_num(Z.ravel())])
    m = malla_limpia(V, f)
    if m is None:
        raise ErrorDeEntrada("no se pudo construir la superficie frontal")
    return m, float(h * h / (8 * radio))


def _zonas(superficies, R, t, campos, mascara, X, Y, z0, op):
    zonas, ignoradas = [], []
    for s in superficies:
        V = s.malla.v @ R.T + t
        dist = []
        for c in campos:
            zk = c.evaluar(V[:, :2])
            dist.append(np.nanmedian(np.abs(V[:, 2] - zk)) if np.isfinite(zk).any() else np.inf)
        k = int(np.argmin(dist))
        if not np.isfinite(dist[k]) or dist[k] > op.distancia_zona_um:
            ignoradas.append({"nombre": s.nombre, "motivo": "no_esta_sobre_el_vidrio"})
            continue
        lazos = []
        for lazo in lazos_borde(s.malla):
            P = V[lazo, :2]
            if len(P) >= 3:
                lazos.append(simplificar_lazo(P, 200.0).round(1).tolist())
        if not lazos:
            ignoradas.append({"nombre": s.nombre, "motivo": "superficie_sin_borde"})
            continue
        area = float(s.malla.areas().sum()) / 1e6
        zonas.append(
            {
                "nombre": s.nombre,
                "interfaz": k,
                "distancia_mediana_um": float(dist[k]),
                "area_mm2": area,
                "lazos_xy_um": lazos,
            }
        )
    return zonas, ignoradas


def _diagnostico(nombres, X, Y, d, mascara, g0, curv, campos, loc, paso, flecha, op, err, h_stl):
    avisos = []
    n = _normal_desde_gradiente(np.nan_to_num(g0))
    incl = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
    k = curv[mascara]
    diag = {
        "capas": {},
        "inclinacion_max_respecto_a_z_local_deg": float(incl[mascara].max()),
        "radio_curvatura_min_mm": float(1 / max(np.percentile(k, 99), 1e-15) / UM),
        "radio_curvatura_mediano_mm": float(1 / max(np.median(k), 1e-15) / UM),
        "flecha_um": flecha,
        "paso_rejilla_nominal_um": paso,
        "paso_stl_frontal_um": h_stl,
        "error_cuerda_stl_frontal_um": err,
        "espaciado_vertices_cad_um": [float(c.h) for c in campos],
        "area_util_mm2": float(mascara.sum() * paso * paso / 1e6),
    }
    if diag["inclinacion_max_respecto_a_z_local_deg"] > 60:
        avisos.append("inclinacion_mayor_a_60_grados:la_pieza_no_es_un_campo_de_altura_confiable")
    ng = np.asarray(op.ng_aprox if op.ng_aprox is not None else [1.5] * len(nombres), float)
    for j, nombre in enumerate(nombres):
        dj = d[j][mascara]
        gy, gx = np.gradient(np.where(mascara, d[j], np.nan), paso, paso)
        w = np.hypot(gx, gy)[mascara] * UM
        w = w[np.isfinite(w)]
        sep = 2 * ng[j] * float(dj.min())
        diag["capas"][nombre] = {
            "espesor_media_um": float(dj.mean()),
            "espesor_min_um": float(dj.min()),
            "espesor_max_um": float(dj.max()),
            "variacion_um": float(np.ptp(dj)),
            "cuna_p50_mrad": float(np.median(w)) if w.size else 0.0,
            "cuna_p95_mrad": float(np.percentile(w, 95)) if w.size else 0.0,
            "separacion_minima_picos_L_um": sep,
        }
        if sep < op.sep_min_picos_um:
            avisos.append(f"capa_{nombre}_demasiado_delgada_para_separar_picos")
    if err > 1.0:
        avisos.append(f"error_de_cuerda_del_stl_frontal_{err:.2f}um")
    diag["supuesto_capas_localmente_planas"] = {
        "radio_min_mm": diag["radio_curvatura_min_mm"],
        "flecha_en_spot_1mm_um": float(1e6 / (8 * diag["radio_curvatura_min_mm"] * UM)),
        "valido": diag["radio_curvatura_min_mm"] > 50.0,
    }
    if not diag["supuesto_capas_localmente_planas"]["valido"]:
        avisos.append("radio_de_curvatura_menor_a_50mm:revisar_modelo_local_plano")
    diag["avisos"] = avisos
    return diag


def _sugerencias(d, mascara, paso, diag):
    p95 = [c["cuna_p95_mrad"] for c in diag["capas"].values()]
    idx = int(np.argmax(p95))
    sug = {"idx_cuna": None, "direccion_cuna": None}
    if p95[idx] > 0.02:
        gy, gx = np.gradient(np.where(mascara, d[idx], np.nan), paso, paso)
        g = np.array([np.nanmean(gx[mascara]), np.nanmean(gy[mascara])])
        if np.linalg.norm(g) > 0:
            sug = {"idx_cuna": idx, "direccion_cuna": (g / np.linalg.norm(g)).tolist()}
    return sug


# --------------------------------------------------------------------------- salida


def guardar_modelo(modelo: ModeloImportado, carpeta: str, figura: bool = True) -> str:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    escribir_stl(carpeta / "cad_frontal.stl", modelo.malla_frontal, "cad_frontal")
    np.savez_compressed(
        carpeta / "nominal.npz",
        X=modelo.X,
        Y=modelo.Y,
        z0=modelo.z0,
        d_nom=modelo.d_nom,
        mascara=modelo.mascara,
    )
    medias = [float(np.nanmean(modelo.d_nom[j][modelo.mascara])) for j in range(len(modelo.d_nom))]
    meta = {
        "version_formato": 1,
        "unidades": "um",
        "fuente": modelo.fuente,
        "capas": modelo.nombres_capas,
        "d_nom_media_um": medias,
        "marco": {
            "descripcion": "P_local = R @ P_cad + t (µm). Interfaz 0 = cara del lado de la sonda; z local hacia la sonda.",
            "R": modelo.marco_R.tolist(),
            "t": modelo.marco_t.tolist(),
        },
        "archivos": {"cad": "cad_frontal.stl", "nominal": "nominal.npz"},
        "sugerencias": modelo.sugerencias,
        "diagnostico": modelo.diagnostico,
        "avisos": modelo.avisos,
        "zonas": modelo.zonas,
    }
    ruta = carpeta / "cad_importado.json"
    ruta.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    fragmento = json.dumps(fragmento_config(modelo, medias), indent=2, ensure_ascii=False)
    (carpeta / "config_fragmento.json").write_text(fragmento, encoding="utf-8")
    if figura:
        figura_modelo(modelo, carpeta / "vista_cad.png")
    return str(ruta)


def fragmento_config(modelo: ModeloImportado, medias) -> dict:
    sug = modelo.sugerencias
    frag = {
        "cad": "cad_frontal.stl",
        "cad_importado": "cad_importado.json",
        "capas": {
            "nombres": modelo.nombres_capas,
            "d_nom": [round(m, 3) for m in medias],
            "d_nom_mapa": "nominal.npz",
        },
    }
    if sug.get("idx_cuna") is not None:
        frag["capas"]["idx_pvb"] = sug["idx_cuna"]
        frag["cuna"] = {"w_nom": "cad", "tol": "<tolerancia de cuña en mrad>"}
        frag["direccion_cuna"] = "cad"
    return frag


def figura_modelo(modelo: ModeloImportado, ruta: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(modelo.nombres_capas)
    fig, ejes = plt.subplots(1, N, figsize=(5.2 * N, 4.6), squeeze=False)
    X, Y = modelo.X / UM, modelo.Y / UM
    for j, ax in enumerate(ejes[0]):
        d = modelo.d_nom[j]
        centro = float(np.nanmean(d))
        mitad = max(float(np.nanmax(d) - np.nanmin(d)) / 2, 1.0)  # ±1 µm mínimo: no exagerar ruido
        im = ax.pcolormesh(
            X, Y, d, shading="auto", cmap="viridis", vmin=centro - mitad, vmax=centro + mitad
        )
        barra = fig.colorbar(im, ax=ax, label="espesor nominal (µm)")
        barra.formatter.set_useOffset(False)
        barra.update_ticks()
        for z in modelo.zonas:
            if z["area_mm2"] < 100:
                continue
            for lazo in z["lazos_xy_um"]:
                P = np.asarray(lazo) / UM
                ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], lw=0.8, color="w")
        ax.set_aspect("equal")
        ax.set_title(modelo.nombres_capas[j], fontsize=10)
        ax.set_xlabel("x local (mm)")
        ax.set_ylabel("y local (mm)")
    fig.suptitle("Espesor nominal por capa en el marco local (contornos: zonas del CAD)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def cargar_modelo(ruta_json: str) -> dict:
    meta = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
    if meta.get("version_formato") != 1 or meta.get("unidades") != "um":
        raise ErrorDeEntrada(f"{ruta_json}: formato de CAD importado no reconocido")
    meta["_base"] = str(Path(ruta_json).resolve().parent)
    return meta


def zonas_por_nombre(meta: dict) -> dict[str, list[list]]:
    out: dict[str, list[list]] = {}
    for z in meta.get("zonas", []):
        out.setdefault(z["nombre"], []).extend(z["lazos_xy_um"])
    return out


def importar(
    ruta_o_rutas,
    carpeta: str,
    opciones: OpcionesImportacion | None = None,
    unidades: str = "mm",
    tam_malla_mm: float | None = None,
    zonas_stl: dict[str, str] | None = None,
) -> tuple[ModeloImportado, str]:
    if isinstance(ruta_o_rutas, (list, tuple)) and len(ruta_o_rutas) > 1:
        pieza = leer_stl_capas(ruta_o_rutas, unidades, zonas_stl)
    else:
        ruta = ruta_o_rutas[0] if isinstance(ruta_o_rutas, (list, tuple)) else ruta_o_rutas
        if ruta.lower().endswith(".stl"):
            raise ErrorDeEntrada("con STL indica un archivo cerrado por capa (al menos 2)")
        pieza = leer_step(ruta, tam_malla_mm)
    modelo = analizar_pieza(pieza, opciones)
    return modelo, guardar_modelo(modelo, carpeta)
