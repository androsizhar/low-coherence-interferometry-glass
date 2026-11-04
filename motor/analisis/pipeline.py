from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from motor.analisis.calidad import Cuna, cuna_pvb, decidir
from motor.analisis.gemelo import (
    ConfigGemelo,
    ParametrosHUD,
    cuna_optima_receta,
    distinguibilidad,
    imagen_doble_receta,
    sensibilidad,
)
from motor.geometria.cad import (
    Malla,
    ResultadoRegistro,
    Transformacion,
    interfaces_nominales,
    leer_stl,
    registrar_icp,
)
from motor.geometria.exportar_cad import exportar_stl
from motor.geometria.soporte import (
    aplicar_mascara,
    cargar_especificacion,
    mapas_tolerancia,
    probabilidad_con_mapa,
)
from motor.geometria.superficies import (
    ProcesoGaussiano2D,
    desviacion_contra_cad,
    puntos_interfaces,
)
from motor.medicion.lectura import (
    ErrorDeEntrada,
    Instrumento,
    Medicion,
    preparar_espectros_archivo,
)
from motor.medicion.materiales import Desplazado, Material
from motor.medicion.modelo_directo import Capa
from motor.medicion.picos import ResultadoPicos, detectar_picos
from motor.metrologia.calibracion import Calibracion, calibrar, validar_con_muestra
from motor.metrologia.diagnosticos import (
    covarianza_normal,
    diagnostico_incidencia,
    notificar,
    presupuesto,
)
from motor.metrologia.espesor import espesor_calibrado

UM = 1e3


@dataclass
class ConfigPipeline:
    instrumento: Instrumento
    d_nom: np.ndarray
    tol_d: np.ndarray
    ng_prior: np.ndarray
    grupos: List[str]
    nombres_capas: List[str] = field(default_factory=lambda: ["vidrio_ext", "pvb", "vidrio_int"])
    idx_pvb: int = 1
    w_nom: float = 0.0
    tol_w: float = 0.1
    sigma_pos: float = 3.0
    rango_frontal: Tuple[float, float] = (19000.0, 21000.0)
    umbral_falla: float = 0.05
    grilla: Tuple[int, int] = (31, 31)
    materiales_apriori: Optional[Sequence[Material]] = None
    hud: Optional[ParametrosHUD] = None
    prior_montaje: Optional[Tuple[float, float]] = None
    modo_registro: str = "fijo"
    transformacion: Optional[Transformacion] = None
    cov_transformacion: Optional[np.ndarray] = None
    direccion_cuna: Tuple[float, float] = (1.0, 0.0)
    gradiente_nominal: Optional[np.ndarray] = None
    origen_nominal_xy: Tuple[float, float] = (0.0, 0.0)
    tol_delta: Optional[np.ndarray] = None
    origen_datos: str = "no_declarado"
    mascara_adquisicion: Optional[dict] = None
    zonas_tolerancia: List[dict] = field(default_factory=list)
    presupuesto_declarado: dict = field(default_factory=dict)
    pose_exacta_declarada: bool = False
    fuente_registro: str = "no_documentada"
    rangos_ng_esperados: Optional[np.ndarray] = None

    def __post_init__(self):
        self.d_nom = np.asarray(self.d_nom, float)
        self.tol_d = np.asarray(self.tol_d, float)
        self.ng_prior = np.asarray(self.ng_prior, float)
        n = self.d_nom.size
        if (
            self.d_nom.ndim != 1
            or n < 1
            or self.tol_d.shape != (n,)
            or self.ng_prior.shape != (n,)
            or len(self.grupos) != n
            or len(self.nombres_capas) != n
        ):
            raise ErrorDeEntrada("d_nom, tol_d, ng_prior y grupos deben tener una entrada por capa")
        if not (
            np.all(np.isfinite(self.d_nom))
            and np.all(self.d_nom > 0)
            and np.all(np.isfinite(self.tol_d))
            and np.all(self.tol_d >= 0)
            and np.all(np.isfinite(self.ng_prior))
            and np.all(self.ng_prior > 1)
        ):
            raise ErrorDeEntrada(
                "espesores e indices positivos finitos; tolerancias no negativas finitas"
            )
        if len(set(self.nombres_capas)) != n or any(
            not isinstance(s, str) or not s or "/" in s or "\\" in s or s in (".", "..")
            for s in self.nombres_capas
        ):
            raise ErrorDeEntrada("nombres de capas unicos y seguros para archivos")
        if not isinstance(self.idx_pvb, int) or not 0 <= self.idx_pvb < n:
            raise ErrorDeEntrada("idx_pvb fuera de rango")
        if len(self.grilla) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 2
            for v in self.grilla
        ):
            raise ErrorDeEntrada("grilla debe contener dos enteros >= 2")
        if not (
            np.isfinite(self.sigma_pos)
            and self.sigma_pos > 0
            and np.isfinite(self.w_nom)
            and np.isfinite(self.tol_w)
            and self.tol_w >= 0
            and 0 < self.umbral_falla < 1
        ):
            raise ErrorDeEntrada(
                "sigma_pos > 0, tolerancia de cuna >= 0 y umbral en (0,1), finitos"
            )
        if (
            len(self.rango_frontal) != 2
            or not np.all(np.isfinite(self.rango_frontal))
            or not 0 < self.rango_frontal[0] < self.rango_frontal[1]
        ):
            raise ErrorDeEntrada("rango_frontal debe ser positivo, finito y creciente")
        if self.materiales_apriori is not None and len(self.materiales_apriori) != n:
            raise ErrorDeEntrada("materiales_apriori debe contener un material por capa")
        if self.modo_registro not in ("fijo", "icp_forma"):
            raise ErrorDeEntrada("modo_registro debe ser fijo o icp_forma")
        if self.prior_montaje is not None and (
            np.shape(self.prior_montaje) != (2,)
            or not np.all(np.isfinite(self.prior_montaje))
            or np.any(np.asarray(self.prior_montaje) <= 0)
        ):
            raise ErrorDeEntrada("prior_montaje debe tener dos desviaciones positivas finitas")
        if self.transformacion is not None:
            R, t = np.asarray(self.transformacion.R, float), np.asarray(
                self.transformacion.t, float
            )
            if (
                R.shape != (3, 3)
                or t.shape != (3,)
                or not np.all(np.isfinite(R))
                or not np.all(np.isfinite(t))
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(R), 1, atol=1e-8)
            ):
                raise ErrorDeEntrada(
                    "transformacion debe ser rigida propia: R ortogonal, det=1 y t finita"
                )
        if self.cov_transformacion is not None:
            C = np.asarray(self.cov_transformacion, float)
            if (
                C.shape != (6, 6)
                or not np.all(np.isfinite(C))
                or not np.allclose(C, C.T)
                or np.linalg.eigvalsh(C).min() < -1e-12
            ):
                raise ErrorDeEntrada(
                    "cov_transformacion debe ser simetrica semidefinida positiva (6,6)"
                )
            self.cov_transformacion = C
        direc = np.asarray(self.direccion_cuna, float)
        if direc.shape != (2,) or not np.all(np.isfinite(direc)) or np.linalg.norm(direc) == 0:
            raise ErrorDeEntrada("direccion_cuna debe ser un vector finito 2-D no nulo")
        self.direccion_cuna = tuple(direc / np.linalg.norm(direc))
        self.gradiente_nominal = (
            np.zeros((n, 2))
            if self.gradiente_nominal is None
            else np.asarray(self.gradiente_nominal, float)
        )
        if (
            self.gradiente_nominal.shape != (n, 2)
            or not np.all(np.isfinite(self.gradiente_nominal))
            or np.shape(self.origen_nominal_xy) != (2,)
            or not np.all(np.isfinite(self.origen_nominal_xy))
        ):
            raise ErrorDeEntrada(
                "gradiente_nominal (N,2) y origen_nominal_xy (2,) deben ser finitos"
            )
        if self.tol_delta is not None:
            self.tol_delta = np.broadcast_to(np.asarray(self.tol_delta, float), (n + 1,)).copy()
            if not np.all(np.isfinite(self.tol_delta)) or np.any(self.tol_delta < 0):
                raise ErrorDeEntrada("tol_delta debe ser finita >= 0 por interfaz")
        if self.origen_datos not in ("simulado", "real_declarado", "no_declarado"):
            raise ErrorDeEntrada("origen_datos debe ser simulado, real_declarado o no_declarado")
        if (
            self.origen_datos == "real_declarado"
            and self.cov_transformacion is not None
            and np.all(self.cov_transformacion == 0)
            and not (
                self.pose_exacta_declarada is True
                and self.fuente_registro not in ("", "no_documentada")
            )
        ):
            raise ErrorDeEntrada(
                "Covarianza cero en pose real requiere registro.pose_exacta_declarada=true y fuente documental; si se desconoce, omite covarianza (null)"
            )
        if self.rangos_ng_esperados is not None:
            rangos = np.asarray(self.rangos_ng_esperados, float)
            if (
                rangos.shape != (n, 2)
                or not np.all(np.isfinite(rangos))
                or np.any(rangos[:, 0] <= 1)
                or np.any(rangos[:, 0] >= rangos[:, 1])
            ):
                raise ErrorDeEntrada(
                    "ng_rangos_esperados debe ser (N,2), finito creciente >1 por capa"
                )
            self.rangos_ng_esperados = rangos
        self.instrumento.validar()

    def nominal_en_xy(self, xy):
        xy = np.asarray(xy, float)
        return self.d_nom + ((xy - self.origen_nominal_xy) / UM) @ self.gradiente_nominal.T


def picos_punto(
    k: np.ndarray, I: np.ndarray, cfg: ConfigPipeline, d_esperado=None
) -> ResultadoPicos:
    if cfg.instrumento.limites_saturacion is not None:
        lo, hi = cfg.instrumento.limites_saturacion
        if np.any(np.asarray(I) <= lo) or np.any(np.asarray(I) >= hi):
            raise ErrorDeEntrada(
                "Señal saturada según límites ADC declarados: reduce exposición/ganancia y repite la adquisición"
            )
    off = cfg.instrumento.offset
    smin, smax = cfg.rango_frontal
    r0 = detectar_picos(k, I, 1, ventanas=[(2 * (off + smin), 2 * (off + smax))])
    L0 = r0.L[0]
    esperado = cfg.d_nom if d_esperado is None else np.asarray(d_esperado, float)
    if (
        esperado.shape != cfg.d_nom.shape
        or not np.all(np.isfinite(esperado))
        or np.any(esperado <= 0)
    ):
        raise ErrorDeEntrada("espesores esperados deben ser positivos finitos, uno por capa")
    Z = np.concatenate([[0.0], np.cumsum(cfg.ng_prior * esperado)])
    margen = 2.0 * (150.0 + 0.01 * Z)
    vent = [(L0 + 2 * z - m, L0 + 2 * z + m) for z, m in zip(Z, margen)]
    r = detectar_picos(k, I, len(Z), ventanas=vent)
    r.banderas.extend("frontal:" + b for b in r0.banderas)
    return r


def calibrar_desde_interferogramas(
    k, Is: Sequence[np.ndarray], d_ref, sigma_d_ref, cfg: ConfigPipeline, cov_d_ref=None
) -> Calibracion:
    res = [picos_punto(k, I, cfg, d_ref) for I in Is]
    return calibrar(res, d_ref, sigma_d_ref, cfg.grupos, cov_d_ref=cov_d_ref)


def evaluar_malla_en_xy(malla: Malla, X: np.ndarray, Y: np.ndarray):
    from matplotlib.tri import Triangulation

    X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
    try:
        tri = Triangulation(malla.v[:, 0], malla.v[:, 1], malla.f)
        indices = tri.get_trifinder()(X, Y)
    except (ValueError, RuntimeError) as e:
        raise ErrorDeEntrada("CAD debe ser una superficie z(x,y) sin caras superpuestas") from e
    nv = malla.normales_vertice()
    f = malla.f[np.maximum(indices, 0)]
    a, b, c = (malla.v[f[..., j], :2] for j in range(3))
    uv = np.linalg.solve(
        np.stack([b - a, c - a], axis=-1), (np.stack([X, Y], axis=-1) - a)[..., None]
    )[..., 0]
    pesos = np.stack([1 - uv[..., 0] - uv[..., 1], uv[..., 0], uv[..., 1]], axis=-1)
    Z = (pesos * malla.v[f, 2]).sum(axis=-1)
    normals = (pesos[..., None] * nv[f]).sum(axis=-2)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.where(indices >= 0, Z, np.nan), np.where((indices >= 0)[..., None], normals, np.nan)


@dataclass
class Resultado:
    ids_ok: np.ndarray
    ids_defecto: Dict[int, List[str]]
    banderas: List[str]
    registro: object
    d_puntos: np.ndarray
    sigma_aleatoria: np.ndarray
    sigma_sistematica: np.ndarray
    delta_puntos: np.ndarray
    q_registrado: np.ndarray
    Xg: np.ndarray
    Yg: np.ndarray
    d_media: np.ndarray
    d_std: np.ndarray
    delta_media: np.ndarray
    delta_std: np.ndarray
    cuna: Cuna
    w_mapa: np.ndarray
    sigma_w_mapa: np.ndarray
    P_fuera_d: np.ndarray
    P_fuera_w: np.ndarray
    falla_d: np.ndarray
    falla_w: np.ndarray
    interfaces: List[np.ndarray]
    resumen_gemelo: Dict[str, object]
    gps_d: List[ProcesoGaussiano2D]
    gps_delta: List[ProcesoGaussiano2D]
    mascara_soporte: np.ndarray
    d_std_aleatoria: np.ndarray
    delta_std_aleatoria: np.ndarray
    P_fuera_delta: Optional[np.ndarray]
    delta_diferencia_gp: np.ndarray
    estado_validacion: Dict[str, object] = field(default_factory=dict)
    incidencia: Dict[str, object] = field(default_factory=dict)
    presupuesto_incertidumbre: List[dict] = field(default_factory=list)
    tol_d_mapa: Optional[np.ndarray] = None
    tol_w_mapa: Optional[np.ndarray] = None
    tol_delta_mapa: Optional[np.ndarray] = None
    interfaces_nominales: Optional[np.ndarray] = None
    mascara_puntos: Optional[np.ndarray] = None


def ejecutar(
    mediciones: Sequence[Medicion],
    cfg: ConfigPipeline,
    cal: Calibracion,
    malla_cad: Malla,
    *,
    progreso=None,
    cancelar=None,
) -> Resultado:
    notificar(progreso, cancelar, 0.05, "Validando mediciones y calibración")
    N = cfg.d_nom.size
    banderas: List[str] = []
    if not cal.ok:
        raise ErrorDeEntrada("calibracion inconsistente: " + ",".join(cal.banderas))
    if len(cal.ng) != N or list(cal.grupos) != list(cfg.grupos):
        raise ErrorDeEntrada(
            "calibracion y configuracion no corresponden a las mismas capas/grupos"
        )
    if cfg.rangos_ng_esperados is not None:
        fuera_ng = (cal.ng < cfg.rangos_ng_esperados[:, 0]) | (
            cal.ng > cfg.rangos_ng_esperados[:, 1]
        )
        banderas.extend(
            f"indice_grupo_fuera_banda_declarada:capa{j}" for j in np.flatnonzero(fuera_ng)
        )
    if len({m.id for m in mediciones}) != len(mediciones):
        raise ErrorDeEntrada("los id de medicion deben ser unicos")
    res: List[ResultadoPicos] = []
    for i, m in enumerate(mediciones):
        notificar(
            progreso, cancelar, 0.05 + 0.25 * i / max(len(mediciones), 1), "Localizando interfaces"
        )
        m.validar()
        res.append(picos_punto(m.k, m.I, cfg))
    ok = np.array([r.ok for r in res])
    ids_def = {int(m.id): r.banderas for m, r in zip(mediciones, res) if not r.ok}
    if ok.sum() < 20:
        raise ErrorDeEntrada(
            f"solo {int(ok.sum())} puntos validos de {len(res)}: no hay base para un mapa (>= 20)"
        )
    if (~ok).any():
        banderas.append(f"puntos_descartados:{int((~ok).sum())}")
    med_ok = [m for m, o in zip(mediciones, ok) if o]
    res_ok = [r for r, o in zip(res, ok) if o]
    a = np.array([r.caminos_capas() for r in res_ok])
    L0 = np.array([r.L[0] for r in res_ok])
    Ccal = cal.cov_a if getattr(cal, "cov_a", None) is not None else np.diag(cal.sigma_a**2)
    cov_a = []
    for r in res_ok:
        v, U = np.linalg.eigh(r.cov_caminos() - Ccal)
        cov_a.append(Ccal + (U * np.maximum(v, 0)) @ U.T)
    cov_a = np.asarray(cov_a)
    sig_a = np.sqrt(np.diagonal(cov_a, axis1=-2, axis2=-1))
    esp = espesor_calibrado(a, sig_a, cal, cov_a=cov_a)
    d = esp.d
    notificar(progreso, cancelar, 0.32, "Reconstruyendo espesores y marco CAD")
    off = cfg.instrumento.offset
    pos = np.array([m.posicion for m in med_ok])
    n_in = np.array([m.normal for m in med_ok])
    q = pos + n_in * (L0 / 2.0 - off)[:, None]
    if cfg.modo_registro == "fijo":
        T = cfg.transformacion or Transformacion(np.eye(3), np.zeros(3))
        _, dist, _ = malla_cad.proyectar(T.aplicar(q))
        reg = ResultadoRegistro(
            T, float(np.sqrt(np.mean(dist**2))), 0, True, 1.0, covarianza=cfg.cov_transformacion
        )
        if cfg.cov_transformacion is None:
            banderas.append("incertidumbre_marco_fijo_no_declarada")
    else:
        reg = (
            registrar_icp(
                q,
                malla_cad,
                T0=cfg.transformacion,
                sigma_ruido=cfg.sigma_pos,
                sigma_prior_traslacion=cfg.prior_montaje[0],
                sigma_prior_rotacion=cfg.prior_montaje[1],
            )
            if cfg.prior_montaje
            else registrar_icp(q, malla_cad, T0=cfg.transformacion, sigma_ruido=cfg.sigma_pos)
        )
        banderas.append("cad_solo_desviacion_de_forma_ajuste_icp_absorbe_pose_y_error_global")
    banderas.extend("registro:" + b for b in reg.banderas)
    qr = reg.T.aplicar(q)
    nr = n_in @ reg.T.R.T
    nr /= np.linalg.norm(nr, axis=1, keepdims=True)
    Cnorm = [covarianza_normal(m.normal, m.cov_normal) for m in med_ok]
    Cnorm = [None if C is None else reg.T.R @ C @ reg.T.R.T for C in Cnorm]
    normal_declarada = all(C is not None for C in Cnorm)
    Qfrontal, _, tifrontal = malla_cad.proyectar(qr)
    Nfrontal = malla_cad.normales_interpoladas(Qfrontal, tifrontal)
    fase = (
        None
        if cfg.materiales_apriori is None
        else [float(m.n(cfg.instrumento.lam0)) for m in cfg.materiales_apriori]
    )
    incidencia = diagnostico_incidencia(nr, Nfrontal, d, fase, Cnorm)
    if not normal_declarada:
        banderas.append("incertidumbre_normal_sonda_no_declarada_o_incompleta")
    if incidencia["angulo_max_deg"] >= 1:
        banderas.append(
            "incidencia_oblicua_no_corregida_espesores_condicionados_al_modelo_casi_normal"
        )
    P = puntos_interfaces(qr, nr, d)
    cad_k = interfaces_nominales(malla_cad, cfg.nominal_en_xy(malla_cad.v[:, :2]).T)
    delta = np.zeros((N + 1, len(qr)))
    Qk = []
    for kk in range(N + 1):
        delta[kk], Q = desviacion_contra_cad(P[kk], cad_k[kk])
        Qk.append(Q)
    gps_delta, gps_d = [], []
    C_d = esp.cov_aleatoria
    sig_cum = np.stack(
        [np.sqrt(np.maximum(C_d[:, :k, :k].sum(axis=(1, 2)), 0)) for k in range(1, N + 1)], axis=1
    )
    for kk in range(N + 1):
        notificar(progreso, cancelar, 0.40 + 0.18 * kk / (N + 1), "Ajustando interfaces espaciales")
        s = np.sqrt(cfg.sigma_pos**2 + (sig_cum[:, kk - 1] ** 2 if kk > 0 else np.zeros(len(qr))))
        gps_delta.append(ProcesoGaussiano2D().ajustar(Qk[kk][:, :2] / UM, delta[kk], s))
    for j in range(N):
        notificar(progreso, cancelar, 0.58 + 0.10 * j / N, "Ajustando mapas de espesor")
        gps_d.append(
            ProcesoGaussiano2D().ajustar(qr[:, :2] / UM, d[:, j], esp.sigma_aleatoria[:, j])
        )
    nx, ny = cfg.grilla
    x0, x1 = qr[:, 0].min(), qr[:, 0].max()
    y0, y1 = qr[:, 1].min(), qr[:, 1].max()
    Xg, Yg = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    Xs = np.column_stack([Xg.ravel(), Yg.ravel()]) / UM
    from scipy.spatial import Delaunay

    mascara = (Delaunay(qr[:, :2] / UM).find_simplex(Xs, tol=1e-9) >= 0).reshape(ny, nx)
    mascara = aplicar_mascara(Xg, Yg, mascara, cfg.mascara_adquisicion)
    if not np.any(mascara[:-1, :-1] & mascara[1:, :-1] & mascara[:-1, 1:] & mascara[1:, 1:]):
        raise ErrorDeEntrada("grilla sin celdas interiores al soporte medido")
    mascara_puntos = np.ones(len(qr), bool)
    if cfg.mascara_adquisicion is not None:
        from scipy.interpolate import RegularGridInterpolator

        mascara_puntos = (
            RegularGridInterpolator(
                (Yg[:, 0], Xg[0]),
                mascara.astype(float),
                method="nearest",
                bounds_error=False,
                fill_value=0,
            )(qr[:, [1, 0]])
            > 0.5
        )
        if not mascara_puntos.any():
            raise ErrorDeEntrada(
                "La máscara no contiene ninguna adquisición válida; corrige marco/unidades o soporte"
            )
        if not mascara_puntos.all():
            banderas.append(
                "puntos_fuera_mascara_excluidos_de_resumen_y_exportacion;GP_condicionado_a_mediciones_validas_aportadas"
            )
    if not mascara.all():
        banderas.append("predicciones_fuera_soporte_excluidas_de_calidad_y_stl")
    notificar(progreso, cancelar, 0.70, "Prediciendo mapas dentro del soporte")
    d_media = np.zeros((N, ny, nx))
    d_std = np.zeros((N, ny, nx))
    for j in range(N):
        m_, s_ = gps_d[j].predecir(Xs)
        d_media[j], d_std[j] = m_.reshape(ny, nx), s_.reshape(ny, nx)
    dl_media = np.zeros((N + 1, ny, nx))
    dl_std = np.zeros((N + 1, ny, nx))
    for kk in range(N + 1):
        m_, s_ = gps_delta[kk].predecir(Xs)
        dl_media[kk], dl_std[kk] = m_.reshape(ny, nx), s_.reshape(ny, nx)
    d_std_al = d_std.copy()
    d_std = np.sqrt(d_std**2 + (d_media * cal.u_rel[:, None, None]) ** 2)
    dl_std_al = dl_std.copy()
    Cng = cal.cov_ng if getattr(cal, "cov_ng", None) is not None else np.diag(cal.sigma_ng**2)
    for kk in range(N + 1):
        if kk:
            coef = np.moveaxis(d_media[:kk], 0, -1) / cal.ng[:kk]
            dl_std[kk] = np.sqrt(
                dl_std[kk] ** 2 + np.einsum("...i,ij,...j->...", coef, Cng[:kk, :kk], coef)
            )
    cuna = cuna_pvb(gps_d[cfg.idx_pvb], Xs, direccion=cfg.direccion_cuna)
    cuna.sigma_w = np.sqrt(cuna.sigma_w**2 + (cuna.w * cal.u_rel[cfg.idx_pvb]) ** 2)
    w_mapa, sw_mapa = cuna.w.reshape(ny, nx), cuna.sigma_w.reshape(ny, nx)
    from scipy.spatial import cKDTree

    n_grid = LinearNDInterpolator(qr[:, :2], nr)(Xg, Yg)
    faltantes = ~np.isfinite(n_grid).all(axis=-1)
    if np.any(faltantes):
        cerca = cKDTree(qr[:, :2]).query(np.stack([Xg[faltantes], Yg[faltantes]], axis=-1))[1]
        n_grid[faltantes] = nr[cerca]
    n_grid /= np.linalg.norm(n_grid, axis=-1, keepdims=True)
    Zc, Nc = evaluar_malla_en_xy(cad_k[0], Xg, Yg)
    base = np.stack([Xg, Yg, Zc], axis=-1)
    S0 = base + dl_media[0][..., None] * Nc
    media_gp = dl_media.copy()
    frente_std = dl_std_al[0].copy()
    interfaces, delta_diferencia_gp = [], np.zeros_like(dl_media)
    Snom = np.full((N + 1, ny, nx, 3), np.nan)
    Cnormal_grid = None
    if normal_declarada:
        Cnormal_grid = LinearNDInterpolator(qr[:, :2], np.asarray(Cnorm))(Xg, Yg)
        faltan_cov = ~np.isfinite(Cnormal_grid).all(axis=(-1, -2))
        if faltan_cov.any():
            ind = cKDTree(qr[:, :2]).query(np.stack([Xg[faltan_cov], Yg[faltan_cov]], axis=-1))[1]
            Cnormal_grid[faltan_cov] = np.asarray(Cnorm)[ind]
        distancia_frontal_grid = LinearNDInterpolator(qr[:, :2], L0 / 2 - off)(Xg, Yg)
        distancia_frontal_grid = np.nan_to_num(
            distancia_frontal_grid, nan=float(np.max(L0 / 2 - off))
        )

    for kk in range(N + 1):
        D = d_media[:kk].sum(axis=0)
        S = S0 + D[..., None] * n_grid
        if not np.all(np.isfinite(S[mascara])):
            raise ErrorDeEntrada(
                "la rejilla sale del dominio del CAD: reduce la zona de prediccion"
            )
        interfaces.append(S)
        notificar(
            progreso, cancelar, 0.75 + 0.12 * kk / (N + 1), "Construyendo superficies coherentes"
        )
        Q, dist, ti = cad_k[kk].proyectar(S[mascara])
        Snom[kk][mascara] = Q
        dl_media[kk][mascara] = dist
        delta_diferencia_gp[kk] = dl_media[kk] - media_gp[kk]
        dl_std_al[kk] = frente_std + d_std_al[:kk].sum(axis=0)
        dl_std[kk] = dl_std_al[kk]
        if kk:
            coef = np.moveaxis(d_media[:kk], 0, -1) / cal.ng[:kk]
            dl_std[kk] = np.sqrt(
                dl_std[kk] ** 2 + np.einsum("...i,ij,...j->...", coef, Cng[:kk, :kk], coef)
            )
        normals = cad_k[kk].normales_interpoladas(Q, ti)
        if Cnormal_grid is not None:
            Jn = (distancia_frontal_grid[mascara] + D[mascara])[:, None] * normals
            vn = np.einsum("...i,...ij,...j->...", Jn, Cnormal_grid[mascara], Jn)
            dl_std[kk][mascara] = np.sqrt(dl_std[kk][mascara] ** 2 + np.maximum(vn, 0))
        if reg.covarianza is not None:
            J = np.concatenate([np.cross(S[mascara], normals), normals], axis=-1)
            vreg = np.einsum("...i,ij,...j->...", J, reg.covarianza, J)
            dl_std[kk][mascara] = np.sqrt(dl_std[kk][mascara] ** 2 + np.maximum(vreg, 0))
    notificar(progreso, cancelar, 0.90, "Evaluando tolerancias y límites de incertidumbre")
    td, tw, tdelta = mapas_tolerancia(
        Xg, Yg, cfg.tol_d, cfg.tol_w, cfg.tol_delta, cfg.zonas_tolerancia
    )
    Pd = np.zeros((N, ny, nx))
    nominal_grilla = cfg.nominal_en_xy(np.stack([Xg, Yg], axis=-1))
    for j in range(N):
        Pd[j] = probabilidad_con_mapa(d_media[j], d_std[j], nominal_grilla[..., j], td[j])
    Pw = probabilidad_con_mapa(w_mapa, sw_mapa, cfg.w_nom, tw)
    Pdelta = (
        None
        if (cfg.tol_delta is None or cfg.modo_registro != "fijo" or cfg.cov_transformacion is None)
        else np.array(
            [probabilidad_con_mapa(dl_media[k], dl_std[k], 0, tdelta[k]) for k in range(N + 1)]
        )
    )
    falla_d = np.array([decidir(Pd[j], cfg.umbral_falla).falla for j in range(N)])
    falla_w = decidir(Pw, cfg.umbral_falla).falla
    falla_d &= mascara[None]
    falla_w &= mascara
    resumen = _resumen_gemelo(cfg, cal, d[mascara_puntos], w_mapa[mascara])
    notificar(progreso, cancelar, 0.96, "Resultados calculados; preparando exportación")
    return Resultado(
        ids_ok=np.array([m.id for m in med_ok]),
        ids_defecto=ids_def,
        banderas=banderas,
        registro=reg,
        d_puntos=d,
        sigma_aleatoria=esp.sigma_aleatoria,
        sigma_sistematica=esp.sigma_sistematica,
        delta_puntos=delta,
        q_registrado=qr,
        Xg=Xg,
        Yg=Yg,
        d_media=d_media,
        d_std=d_std,
        delta_media=dl_media,
        delta_std=dl_std,
        cuna=cuna,
        w_mapa=w_mapa,
        sigma_w_mapa=sw_mapa,
        P_fuera_d=Pd,
        P_fuera_w=Pw,
        falla_d=falla_d,
        falla_w=falla_w,
        interfaces=interfaces,
        resumen_gemelo=resumen,
        gps_d=gps_d,
        gps_delta=gps_delta,
        mascara_soporte=mascara,
        d_std_aleatoria=d_std_al,
        delta_std_aleatoria=dl_std_al,
        P_fuera_delta=Pdelta,
        delta_diferencia_gp=delta_diferencia_gp,
        incidencia=incidencia,
        presupuesto_incertidumbre=presupuesto(cfg, cal, normal_declarada, incidencia),
        tol_d_mapa=td,
        tol_w_mapa=tw,
        tol_delta_mapa=tdelta,
        interfaces_nominales=Snom,
        mascara_puntos=mascara_puntos,
    )


def _resumen_gemelo(
    cfg: ConfigPipeline, cal: Calibracion, d: np.ndarray, w_mapa: np.ndarray
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["espesor_medio_um"] = [float(x) for x in d.mean(axis=0)]
    out["cuna_media_mrad"] = float(np.mean(w_mapa))
    out["cuna_min_max_mrad"] = [float(w_mapa.min()), float(w_mapa.max())]
    if cfg.materiales_apriori is None:
        return out
    lam0 = cfg.instrumento.lam0
    mats = [
        Desplazado(m, float(cal.ng[j] - float(m.n_grupo(lam0))))
        for j, m in enumerate(cfg.materiales_apriori)
    ]
    medida = [Capa(m, float(x)) for m, x in zip(mats, d.mean(axis=0))]
    nominal = [Capa(m, float(x)) for m, x in zip(mats, cfg.d_nom)]
    cg = ConfigGemelo(
        lam0=lam0, sigma_a=float(np.mean(cal.sigma_a)), sigma_lnA=None, covarianza=cal.cov_a
    )
    uniq = sorted(set(cfg.grupos))
    gi = [uniq.index(g) for g in cfg.grupos]
    indices = [cfg.grupos.index(u) for u in uniq]
    sens = sensibilidad(medida, cg, grupos=gi, cov_prior_dn=cal.cov_ng[np.ix_(indices, indices)])
    out["sensibilidad"] = {
        "parametros": sens.nombres,
        "sigma": [float(x) for x in sens.sigma],
        "minimo_detectable": [float(x) for x in sens.delta_min],
        "rango": sens.rango,
        "identificables": [bool(x) for x in sens.identificables],
    }
    dist = distinguibilidad(medida, nominal, cg)
    out["distinguibilidad_medida_vs_nominal"] = {
        "delta": dist.delta,
        "distinguibles_condicionales": bool(dist.distinguibles),
        "resolucion_y_detectabilidad_caracterizadas": dist.limites_observabilidad_especificados,
    }
    out["supuestos"] = [
        "distancia senal/ruido local; umbral 3 no es garantia de potencia de deteccion",
        "indice de fase y dispersion de catalogo, no identificados por calibracion de grupo",
        "comparacion de recetas con parametros fijos; no prueba estadistica de pieza",
    ]
    if cfg.hud is not None:
        w_rad = float(np.mean(w_mapa)) * 1e-3
        out["imagen_doble_arcmin"] = imagen_doble_receta(medida[::-1], w_rad, cfg.hud)
        out["cuna_optima_mrad"] = cuna_optima_receta(medida[::-1], cfg.hud) * 1e3
        out["eje_hud"] = (
            "direccion_cuna debe apuntar desde fuente hacia ojo en el plano de incidencia HUD"
        )
        out["incertidumbre_total_hud_arcmin"] = None
        out["estado_hud"] = (
            "prediccion_exploratoria_sin_validacion_optica_ni_presupuesto_completo_de_incertidumbre"
        )
    return out


def _serializable(valor):
    if isinstance(valor, dict):
        return {str(k): _serializable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, np.ndarray)):
        return [_serializable(v) for v in valor]
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (float, np.floating)):
        return float(valor) if np.isfinite(valor) else None
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    return valor


def _escribir_informe(informe, carpeta):
    import platform
    from importlib.metadata import PackageNotFoundError, version

    from motor import __version__
    from motor.estado import aplicar_estado

    informe = _serializable(informe)
    versiones = {}
    for paquete in ("numpy", "scipy", "matplotlib"):
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            versiones[paquete] = None
    informe["software"] = {
        "motor": __version__,
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "dependencias_nucleo": versiones,
    }
    informe = aplicar_estado(informe)
    with open(os.path.join(carpeta, "informe.json"), "w", encoding="utf-8") as fh:
        json.dump(informe, fh, indent=2, ensure_ascii=False, allow_nan=False)
    return informe


def _estado_validacion(cfg, val=None, independencia=False):
    origen = cfg.origen_datos
    return {
        "origen_datos": origen,
        "origen_es_declaracion_del_usuario": True,
        "segunda_muestra": (
            "no_aportada" if val is None else ("compatible" if val.ok else "incompatible")
        ),
        "independencia_documentada": bool(independencia),
        "validacion_experimental_completa": False,
        "decision_de_liberacion": "no_emitida",
        "motivo": "resultados de investigacion; requiere protocolo experimental, trazabilidad y validacion de cobertura",
    }


def _resumen_validacion(val):
    return {
        "z": val.z,
        "ok": bool(val.ok),
        "error_um": val.error,
        "d_estimado_um": val.d_estimado,
        "sigma_combinada_um": val.sigma_d,
        "compatible_por_capa": [
            bool(np.isfinite(x) and abs(x) < 3.0) for x in np.atleast_1d(val.z)
        ],
        "criterio": "abs(z)<3 por capa; contraste puntual, no cobertura experimental global",
    }


def guardar(
    res: Resultado, cfg: ConfigPipeline, cal: Calibracion, carpeta: str
) -> Dict[str, object]:
    os.makedirs(carpeta, exist_ok=True)
    N = cfg.d_nom.size
    rep = exportar_stl(
        os.path.join(carpeta, "cad"), res.interfaces, cfg.nombres_capas, mascara=res.mascara_soporte
    )

    def soportado(a, vector=False):
        a = np.asarray(a, float).copy()
        if vector:
            a[..., ~res.mascara_soporte, :] = np.nan
        else:
            a[..., ~res.mascara_soporte] = np.nan
        return a

    mp = np.ones(len(res.ids_ok), bool) if res.mascara_puntos is None else res.mascara_puntos
    np.savez(
        os.path.join(carpeta, "mapas.npz"),
        X=res.Xg,
        Y=res.Yg,
        d_media=soportado(res.d_media),
        d_std=soportado(res.d_std),
        delta_media=soportado(res.delta_media),
        delta_std=soportado(res.delta_std),
        w=soportado(res.w_mapa),
        sigma_w=soportado(res.sigma_w_mapa),
        P_fuera_d=soportado(res.P_fuera_d),
        P_fuera_w=soportado(res.P_fuera_w),
        mascara_soporte=res.mascara_soporte,
        d_std_aleatoria=soportado(res.d_std_aleatoria),
        delta_std_aleatoria=soportado(res.delta_std_aleatoria),
        gradiente_cuna=np.where(
            res.mascara_soporte[..., None], res.cuna.grad.reshape(*res.Xg.shape, 2), np.nan
        ),
        cov_gradiente_aleatoria=np.where(
            res.mascara_soporte[..., None, None],
            res.cuna.cov_grad.reshape(*res.Xg.shape, 2, 2),
            np.nan,
        ),
        P_fuera_delta=np.empty(0) if res.P_fuera_delta is None else soportado(res.P_fuera_delta),
        ids_validos=res.ids_ok[mp],
        d_puntos=res.d_puntos[mp],
        delta_puntos=res.delta_puntos[:, mp],
        delta_diferencia_gp=soportado(res.delta_diferencia_gp),
        interfaces=soportado(np.stack(res.interfaces), vector=True),
        interfaces_nominales=res.interfaces_nominales,
        tol_d=soportado(res.tol_d_mapa),
        tol_w=soportado(res.tol_w_mapa),
        tol_delta=np.empty(0) if res.tol_delta_mapa is None else soportado(res.tol_delta_mapa),
    )
    mask = res.mascara_soporte
    informe = {
        "estado_validacion": res.estado_validacion or _estado_validacion(cfg),
        "unidades": {
            "longitudes": "um",
            "STL": "um (STL no almacena unidades)",
            "cuna": "mrad",
            "incertidumbre": "estandar 1 sigma",
        },
        "soporte": {
            "nodos_evaluados": int(mask.sum()),
            "nodos_fuera_del_soporte": int((~mask).sum()),
            "mascara_explicita": cfg.mascara_adquisicion is not None,
            "limite": "envolvente convexa intersectada con máscara de adquisición; soporte de nodos a resolución de rejilla",
        },
        "regla_calidad": {
            "umbral_P_fuera": cfg.umbral_falla,
            "interpretacion": "riesgo puntual bajo modelo normal; no probabilidad global de rechazo ni pieza liberada",
            "fracciones": "proporcion de nodos soportados, no area superficial exacta",
        },
        "incertidumbre_superficies": {
            "aleatoria": "cota conservadora std_frente + suma std_espesor; correlacion de campos GP no estimada",
            "comun": "covarianza indices y marco rigido propagada despues del ajuste; sin reduccion por numero de puntos",
            "limites": "presupuesto parcial conocido; dirección de sonda propagada solo si se aporta cov_normal; sin discrepancia física medida",
            "total_experimental_disponible": False,
            "sigma_total_experimental_um": None,
        },
        "incidencia": res.incidencia,
        "presupuesto_incertidumbre": res.presupuesto_incertidumbre,
        "consistencia_superficies": {
            "espesor_STL": "norma(S_j-S_j-1) = d_media_j en cada vertice",
            "diferencia_rms_vs_GP_delta_um": [
                float(np.sqrt(np.mean(v[mask] ** 2))) for v in res.delta_diferencia_gp
            ],
        },
        "puntos_validos": int(mp.sum()),
        "puntos_validos_espectrales": int(len(res.ids_ok)),
        "puntos_con_defecto": {str(k): v for k, v in res.ids_defecto.items()},
        "banderas": res.banderas,
        "calibracion": {
            "ng": [float(x) for x in cal.ng],
            "u_rel_sistematica": [float(x) for x in cal.u_rel],
            "sigma_a_um": [float(x) for x in cal.sigma_a],
            "repeticiones": cal.m_repeticiones,
            "ok": bool(cal.ok),
            "banderas": cal.banderas,
            "cov_a_um2": cal.cov_a,
            "cov_ng": cal.cov_ng,
            "referencia": cal.metadatos_referencia,
        },
        "registro": {
            "rms_um": res.registro.rms,
            "iteraciones": res.registro.iteraciones,
            "convergio": bool(res.registro.convergio),
            "cond": float(res.registro.cond),
            "banderas": res.registro.banderas,
            "modo": cfg.modo_registro,
            "desviacion_absoluta_disponible": cfg.modo_registro == "fijo",
            "incertidumbre_marco_declarada": (
                cfg.cov_transformacion is not None if cfg.modo_registro == "fijo" else False
            ),
            "decision_CAD_absoluta_evaluable": res.P_fuera_delta is not None,
            "covarianza": res.registro.covarianza,
            "rotacion": res.registro.T.R,
            "traslacion_um": [float(x) for x in res.registro.T.t],
        },
        "espesor": {
            cfg.nombres_capas[j]: {
                "media_um": float(res.d_media[j][mask].mean()),
                "min_um": float(res.d_media[j][mask].min()),
                "max_um": float(res.d_media[j][mask].max()),
                "sigma_total_media_um": float(res.d_std[j][mask].mean()),
                "sigma_parcial_conocida_media_um": float(res.d_std[j][mask].mean()),
                "sigma_total_experimental_um": None,
                "sigma_aleatoria_um": float(res.sigma_aleatoria[mp, j].mean()),
                "sigma_sistematica_um": float(res.sigma_sistematica[mp, j].mean()),
                "fraccion_zona_fuera_de_tolerancia": float(res.falla_d[j][mask].mean()),
            }
            for j in range(N)
        },
        "cuna_pvb_mrad": {
            "indice_capa": cfg.idx_pvb,
            "media": float(res.w_mapa[mask].mean()),
            "min": float(res.w_mapa[mask].min()),
            "max": float(res.w_mapa[mask].max()),
            "sigma_media": float(res.sigma_w_mapa[mask].mean()),
            "direccion": cfg.direccion_cuna,
            "fraccion_zona_fuera_de_tolerancia": float(res.falla_w[mask].mean()),
        },
        "desviacion_vs_cad_um": {
            f"interfaz_{k}": {
                "media": float(res.delta_media[k][mask].mean()),
                "rms": float(np.sqrt(np.mean(res.delta_media[k][mask] ** 2))),
                "sigma_total_media": float(res.delta_std[k][mask].mean()),
                "sigma_parcial_conocida_media": float(res.delta_std[k][mask].mean()),
                "sigma_total_experimental": None,
                "P_fuera_media": (
                    None if res.P_fuera_delta is None else float(res.P_fuera_delta[k][mask].mean())
                ),
            }
            for k in range(N + 1)
        },
        "gemelo": res.resumen_gemelo,
        "cad": {
            "archivos": rep.archivos,
            "volumenes_um3": rep.volumenes_um3,
            "solidos_cerrados": rep.cerradas,
        },
    }
    informe = _escribir_informe(informe, carpeta)
    from motor.salida.visualizacion import crear_mapas

    crear_mapas(informe, os.path.join(carpeta, "mapas.npz"), os.path.join(carpeta, "mapas.png"))
    return informe


def configuracion_demo() -> Tuple[ConfigPipeline, "object"]:
    from motor.analisis import demo

    cfg = ConfigPipeline(
        instrumento=demo.INSTR,
        d_nom=demo.D_NOM,
        tol_d=[10.0, 25.0, 10.0],
        prior_montaje=(150.0, 1e-3),
        modo_registro="fijo",
        transformacion=demo.transformacion_montaje(),
        cov_transformacion=np.zeros((6, 6)),
        gradiente_nominal=demo.GRADIENTE_NOMINAL,
        tol_delta=[15.0] * 4,
        origen_datos="simulado",
        ng_prior=[float(m.n_grupo(demo.LAM0)) for m in demo.MATERIALES_APRIORI],
        grupos=["vidrio", "pvb", "vidrio"],
        w_nom=0.35,
        tol_w=0.10,
        grilla=(31, 31),
        rango_frontal=(demo.STANDOFF - 1500, demo.STANDOFF + 1500),
        materiales_apriori=demo.MATERIALES_APRIORI,
        hud=ParametrosHUD.ejemplo(),
    )
    return cfg, demo


def imprimir_resumen(informe):
    def cifra(valor, formato):
        return "n/d" if valor is None else format(valor, formato)

    print(f"Estado: {informe.get('veredicto', 'no concluyente')}")
    for nombre, v in informe.get("espesor", {}).items():
        print(
            f"{nombre}: {cifra(v.get('media_um'), '.2f')} ± "
            f"{cifra(v.get('sigma_aleatoria_um'), '.3f')} µm"
        )
    cuna = informe.get("cuna_pvb_mrad", {})
    print(f"Cuña PVB: {cifra(cuna.get('media'), '.3f')} mrad")


def correr_demo(
    carpeta: str,
    ruido: float = 0.01,
    semilla: int = 0,
    n_lado: int = 13,
    verbose: bool = True,
    *,
    progreso=None,
    cancelar=None,
):
    notificar(progreso, cancelar, 0.0, "Generando escenario sintético reproducible")
    cfg, demo = configuracion_demo()
    esc = demo.generar_escenario(n_lado=n_lado, ruido=ruido, semilla=semilla)
    d_ref, sd_ref = np.array([2100.0, 760.0, 1600.0]), np.array([1.0, 1.0, 1.0])
    k, Is = demo.muestra_referencia(d_ref, 30, ruido, semilla + 1000)
    cal = calibrar_desde_interferogramas(k, Is, d_ref, sd_ref, cfg)
    d2 = np.array([2050.0, 800.0, 1650.0])
    k2, Is2 = demo.muestra_referencia(d2, 10, ruido, semilla + 2000)
    val = validar_con_muestra(cal, [picos_punto(k2, I, cfg, d2) for I in Is2], d2, sd_ref)
    res = ejecutar(esc.mediciones, cfg, cal, demo.malla_cad(), progreso=progreso, cancelar=cancelar)
    res.estado_validacion = _estado_validacion(cfg, val, independencia=True)
    notificar(progreso, cancelar, 0.98, "Exportando archivos y mapas")
    informe = guardar(res, cfg, cal, carpeta)
    informe["validacion_cruzada"] = _resumen_validacion(val)
    informe["trazabilidad"] = {
        "semilla": int(semilla),
        "origen": "escenario simulado independiente parcial",
    }
    informe = _escribir_informe(informe, carpeta)
    if verbose:
        imprimir_resumen(informe)
    if progreso is not None:
        progreso(1.0, "Exportación finalizada")
    return esc, cfg, cal, res, informe


def _cargar_npz(ruta: str, claves: Sequence[str], remuestrear=False):
    if not os.path.exists(ruta):
        raise ErrorDeEntrada(f"no existe el archivo: {ruta}")
    try:
        with np.load(ruta, allow_pickle=False) as archivo:
            z = {k: archivo[k] for k in archivo.files}
    except (ValueError, OSError) as e:
        raise ErrorDeEntrada(f"no se puede leer NPZ numerico {ruta}: {e}") from e
    if "k" in claves:
        z = preparar_espectros_archivo(z, remuestrear=remuestrear)
    faltan = [c for c in claves if c not in z]
    if faltan:
        raise ErrorDeEntrada(f"{ruta}: faltan las claves {faltan} (tiene {list(z)})")
    if "I" in z and (z["I"].ndim != 2 or z["I"].shape[0] == 0):
        raise ErrorDeEntrada(f"{ruta}: I debe ser una matriz (adquisiciones,muestras)")
    if "k" in z and (z["k"].ndim != 1 or z["I"].shape[1] != len(z["k"])):
        raise ErrorDeEntrada(f"{ruta}: k e I tienen dimensiones incompatibles")
    return z


def ejecutar_desde_config(ruta_config: str, verbose: bool = True, *, progreso=None, cancelar=None):
    try:
        return _ejecutar_desde_config(ruta_config, verbose, progreso=progreso, cancelar=cancelar)
    except ErrorDeEntrada:
        raise
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as e:
        raise ErrorDeEntrada(f"configuracion/archivo invalido: {e}") from e


def _ejecutar_desde_config(ruta_config: str, verbose: bool = True, *, progreso=None, cancelar=None):
    from motor.medicion.materiales import BK7, PVB_SUPUESTO, SODOCALCICO_SUPUESTO

    catalogo = {
        "BK7": BK7,
        "PVB_SUPUESTO": PVB_SUPUESTO,
        "SODOCALCICO_SUPUESTO": SODOCALCICO_SUPUESTO,
    }
    base = os.path.dirname(os.path.abspath(ruta_config))
    ruta = lambda r: r if os.path.isabs(r) else os.path.join(base, r)
    with open(ruta_config, encoding="utf-8") as fh:
        c = json.load(fh)
    for clave in (
        "mediciones",
        "calibracion",
        "cad",
        "instrumento",
        "capas",
        "cuna",
        "rango_frontal",
    ):
        if clave not in c:
            raise ErrorDeEntrada(f"config.json: falta la clave '{clave}'")
    ins = Instrumento(
        lam0=float(c["instrumento"]["lam0"]),
        offset=float(c["instrumento"]["offset"]),
        limites_saturacion=c["instrumento"].get("limites_saturacion"),
    )
    ca = c["capas"]
    if c.get("materiales_apriori") and any(n not in catalogo for n in c["materiales_apriori"]):
        raise ErrorDeEntrada(f"materiales_apriori debe usar nombres de {list(catalogo)}")
    mats = [catalogo[n] for n in c["materiales_apriori"]] if c.get("materiales_apriori") else None
    hud = None
    if c.get("hud"):
        hud = ParametrosHUD(
            theta_i=np.deg2rad(c["hud"]["theta_i_deg"]),
            Rp=c["hud"]["Rp"],
            Rv=c["hud"]["Rv"],
            lam=c["hud"].get("lam", 0.55),
        )
    regc = c.get("registro")
    if not isinstance(regc, dict) or regc.get("modo") not in ("fijo", "icp_forma"):
        raise ErrorDeEntrada(
            "config.json: declara registro.modo fijo (R,t de datums independientes) o icp_forma"
        )
    if regc["modo"] == "fijo" and ("R" not in regc or "t" not in regc):
        raise ErrorDeEntrada(
            "registro fijo requiere R y t; usa identidad explicita solo si ya compartes marco CAD"
        )
    T = Transformacion(regc["R"], regc["t"]) if "R" in regc and "t" in regc else None
    cfg = ConfigPipeline(
        instrumento=ins,
        d_nom=ca["d_nom"],
        tol_d=ca["tol_d"],
        ng_prior=ca["ng_prior"],
        grupos=ca["grupos"],
        nombres_capas=ca.get("nombres", [f"capa{j + 1}" for j in range(len(ca["d_nom"]))]),
        idx_pvb=ca.get("idx_pvb", 1),
        w_nom=c["cuna"]["w_nom"],
        tol_w=c["cuna"]["tol"],
        rango_frontal=tuple(c["rango_frontal"]),
        materiales_apriori=mats,
        hud=hud,
        prior_montaje=tuple(c["prior_montaje"]) if c.get("prior_montaje") else None,
        sigma_pos=c.get("sigma_pos", 3.0),
        grilla=tuple(c.get("grilla", (31, 31))),
    )
    cfg.modo_registro = regc["modo"]
    cfg.transformacion, cfg.cov_transformacion = T, regc.get("covarianza")
    cfg.direccion_cuna = tuple(c.get("direccion_cuna", (1, 0)))
    cfg.gradiente_nominal = ca.get("gradiente_nominal")
    cfg.origen_nominal_xy = tuple(ca.get("origen_nominal_xy", (0, 0)))
    cfg.tol_delta, cfg.origen_datos = c.get("tol_delta"), c.get("origen_datos", "no_declarado")
    cfg.mascara_adquisicion = cargar_especificacion(c.get("mascara_adquisicion"), base)
    cfg.zonas_tolerancia = c.get("zonas_tolerancia", [])
    cfg.presupuesto_declarado = c.get("presupuesto_incertidumbre", {})
    cfg.pose_exacta_declarada = regc.get("pose_exacta_declarada", False)
    cfg.fuente_registro = regc.get("fuente", "no_documentada")
    cfg.rangos_ng_esperados = ca.get("ng_rangos_esperados")
    if cfg.origen_datos == "real_declarado" and "sigma_pos" not in c:
        raise ErrorDeEntrada(
            "Datos reales: falta sigma_pos en config.json; declara la incertidumbre de posición en µm del montaje, no se asigna 3 µm por defecto"
        )
    cfg.__post_init__()
    notificar(progreso, cancelar, 0.01, "Calibrando con muestra A")
    remuestrear = c.get("remuestrear_espectro", False)
    if not isinstance(remuestrear, bool):
        raise ErrorDeEntrada("remuestrear_espectro debe ser booleano")
    zc = _cargar_npz(
        ruta(c["calibracion"]["archivo"]), ("k", "I", "d_ref", "sigma_d_ref"), remuestrear
    )
    cal = calibrar_desde_interferogramas(
        zc["k"], list(zc["I"]), zc["d_ref"], zc["sigma_d_ref"], cfg, zc.get("cov_d_ref")
    )
    val = None
    independiente = False
    if c["calibracion"].get("validacion"):
        import hashlib

        def huella(archivo):
            h = hashlib.sha256()
            with open(archivo, "rb") as fh:
                for bloque in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(bloque)
            return h.hexdigest()

        fref, fval = ruta(c["calibracion"]["archivo"]), ruta(c["calibracion"]["validacion"])
        refid, valid = c["calibracion"].get("muestra_id"), c["calibracion"].get(
            "validacion_muestra_id"
        )
        if (
            os.path.realpath(fref) == os.path.realpath(fval)
            or huella(fref) == huella(fval)
            or (refid is not None and refid == valid)
        ):
            raise ErrorDeEntrada(
                "la segunda muestra no puede reutilizar el archivo o identificador de calibracion"
            )
        independiente = bool(refid and valid and refid != valid)
        zv = _cargar_npz(
            ruta(c["calibracion"]["validacion"]), ("k", "I", "d_ref", "sigma_d_ref"), remuestrear
        )
        if np.array_equal(zc["I"], zv["I"]) and np.array_equal(zc["k"], zv["k"]):
            raise ErrorDeEntrada("validacion reutiliza las mismas senales de calibracion")
        val = validar_con_muestra(
            cal,
            [picos_punto(zv["k"], I, cfg, zv["d_ref"]) for I in zv["I"]],
            zv["d_ref"],
            zv["sigma_d_ref"],
        )
    zm = _cargar_npz(ruta(c["mediciones"]), ("k", "I", "posicion", "normal"), remuestrear)
    M = len(zm["I"])
    if zm["posicion"].shape != (M, 3) or zm["normal"].shape != (M, 3):
        raise ErrorDeEntrada("NPZ mediciones: posicion y normal deben ser (M,3)")
    ids = zm["id"] if "id" in zm else np.arange(M)
    if ids.shape != (M,) or ids.dtype.kind not in "iu":
        raise ErrorDeEntrada("NPZ mediciones: id debe ser un vector entero (M,)")
    if "cov_normal" in zm and zm["cov_normal"].shape not in ((3, 3), (M, 3, 3)):
        raise ErrorDeEntrada(
            "NPZ mediciones: cov_normal debe ser (3,3) común o (M,3,3), adimensional²"
        )
    covn = zm.get("cov_normal")
    meds = [
        Medicion(
            k=zm["k"],
            I=zm["I"][i],
            posicion=zm["posicion"][i],
            normal=zm["normal"][i],
            id=int(ids[i]),
            cov_normal=None if covn is None else (covn if covn.ndim == 2 else covn[i]),
        )
        for i in range(len(zm["I"]))
    ]
    malla = leer_stl(ruta(c["cad"]))
    escala = float(c.get("escala_cad", 1.0))
    if not np.isfinite(escala) or escala <= 0:
        raise ErrorDeEntrada("escala_cad debe ser > 0")
    malla = Malla(malla.v * escala, malla.f)
    res = ejecutar(meds, cfg, cal, malla, progreso=progreso, cancelar=cancelar)
    res.estado_validacion = _estado_validacion(cfg, val, independiente)
    if val is not None and not val.ok:
        res.banderas.append("validacion_segunda_muestra_fallida_resultados_no_liberables")
    salida = ruta(c.get("salida", "salida"))
    notificar(progreso, cancelar, 0.98, "Exportando archivos y mapas")
    informe = guardar(res, cfg, cal, salida)
    if val is not None:
        informe["validacion_cruzada"] = _resumen_validacion(val)
    informe["trazabilidad"] = {
        "semilla": c.get("semilla"),
        "config": os.path.abspath(ruta_config),
        "registro_fuente": regc.get("fuente", "no_documentada"),
        "muestra_calibracion": c["calibracion"].get("muestra_id"),
        "muestra_validacion": c["calibracion"].get("validacion_muestra_id"),
        "remuestreo_espectral_habilitado": remuestrear,
        "limite_remuestreo": "interpolación correlaciona ruido; calibración y repetibilidad deben usar la misma rejilla",
    }
    informe = _escribir_informe(informe, salida)
    if verbose:
        imprimir_resumen(informe)
    if progreso is not None:
        progreso(1.0, "Exportación finalizada")
    return res, informe


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Motor del gemelo digital de vidrio laminado")
    ap.add_argument("--demo", action="store_true", help="ejecuta el escenario sintetico completo")
    ap.add_argument("--config", help="ruta a config.json para correr con datos reales")
    ap.add_argument(
        "--generar-ejemplo",
        metavar="CARPETA",
        help="escribe un conjunto de archivos de ejemplo (formato de datos reales) y su config.json",
    )
    ap.add_argument("--salida", default="salida_demo", help="carpeta de resultados")
    ap.add_argument("--ruido", type=float, default=0.01)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--n-lado", type=int, default=13)
    ap.add_argument("--variante", choices=("base", "conforme", "defecto"), default="base")
    a = ap.parse_args(argv)
    if a.demo:
        correr_demo(a.salida, a.ruido, a.semilla, a.n_lado)
        print(f"\nResultados en: {os.path.abspath(a.salida)}")
        return 0
    if a.generar_ejemplo:
        from motor.analisis import demo

        ruta = demo.guardar_escenario(a.generar_ejemplo, a.ruido, a.semilla, a.n_lado, a.variante)
        print(f"Ejemplo escrito. Corre:  python main.py analizar {ruta}")
        return 0
    if a.config:
        try:
            ejecutar_desde_config(a.config)
        except ErrorDeEntrada as e:
            print(f"ERROR DE ENTRADA: {e}", file=sys.stderr)
            return 2
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
