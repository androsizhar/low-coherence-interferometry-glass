from __future__ import annotations

import numpy as np

from motor.medicion.lectura import ErrorDeEntrada


class CalculoCancelado(RuntimeError):
    pass


def notificar(progreso=None, cancelar=None, fraccion=0.0, texto=""):
    if cancelar is not None and cancelar():
        raise CalculoCancelado("Cálculo cancelado por el usuario; no se emite una decisión nueva")
    if progreso is not None:
        progreso(float(fraccion), texto)


def covarianza_normal(normal, cov):
    if cov is None:
        return None
    C = np.asarray(cov, float)
    if (
        C.shape != (3, 3)
        or not np.all(np.isfinite(C))
        or not np.allclose(C, C.T, atol=1e-12)
        or np.linalg.eigvalsh(C).min() < -1e-12
    ):
        raise ErrorDeEntrada(
            "cov_normal debe ser simétrica semidefinida positiva (3,3), adimensional²"
        )
    n = np.asarray(normal, float)
    P = np.eye(3) - np.outer(n, n)
    return P @ C @ P


def diagnostico_incidencia(n_in, normal_cad, d, indices_fase=None, cov_normales=None):
    n_in, normal_cad, d = map(lambda a: np.asarray(a, float), (n_in, normal_cad, d))
    producto = np.sum(n_in * normal_cad, axis=-1)
    if np.any(producto >= 0):
        raise ErrorDeEntrada(
            "La normal de sonda debe apuntar hacia el vidrio; invierte su dirección y verifica la orientación exterior del CAD"
        )
    coseno = np.clip(-producto, 0.0, 1.0)
    theta = np.arccos(coseno)
    exceso = None
    if indices_fase is not None:
        indices = np.asarray(indices_fase, float)
        if (
            indices.shape != (d.shape[1],)
            or not np.all(np.isfinite(indices))
            or np.any(indices <= 1)
        ):
            raise ErrorDeEntrada("índices de fase para incidencia: uno por capa, finitos > 1")
        theta_t = np.arcsin(np.sin(theta[:, None]) / indices)
        exceso = d * (1 / np.cos(theta_t) - 1)
    sigma_theta = None
    if cov_normales is not None and all(C is not None for C in cov_normales):
        valores = []
        for n, nc, ang, C in zip(n_in, normal_cad, theta, cov_normales):
            if abs(np.sin(ang)) < 1e-7:
                valores.append(np.sqrt(max(float(np.linalg.eigvalsh(C).max()), 0)))
            else:
                J = nc / np.sin(ang)
                valores.append(np.sqrt(max(float(J @ C @ J), 0)))
        sigma_theta = np.rad2deg(valores)
    return {
        "correccion_aplicada": False,
        "modelo": "estimación geométrica de recorrido d·(sec(asin(sin(θ_aire)/n_fase))−1); no respuesta espectral del equipo",
        "angulo_max_deg": float(np.rad2deg(theta).max()),
        "angulo_medio_deg": float(np.rad2deg(theta).mean()),
        "angulo_por_punto_deg": np.rad2deg(theta),
        "sigma_angulo_sonda_deg": sigma_theta,
        "normal_CAD_considerada_exacta_en_diagnostico": True,
        "exceso_camino_geometrico_max_um_por_capa": None if exceso is None else exceso.max(axis=0),
        "indices_fase_usados": None if indices_fase is None else indices_fase,
        "advertencia": (
            "Incidencia ≥1°: espesores e interfaces condicionados al modelo casi normal; medir respuesta oblicua antes de decidir calidad"
            if np.rad2deg(theta).max() >= 1
            else "Incidencia casi normal; modelo oblicuo del equipo no validado"
        ),
        "incertidumbre_correccion_oblicua_um": None,
        "incertidumbre_espesor_por_incidencia_um": None,
        "limite_cov_normal": "cov_normal propagada a posición/desviación CAD de primer orden; no al sesgo de espesor oblicuo no calibrado",
    }


def presupuesto(cfg, cal, normal_declarada, incidencia):
    declaraciones = cfg.presupuesto_declarado or {}
    if not isinstance(declaraciones, dict):
        raise ErrorDeEntrada("presupuesto_incertidumbre debe ser un objeto por componente")
    filas = [
        {
            "componente": "referencia",
            "sigma": cal.metadatos_referencia.get("sigma_d_ref_um"),
            "covarianza": cal.metadatos_referencia.get("cov_d_ref_um2"),
            "unidad": "µm",
            "estado": "incluida en cov_ng; no sumar de nuevo",
            "propagada": True,
            "detalle": "covarianza del espesor certificado aportado a calibrar; consultar archivo de referencia",
        },
        {
            "componente": "grupo_optico",
            "sigma": np.sqrt(np.maximum(np.diag(cal.cov_ng), 0)).tolist(),
            "unidad": "1",
            "estado": "parcial conocida",
            "propagada": True,
            "detalle": "índice común, incluye referencia y media de repeticiones; covarianza entre capas conservada",
        },
        {
            "componente": "eje_espectral",
            "sigma": None,
            "unidad": "µm",
            "estado": "no disponible",
            "propagada": False,
        },
        {
            "componente": "offset",
            "sigma": None,
            "unidad": "µm",
            "estado": "no disponible",
            "propagada": False,
        },
        {
            "componente": "deriva",
            "sigma": None,
            "unidad": "µm",
            "estado": "no disponible",
            "propagada": False,
        },
        {
            "componente": "montaje",
            "sigma": None,
            "unidad": "rad y µm",
            "estado": (
                "covarianza declarada" if cfg.cov_transformacion is not None else "no disponible"
            ),
            "propagada": cfg.cov_transformacion is not None,
            "covarianza": (
                None if cfg.cov_transformacion is None else cfg.cov_transformacion.tolist()
            ),
            "sigma_pos_punto_um": float(cfg.sigma_pos),
        },
        {
            "componente": "normal_sonda",
            "sigma": (
                None
                if incidencia.get("sigma_angulo_sonda_deg") is None
                else float(np.max(incidencia["sigma_angulo_sonda_deg"]))
            ),
            "unidad": "°; máximo de ángulo de sonda",
            "estado": (
                "covarianza declarada; cota geométrica propagada a CAD, espesor por oblicuidad desconocido"
                if normal_declarada
                else "no disponible"
            ),
            "propagada": bool(normal_declarada),
        },
        {
            "componente": "repetibilidad",
            "sigma": cal.sigma_a.tolist(),
            "unidad": "µm de camino óptico",
            "estado": "estimada de repeticiones; piso de ruido",
            "propagada": True,
        },
        {
            "componente": "interpolacion",
            "sigma": None,
            "unidad": "µm",
            "estado": "GP condicionado; hiperparámetros sin incertidumbre",
            "propagada": True,
        },
        {
            "componente": "modelo",
            "sigma": None,
            "unidad": "µm",
            "estado": "no disponible; incluye respuesta oblicua y discrepancia física",
            "propagada": False,
        },
    ]
    permitidos = {fila["componente"] for fila in filas}
    if set(declaraciones) - permitidos:
        raise ErrorDeEntrada(
            "presupuesto_incertidumbre contiene componentes desconocidos: "
            + ", ".join(sorted(set(declaraciones) - permitidos))
        )
    for fila in filas:
        if fila["componente"] in declaraciones:
            evidencia = declaraciones[fila["componente"]]
            if (
                not isinstance(evidencia, dict)
                or not isinstance(evidencia.get("fuente"), str)
                or not evidencia["fuente"].strip()
            ):
                raise ErrorDeEntrada(
                    f"presupuesto {fila['componente']}: declara objeto con fuente documental y sigma/unidad"
                )
            sigma = evidencia.get("sigma")
            if sigma is not None:
                try:
                    a = np.asarray(sigma, float)
                except (TypeError, ValueError) as exc:
                    raise ErrorDeEntrada("sigma del presupuesto debe ser numérica o null") from exc
                if not np.all(np.isfinite(a)) or np.any(a < 0):
                    raise ErrorDeEntrada("sigma del presupuesto debe ser finita no negativa o null")
            fila["declaracion_adicional"] = evidencia
            fila["declaracion_adicional_integrada"] = False
    return filas
