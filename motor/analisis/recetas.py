from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from motor.analisis.gemelo import (
    ConfigGemelo,
    ParametrosHUD,
    covarianza_observables,
    distinguibilidad,
    predecir_imagen_doble,
    sensibilidad,
)
from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.materiales import BK7, PVB_SUPUESTO, SODOCALCICO_SUPUESTO, Cauchy
from motor.medicion.modelo_directo import Capa, camino_optico_grupo

CATALOGO = {"BK7": BK7, "PVB_SUPUESTO": PVB_SUPUESTO, "SODOCALCICO_SUPUESTO": SODOCALCICO_SUPUESTO}
UMBRAL = 3.0


def _objeto(valor, nombre, permitidas, requeridas=()):
    if not isinstance(valor, dict):
        raise ErrorDeEntrada(f"{nombre} debe ser un objeto JSON")
    faltan, sobran = set(requeridas) - valor.keys(), valor.keys() - set(permitidas)
    if faltan or sobran:
        raise ErrorDeEntrada(
            f"{nombre}: faltan {sorted(faltan)}, claves no admitidas {sorted(sobran)}"
        )
    return valor


def _numero(valor, nombre, positivo=False):
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not np.isfinite(valor):
        raise ErrorDeEntrada(f"{nombre} debe ser un numero finito")
    if positivo and valor <= 0:
        raise ErrorDeEntrada(f"{nombre} debe ser > 0")
    return float(valor)


def _material(valor):
    if isinstance(valor, str):
        if valor not in CATALOGO:
            raise ErrorDeEntrada(
                f"material desconocido {valor!r}; usar {list(CATALOGO)} o un modelo Cauchy"
            )
        return CATALOGO[valor]
    d = _objeto(valor, "material", ("tipo", "A", "B", "C", "nombre"), ("tipo", "A"))
    if d["tipo"] != "cauchy":
        raise ErrorDeEntrada("material.tipo debe ser 'cauchy'")
    nombre = d.get("nombre", "Cauchy_propuesto")
    if not isinstance(nombre, str) or not nombre.strip():
        raise ErrorDeEntrada("material.nombre debe ser texto no vacio")
    return Cauchy(
        A=_numero(d["A"], "material.A"),
        B=_numero(d.get("B", 0), "material.B"),
        C=_numero(d.get("C", 0), "material.C"),
        nombre=nombre,
    )


def _capas(valor, nombre, lam0, hud):
    if not isinstance(valor, list) or not valor:
        raise ErrorDeEntrada(f"{nombre} debe ser una lista no vacia de capas")
    capas = []
    for j, entrada in enumerate(valor):
        e = _objeto(entrada, f"{nombre}[{j}]", ("material", "d"), ("material", "d"))
        m = _material(e["material"])
        m.n(lam0)
        m.n_grupo(lam0)
        if hud is not None:
            m.n(hud.lam)
        capas.append(Capa(m, _numero(e["d"], f"{nombre}[{j}].d", positivo=True)))
    return capas


def _ruido(valor, n, lam0):
    r = _objeto(valor, "ruido", ("sigma_a", "covarianza"))
    if ("sigma_a" in r) == ("covarianza" in r):
        raise ErrorDeEntrada("ruido requiere sigma_a O covarianza, exactamente una")
    if "sigma_a" in r:
        s = np.asarray(r["sigma_a"], float)
        if s.ndim == 0:
            s = np.full(n, _numero(r["sigma_a"], "ruido.sigma_a", positivo=True))
        if s.shape != (n,) or not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ErrorDeEntrada("ruido.sigma_a debe ser > 0: escalar o una entrada por capa")
        cov = np.diag(s**2)
        origen = "sigma_a; independencia entre capas SUPUESTA"
    else:
        cov = np.asarray(r["covarianza"], float)
        origen = "covarianza suministrada; trazabilidad experimental no verificada"
    cfg = ConfigGemelo(lam0=lam0, covarianza=cov, sigma_lnA=None)
    return cfg, covarianza_observables(n, cfg), origen


def _sin_no_finitos(x):
    if isinstance(x, np.ndarray):
        return _sin_no_finitos(x.tolist())
    if isinstance(x, dict):
        return {k: _sin_no_finitos(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sin_no_finitos(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def _analizar(capas, cfg, cov_diferencia, hud, w):
    ng = np.array([float(c.material.n_grupo(cfg.lam0)) for c in capas])
    Cd = cov_diferencia / np.outer(ng, ng)
    sigma_d = np.sqrt(np.diag(Cd))
    scfg = ConfigGemelo(lam0=cfg.lam0, covarianza=cov_diferencia, sigma_lnA=None)
    s = sensibilidad(capas, scfg)
    salida = {
        "espesores_um": [c.d for c in capas],
        "materiales": [c.material.nombre for c in capas],
        "indices_grupo_modelo": ng,
        "caminos_opticos_grupo_um": camino_optico_grupo(capas, cfg.lam0),
        "sensibilidad_indices_exactos": {
            "hipotesis": "n_g conocido exactamente; error del material/modelo excluido",
            "parametros": [f"d{j+1}" for j in range(len(capas))],
            "sigma_um": sigma_d,
            "cambio_minimo_local_um": UMBRAL * sigma_d,
            "covarianza_espesor_um2": Cd,
        },
        "sensibilidad_sin_prior_indice": {
            "hipotesis": "un indice desconocido por capa; solo caminos opticos",
            "parametros": s.nombres,
            "rango": s.rango,
            "identificables": s.identificables,
            "sigma": s.sigma,
            "cambio_minimo_local": s.delta_min,
            "unidades_parametros": ["um"] * len(capas) + ["adimensional"] * len(capas),
            "null_significa": "no identificable con estos observables, no precision cero",
        },
        "w_mrad": w,
    }
    if hud is None:
        salida["hud"] = {"estado": "no calculado; falta geometria HUD"}
    elif w is None:
        salida["hud"] = {"estado": "no calculado; falta w_mrad"}
    elif len(capas) != 3:
        salida["hud"] = {
            "estado": "no calculado; trazador HUD requiere tres capas vidrio/PVB/vidrio"
        }
    else:
        try:
            p = predecir_imagen_doble(list(reversed(capas)), w * 1e-3, hud)
            salida["hud"] = {
                "estado": "exploratorio",
                "separacion_firmada_arcmin": p.separacion_arcmin,
                "separacion_magnitud_arcmin": abs(p.separacion_arcmin),
                "sigma_total_arcmin": None,
                "incertidumbre_modelo_incluida": False,
            }
        except ErrorDeEntrada as exc:
            salida["hud"] = {
                "estado": "geometria fuera del dominio del trazador",
                "motivo": str(exc),
                "sigma_total_arcmin": None,
            }
    return salida


def evaluar_config(c):
    c = _objeto(
        c,
        "config",
        ("lam0", "ruido", "receta_base", "alternativas", "hud", "dos_mediciones", "w_base_mrad"),
        ("lam0", "ruido", "receta_base", "alternativas"),
    )
    lam0 = _numero(c["lam0"], "lam0", positivo=True)
    dos = c.get("dos_mediciones", False)
    if not isinstance(dos, bool):
        raise ErrorDeEntrada("dos_mediciones debe ser true o false")
    hud = None
    if c.get("hud") is not None:
        h = _objeto(
            c["hud"], "hud", ("theta_i_deg", "Rp", "Rv", "lam"), ("theta_i_deg", "Rp", "Rv")
        )
        hud = ParametrosHUD(
            theta_i=np.deg2rad(_numero(h["theta_i_deg"], "hud.theta_i_deg")),
            Rp=_numero(h["Rp"], "hud.Rp", True),
            Rv=_numero(h["Rv"], "hud.Rv", True),
            lam=_numero(h.get("lam", 0.55), "hud.lam", True),
        )
    base = _capas(c["receta_base"], "receta_base", lam0, hud)
    cfg, cov, origen = _ruido(c["ruido"], len(base), lam0)
    covdif = cov * (2 if dos else 1)

    def cuna(valor, nombre):
        if valor is None:
            return None
        w = _numero(valor, nombre)
        if abs(w) >= np.pi / 2 * 1e3:
            raise ErrorDeEntrada(f"{nombre} fuera del dominio angular")
        return w

    wb = cuna(c.get("w_base_mrad"), "w_base_mrad")
    if not isinstance(c["alternativas"], list) or not c["alternativas"]:
        raise ErrorDeEntrada("alternativas debe ser una lista no vacia")
    alternativas, nombres = [], set()
    for j, entrada in enumerate(c["alternativas"]):
        e = _objeto(
            entrada, f"alternativas[{j}]", ("nombre", "capas", "w_mrad"), ("nombre", "capas")
        )
        nombre = e["nombre"]
        if not isinstance(nombre, str) or not nombre.strip() or nombre in nombres:
            raise ErrorDeEntrada("cada alternativa requiere un nombre no vacio y unico")
        nombres.add(nombre)
        capas = _capas(e["capas"], nombre, lam0, hud)
        if len(capas) != len(base):
            raise ErrorDeEntrada(
                "las alternativas deben tener el mismo numero de capas que receta_base"
            )
        dist = distinguibilidad(base, capas, cfg, umbral=UMBRAL, dos_mediciones=dos)
        resultado = _analizar(capas, cfg, covdif, hud, cuna(e.get("w_mrad"), f"{nombre}.w_mrad"))
        resultado.update(
            nombre=nombre,
            comparacion_con_base={
                "distancia_mahalanobis": dist.delta,
                "umbral": UMBRAL,
                "distinguible_condicional": dist.distinguibles,
                "contribuciones_blanqueadas": dist.contribuciones,
                "error_clasificacion_ideal": dist.error_clasificacion,
                "hipotesis_error": "dos hipotesis simples gaussianas, igual covarianza, prior 1/2",
            },
        )
        alternativas.append(resultado)
    informe = {
        "version_esquema": 1,
        "naturaleza": "predicciones condicionales; no validacion experimental",
        "unidades": {
            "longitudes_y_Rp_Rv": "um",
            "lam0_y_hud_lam": "um",
            "cuña": "mrad",
            "covarianza_ruido": "um^2",
            "imagen_doble": "arcmin",
        },
        "orden_capas": "exterior -> interior del auto",
        "signo_cuna": "w>0: PVB crece hacia +x, desde fuente HUD hacia ojo; mismo eje al invertir capas",
        "lam0_um": lam0,
        "observables": "solo caminos opticos de grupo por capa; sin amplitudes",
        "ruido": {
            "origen": origen,
            "covarianza_una_medicion_um2": cov,
            "dos_mediciones_independientes": dos,
            "covarianza_comparacion_um2": covdif,
        },
        "base": _analizar(base, cfg, covdif, hud, wb),
        "alternativas": alternativas,
        "comprobaciones_equipo": {
            "resolucion_de_interfaces_verificada": False,
            "rango_del_escaner_verificado": False,
            "banda_y_muestreo_espectral_verificados": False,
        },
        "limites": [
            "Distinguibilidad presupone que todas las interfaces se localizan: faltan banda, rejilla espectral, rango y contraste medidos.",
            "El ruido suministrado debe provenir de repetibilidad real; el programa no certifica su origen.",
            "Indices del catalogo/modelo son supuestos para la receta; no sustituyen calibracion ni validacion con una segunda muestra.",
            "Cambio minimo = 3 sigma es local y gaussiano; no garantiza una potencia de deteccion ni error de clasificacion de 0.3%.",
            "HUD es trazado local 2D, tres capas, solo cuña PVB, sin curvatura, polarizacion, intensidad ni dispersion validada; sigma total desconocida.",
        ],
    }
    return _sin_no_finitos(informe)


def ejecutar_config(ruta_config, salida):
    def rechazar_constante(valor):
        raise ErrorDeEntrada(f"JSON no estandar: {valor}")

    try:
        with open(ruta_config, encoding="utf-8") as f:
            config = json.load(f, parse_constant=rechazar_constante)
        informe = evaluar_config(config)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ErrorDeEntrada):
            raise
        raise ErrorDeEntrada(f"configuracion de recetas invalida: {exc}") from exc
    destino = Path(salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return informe


def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Explorar recetas de vidrio laminado sin fabricar")
    p.add_argument("--config", required=True, help="JSON de materiales, espesores y ruido real")
    p.add_argument("--salida", default="informe_recetas.json", help="informe JSON")
    args = p.parse_args(argv)
    try:
        ejecutar_config(args.config, args.salida)
    except (ErrorDeEntrada, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Informe exploratorio escrito en: {Path(args.salida).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
