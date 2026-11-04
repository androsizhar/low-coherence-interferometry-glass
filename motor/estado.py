from __future__ import annotations

import copy
import json

ETIQUETA_HUD = "predicción exploratoria"
ESTADO_PROTOTIPO = (
    "prototipo de investigación endurecido, listo para iniciar validación experimental"
)


def condicion_pose(informe: dict) -> str:
    registro = informe.get("registro") or {}
    modo = registro.get("modo")
    if modo == "icp_forma":
        return "mejor ajuste: forma relativa, no absoluta"
    origen = (informe.get("estado_validacion") or {}).get("origen_datos", "no_declarado")
    cov = registro.get("covarianza")
    if hasattr(cov, "tolist"):
        cov = cov.tolist()
    if modo != "fijo":
        return "pose no documentada: desviación absoluta no caracterizada"
    if cov is None:
        return "pose declarada; covarianza no disponible: incertidumbre absoluta incompleta"
    correcta = (
        isinstance(cov, (list, tuple))
        and len(cov) == 6
        and all(isinstance(f, (list, tuple)) and len(f) == 6 for f in cov)
    )
    cero = correcta and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0 for f in cov for v in f
    )
    if origen == "simulado" and cero:
        return "pose sintética exacta; covarianza 0: escenario ideal, no precisión esperada del montaje real"
    descripcion = (
        "sintética"
        if origen == "simulado"
        else (
            "declarada (real)" if origen == "real_declarado" else "declarada (origen no verificado)"
        )
    )
    return f'pose {descripcion} con covarianza X = {json.dumps(cov, ensure_ascii=False, separators=(",", ":"))} [rotación rad, traslación µm], marco CAD'


def motivos_no_concluyente(informe: dict) -> list[str]:
    cal = informe.get("calibracion") or {}
    estado = informe.get("estado_validacion") or {}
    cruzada = informe.get("validacion_cruzada") or {}
    razones = []
    if cal.get("ok") is not True or cal.get("banderas"):
        razones.append("calibración no compatible o con banderas")
    if (
        estado.get("segunda_muestra") != "compatible"
        or cruzada.get("ok") is not True
        or estado.get("independencia_documentada") is not True
    ):
        razones.append(
            "falta validación cruzada compatible con una segunda muestra independiente documentada"
        )
    if estado.get("validacion_experimental_completa") is not True:
        razones.append("validación experimental incompleta")
    if estado.get("origen_datos") != "real_declarado":
        razones.append("origen real de datos no declarado")
    return razones


def veredicto(informe: dict) -> str:
    if motivos_no_concluyente(informe):
        return "no concluyente"
    return "resultado de investigación; revisar criterio de aceptación acordado"


def aplicar_estado(informe: dict) -> dict:
    out = copy.deepcopy(informe)
    out["veredicto"] = veredicto(out)
    out["motivos_no_concluyente"] = motivos_no_concluyente(out)
    out["condicion_CAD"] = condicion_pose(out)
    out.setdefault("registro", {})["condicion"] = out["condicion_CAD"]
    for fila in (out.get("desviacion_vs_cad_um") or {}).values():
        if isinstance(fila, dict):
            fila["condicion"] = out["condicion_CAD"]
    gemelo = out.setdefault("gemelo", {})
    gemelo["etiqueta_hud"] = ETIQUETA_HUD
    gemelo["estado_hud"] = ETIQUETA_HUD
    out["etiqueta_HUD"] = ETIQUETA_HUD
    out["estado_prototipo"] = ESTADO_PROTOTIPO
    return out
