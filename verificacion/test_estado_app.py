import copy

import pytest

from motor.estado import aplicar_estado, condicion_pose, veredicto


def _inf():
    return {
        "registro": {"modo": "fijo", "covarianza": [[0.0] * 6 for _ in range(6)]},
        "calibracion": {"ok": True, "banderas": []},
        "estado_validacion": {
            "origen_datos": "simulado",
            "segunda_muestra": "compatible",
            "independencia_documentada": True,
            "validacion_experimental_completa": False,
        },
        "validacion_cruzada": {"ok": True},
        "desviacion_vs_cad_um": {"interfaz_0": {"rms": 0.35}},
    }


def test_pose_exacta_etiqueta_junto_a_cifra():
    i = aplicar_estado(_inf())
    assert "pose sintética exacta" in i["condicion_CAD"]
    assert "no precisión esperada" in i["desviacion_vs_cad_um"]["interfaz_0"]["condicion"]
    assert i["etiqueta_HUD"] == "predicción exploratoria"
    assert i["veredicto"] == "no concluyente"


def test_pose_real_con_covarianza_visible():
    i = _inf()
    i["estado_validacion"]["origen_datos"] = "real_declarado"
    i["registro"]["covarianza"][5][5] = 9.0
    label = condicion_pose(i)
    assert "pose declarada (real)" in label and "9.0" in label and "µm" in label


def test_pose_icp_y_desconocida():
    i = _inf()
    i["registro"]["modo"] = "icp_forma"
    assert condicion_pose(i) == "mejor ajuste: forma relativa, no absoluta"
    i["registro"]["modo"] = "fijo"
    i["registro"]["covarianza"] = None
    assert "no disponible" in condicion_pose(i)


@pytest.mark.parametrize("campo", ["cal", "bandera", "muestra", "experimento", "independencia"])
def test_puertas_impiden_aprobacion(campo):
    i = _inf()
    i["estado_validacion"].update(
        origen_datos="real_declarado", validacion_experimental_completa=True
    )
    if campo == "cal":
        i["calibracion"]["ok"] = False
    if campo == "bandera":
        i["calibracion"]["banderas"] = ["inconsistencia"]
    if campo == "muestra":
        i.pop("validacion_cruzada")
    if campo == "experimento":
        i["estado_validacion"]["validacion_experimental_completa"] = False
    if campo == "independencia":
        i["estado_validacion"]["independencia_documentada"] = False
    assert veredicto(i) == "no concluyente"


def test_aplicar_no_muta_fuente():
    i = _inf()
    antes = copy.deepcopy(i)
    aplicar_estado(i)
    assert i == antes
