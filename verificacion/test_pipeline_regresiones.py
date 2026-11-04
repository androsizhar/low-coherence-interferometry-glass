import json
from dataclasses import replace

import numpy as np
import pytest

from motor.analisis.pipeline import (
    _escribir_informe,
    _estado_validacion,
    _resumen_validacion,
    _serializable,
    configuracion_demo,
    ejecutar,
    ejecutar_desde_config,
)
from motor.medicion.lectura import ErrorDeEntrada
from motor.metrologia.calibracion import Calibracion, ValidacionCruzada


@pytest.mark.parametrize(
    "cambio",
    [
        {"tol_d": [10, np.nan, 10]},
        {"d_nom": [2100, -760, 1600]},
        {"grilla": (1, 20)},
        {"idx_pvb": 3},
        {"sigma_pos": np.inf},
        {"direccion_cuna": (0, 0)},
        {"nombres_capas": ["a", "../b", "c"]},
        {"cov_transformacion": -np.eye(6)},
        {"modo_registro": "inventado"},
        {"gradiente_nominal": [[0, 0]]},
        {"rango_frontal": (100, 1)},
    ],
)
def test_config_rechaza_parametros_no_fisicos(cambio):
    cfg, _ = configuracion_demo()
    with pytest.raises(ErrorDeEntrada):
        replace(cfg, **cambio)


def test_calibracion_inconsistente_detiene_antes_de_reconstruir():
    cfg, D = configuracion_demo()
    cal = Calibracion(
        np.array([1.5, 1.48, 1.5]),
        np.full(3, 0.001),
        np.full(3, 0.2),
        cfg.grupos,
        banderas=["inconsistencia_grupo:vidrio"],
    )
    with pytest.raises(ErrorDeEntrada, match="calibracion inconsistente"):
        ejecutar([], cfg, cal, D.malla_cad(n=5))


def test_informe_persistido_conserva_segunda_muestra_y_json_estricto(tmp_path):
    cfg, _ = configuracion_demo()
    val = ValidacionCruzada(
        np.array([0.0, 0.1, np.inf]),
        False,
        np.array([2100.0, 760.0, 1601.0]),
        np.ones(3),
        np.array([0.0, 0.0, 1.0]),
    )
    informe = {
        "estado_validacion": _estado_validacion(cfg, val, True),
        "validacion_cruzada": _resumen_validacion(val),
    }
    esperado = _escribir_informe(informe, tmp_path)
    texto = (tmp_path / "informe.json").read_text()
    assert "Infinity" not in texto and "NaN" not in texto
    escrito = json.loads(texto)
    assert escrito == esperado
    assert escrito["validacion_cruzada"]["ok"] is False
    assert escrito["estado_validacion"]["validacion_experimental_completa"] is False
    assert escrito["estado_validacion"]["decision_de_liberacion"] == "no_emitida"
    assert escrito["validacion_cruzada"]["z"][-1] is None


def test_config_material_desconocido_da_error_util(tmp_path):
    config = {
        "mediciones": "m.npz",
        "calibracion": {"archivo": "c.npz"},
        "cad": "cad.stl",
        "instrumento": {"lam0": 1.55, "offset": 10000},
        "capas": {},
        "cuna": {},
        "rango_frontal": [19000, 21000],
        "materiales_apriori": ["inexistente"],
    }
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(config))
    with pytest.raises(ErrorDeEntrada, match="materiales_apriori"):
        ejecutar_desde_config(str(ruta), verbose=False)


def test_config_requiere_marco_de_comparacion_explicito(tmp_path):
    config = {
        "mediciones": "m.npz",
        "calibracion": {"archivo": "c.npz"},
        "cad": "cad.stl",
        "instrumento": {"lam0": 1.55, "offset": 10000},
        "capas": {},
        "cuna": {},
        "rango_frontal": [19000, 21000],
    }
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(config))
    with pytest.raises(ErrorDeEntrada, match="registro.modo"):
        ejecutar_desde_config(str(ruta), verbose=False)


def test_json_preserva_booleanos_y_nulos():
    assert _serializable({"x": True, "y": np.bool_(False), "z": float("nan")}) == {
        "x": True,
        "y": False,
        "z": None,
    }
