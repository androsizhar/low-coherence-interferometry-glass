import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from motor.analisis.gemelo import ParametrosHUD, imagen_doble_receta
from motor.analisis.recetas import evaluar_config, main
from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.materiales import BK7, PVB_SUPUESTO
from motor.medicion.modelo_directo import Capa


def _config():
    capas = [
        {"material": "BK7", "d": 2100.0},
        {"material": "PVB_SUPUESTO", "d": 760.0},
        {"material": "BK7", "d": 1600.0},
    ]
    alternativa = copy.deepcopy(capas)
    alternativa[1]["d"] += 1
    return {
        "lam0": 1.55,
        "ruido": {"sigma_a": 0.3},
        "receta_base": capas,
        "alternativas": [{"nombre": "un um PVB", "capas": alternativa, "w_mrad": 0.4}],
        "hud": {"theta_i_deg": 62.0, "Rp": 600000.0, "Rv": 900000.0, "lam": 0.55},
    }


def test_receta_con_indices_conocidos_y_sin_prior_son_distintas_hipotesis():
    r = evaluar_config(_config())
    a = r["alternativas"][0]
    assert a["comparacion_con_base"]["distinguible_condicional"]
    assert a["sensibilidad_indices_exactos"]["cambio_minimo_local_um"][1] < 1
    ambiguo = a["sensibilidad_sin_prior_indice"]
    assert ambiguo["rango"] == 3
    assert not any(ambiguo["identificables"])
    assert ambiguo["sigma"] == [None] * 6
    assert not any(r["comprobaciones_equipo"].values())
    assert a["hud"]["sigma_total_arcmin"] is None


def test_covarianza_completa_y_comparacion_dos_mediciones():
    c = _config()
    c["ruido"] = {"covarianza": [[0.09, -0.03, 0], [-0.03, 0.09, -0.03], [0, -0.03, 0.09]]}
    r1 = evaluar_config(c)
    c["dos_mediciones"] = True
    r2 = evaluar_config(c)
    a1, a2 = r1["alternativas"][0], r2["alternativas"][0]
    d1 = a1["comparacion_con_base"]["distancia_mahalanobis"]
    d2 = a2["comparacion_con_base"]["distancia_mahalanobis"]
    assert d1 / d2 == pytest.approx(np.sqrt(2))
    s1 = a1["sensibilidad_indices_exactos"]["sigma_um"]
    s2 = a2["sensibilidad_indices_exactos"]["sigma_um"]
    assert np.array(s2) == pytest.approx(np.sqrt(2) * np.array(s1))
    assert a1["sensibilidad_indices_exactos"]["covarianza_espesor_um2"][0][1] < 0


def test_receta_identica_no_distinguible():
    c = _config()
    c["alternativas"][0]["capas"] = copy.deepcopy(c["receta_base"])
    a = evaluar_config(c)["alternativas"][0]["comparacion_con_base"]
    assert a["distancia_mahalanobis"] == 0
    assert not a["distinguible_condicional"] and a["error_clasificacion_ideal"] == 0.5


def test_hud_usa_orden_interior_exterior_con_vidrios_asimetricos():
    a = evaluar_config(_config())["alternativas"][0]
    hud = ParametrosHUD(np.deg2rad(62), 600000, 900000, 0.55)
    interior = [Capa(BK7, 1600), Capa(PVB_SUPUESTO, 761), Capa(BK7, 2100)]
    esperado = imagen_doble_receta(interior, 0.4e-3, hud)
    assert a["hud"]["separacion_firmada_arcmin"] == pytest.approx(esperado)


def test_material_personalizado_y_ausencia_de_hud():
    c = _config()
    del c["hud"]
    c["alternativas"][0]["capas"][1]["material"] = {
        "tipo": "cauchy",
        "A": 1.48,
        "B": 0.005,
        "nombre": "nuevo",
    }
    a = evaluar_config(c)["alternativas"][0]
    assert a["materiales"][1] == "nuevo"
    assert a["hud"]["estado"].startswith("no calculado")


def test_no_se_inventa_cuna_cuando_no_se_declara():
    c = _config()
    del c["alternativas"][0]["w_mrad"]
    a = evaluar_config(c)["alternativas"][0]
    assert a["w_mrad"] is None and a["hud"]["estado"] == "no calculado; falta w_mrad"


@pytest.mark.parametrize(
    "mutacion",
    [
        "material",
        "numero_capas",
        "ruido_duplicado",
        "ruido_singular",
        "nan",
        "espesor",
        "typo",
        "dos_no_booleano",
        "nombre_repetido",
    ],
)
def test_configuracion_invalida_falla_explicita(mutacion):
    c = _config()
    if mutacion == "material":
        c["receta_base"][0]["material"] = "sin_experimento"
    elif mutacion == "numero_capas":
        c["alternativas"][0]["capas"].pop()
    elif mutacion == "ruido_duplicado":
        c["ruido"]["covarianza"] = np.eye(3).tolist()
    elif mutacion == "ruido_singular":
        c["ruido"] = {"covarianza": np.zeros((3, 3)).tolist()}
    elif mutacion == "nan":
        c["lam0"] = float("nan")
    elif mutacion == "espesor":
        c["receta_base"][0]["d"] = -1
    elif mutacion == "typo":
        c["lamda0"] = c.pop("lam0")
    elif mutacion == "dos_no_booleano":
        c["dos_mediciones"] = 1
    else:
        c["alternativas"].append(copy.deepcopy(c["alternativas"][0]))
    with pytest.raises(ErrorDeEntrada):
        evaluar_config(c)


def test_cli_escribe_json_estricto_y_ejemplo_funciona(tmp_path):
    cfg = tmp_path / "recetas.json"
    cfg.write_text(json.dumps(_config()))
    salida = tmp_path / "salida" / "informe.json"
    assert main(["--config", str(cfg), "--salida", str(salida)]) == 0
    texto = salida.read_text()
    assert "Infinity" not in texto and "NaN" not in texto
    assert json.loads(texto)["base"]["sensibilidad_sin_prior_indice"]["sigma"] == [None] * 6
    raiz = Path(__file__).resolve().parents[1]
    proceso = subprocess.run(
        [
            sys.executable,
            "-m",
            "motor.analisis.recetas",
            "--config",
            str(raiz / "ejemplos" / "recetas.json"),
            "--salida",
            str(tmp_path / "ejemplo.json"),
        ],
        cwd=raiz,
        capture_output=True,
        text=True,
    )
    assert proceso.returncode == 0, proceso.stderr
    assert len(json.loads((tmp_path / "ejemplo.json").read_text())["alternativas"]) == 2


def test_json_nan_de_entrada_y_error_cli_no_crean_informe(tmp_path):
    cfg = tmp_path / "mal.json"
    cfg.write_text('{"lam0": NaN}')
    salida = tmp_path / "informe.json"
    assert main(["--config", str(cfg), "--salida", str(salida)]) == 2
    assert not salida.exists()
