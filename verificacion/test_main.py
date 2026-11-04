import subprocess
import sys
from pathlib import Path

import pytest

import main
from motor.analisis.pipeline import imprimir_resumen

RAIZ = Path(__file__).resolve().parent.parent


def test_parser_reconoce_todas_las_ordenes():
    parser = main.construir_parser()
    assert parser.parse_args(["demo", "--n-lado", "7"]).n_lado == 7
    assert parser.parse_args(["analizar", "config.json"]).config == "config.json"
    assert parser.parse_args(["receta", "r.json"]).salida == "salidas/recetas.json"
    assert parser.parse_args(["pdf", "i.json", "m.npz", "o.pdf"]).destino == "o.pdf"
    assert parser.parse_args([]).orden is None
    ejemplo = parser.parse_args(["ejemplo", "carpeta"])
    assert (ejemplo.variante, ejemplo.n_lado, ejemplo.semilla) == ("conforme", 17, 0)
    assert parser.parse_args(["ejemplo", "c", "--variante", "defecto"]).variante == "defecto"


def test_ejemplo_rechaza_variante_desconocida():
    with pytest.raises(SystemExit):
        main.construir_parser().parse_args(["ejemplo", "c", "--variante", "otra"])


def test_orden_desconocida_se_rechaza():
    with pytest.raises(SystemExit):
        main.construir_parser().parse_args(["inexistente"])


def test_resumen_de_consola_es_breve_y_declara_el_estado(capsys):
    imprimir_resumen(
        {
            "veredicto": "no concluyente",
            "espesor": {"pvb": {"media_um": 760.0, "sigma_aleatoria_um": 0.2}},
            "cuna_pvb_mrad": {"media": 0.35},
        }
    )
    salida = capsys.readouterr().out.splitlines()
    assert salida == ["Estado: no concluyente", "pvb: 760.00 ± 0.200 µm", "Cuña PVB: 0.350 mrad"]


def test_resumen_tolera_valores_ausentes(capsys):
    imprimir_resumen({"espesor": {"pvb": {}}})
    assert "n/d" in capsys.readouterr().out


def test_ayuda_de_la_linea_de_comandos_funciona():
    resultado = subprocess.run(
        [sys.executable, str(RAIZ / "main.py"), "--help"], capture_output=True, text=True, cwd=RAIZ
    )
    assert resultado.returncode == 0
    for orden in ("interfaz", "demo", "analizar", "receta", "pdf"):
        assert orden in resultado.stdout
