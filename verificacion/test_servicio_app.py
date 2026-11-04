import json
from pathlib import Path

import pytest

from motor.servicio import (
    ErrorProyecto,
    OperacionCancelada,
    _publicar_resultado,
    abrir_proyecto,
    cargar_resultado,
    generar_ejemplo,
    guardar_proyecto,
    validar_proyecto,
)


def test_proyecto_restaurable_con_rutas_semilla_y_tema(tmp_path):
    estado = {
        "config_path": str(tmp_path / "config.json"),
        "semilla": 31,
        "tema_oscuro": True,
        "receta": {"a": 1},
    }
    ruta = guardar_proyecto(tmp_path, estado)
    assert abrir_proyecto(tmp_path) == estado
    assert abrir_proyecto(ruta) == estado


def test_falta_clave_mensaje_espanol(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{}")
    with pytest.raises(ErrorProyecto, match="Falta la clave mediciones"):
        validar_proyecto(p)


def test_importar_resultado_antiguo_no_aprueba_ni_pierde_condicion(tmp_path):
    p = tmp_path / "informe.json"
    p.write_text(json.dumps({"registro": {"modo": "icp_forma"}}))
    r = cargar_resultado(p)
    assert r["informe"]["veredicto"] == "no concluyente"
    assert r["informe"]["condicion_CAD"] == "mejor ajuste: forma relativa, no absoluta"


def test_cancelar_proceso_no_toca_archivo_previo(tmp_path):
    (tmp_path / "informe.json").write_text("anterior")
    with pytest.raises(OperacionCancelada):
        generar_ejemplo(tmp_path, cancelar=lambda: True)
    assert (tmp_path / "informe.json").read_text() == "anterior"


def test_publicar_rechaza_salida_ajena(tmp_path):
    origen = tmp_path / "stage"
    origen.mkdir()
    (origen / "informe.json").write_text("{}")
    destino = tmp_path / "datos"
    destino.mkdir()
    (destino / "crudos.npz").write_bytes(b"original")
    with pytest.raises(ErrorProyecto, match="archivos ajenos"):
        _publicar_resultado(origen, destino, tmp_path / "config.json")
    assert (destino / "crudos.npz").read_bytes() == b"original"


def test_publicar_sustituye_solo_corrida_completa_y_relocaliza_cad(tmp_path):
    origen = tmp_path / "stage"
    origen.mkdir()
    (origen / "cad").mkdir()
    (origen / "cad/a.stl").write_bytes(b"nuevo")
    (origen / "informe.json").write_text(
        json.dumps({"cad": {"archivos": {"a": "/temporal/a.stl"}}})
    )
    destino = tmp_path / "salida"
    destino.mkdir()
    (destino / "informe.json").write_text("{}")
    _publicar_resultado(origen, destino, tmp_path / "config.json")
    inf = json.loads((destino / "informe.json").read_text())
    assert Path(inf["cad"]["archivos"]["a"]).read_bytes() == b"nuevo"
    assert inf["trazabilidad"]["config"] == str(tmp_path / "config.json")
    assert not list(tmp_path.glob("salida.anterior-*"))
