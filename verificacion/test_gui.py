import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import PySide6
except ImportError:
    pytest.importorskip("PyQt6", reason="GUI opcional: no hay binding Qt local")

from motor.estado import aplicar_estado
from motor.interfaz.componentes import Trabajador
from motor.interfaz.qt import QtWidgets as W
from motor.interfaz.ventana import VentanaPrincipal


@pytest.fixture(scope="module")
def app():
    return W.QApplication.instance() or W.QApplication([])


@pytest.fixture
def ventana(app):
    ventana = VentanaPrincipal()
    ventana.show()
    app.processEvents()
    yield ventana
    if ventana.trabajador and ventana.trabajador.isRunning():
        ventana.cancelar()
        ventana.trabajador.wait(5000)
    ventana.close()
    app.processEvents()


def esperar(app, condicion, segundos=5):
    limite = time.monotonic() + segundos
    while not condicion() and time.monotonic() < limite:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert condicion(), "El flujo de la GUI no terminó dentro del límite"


def test_seis_pestanas_minimo_temas_y_atajos(ventana, app):
    assert [ventana.tabs.tabText(i) for i in range(6)] == [
        "Proyecto",
        "Calibración",
        "Resultados",
        "Calidad",
        "Gemelo",
        "Exportar",
    ]
    assert ventana.minimumWidth() == 1280
    assert ventana.minimumHeight() == 800
    assert len(ventana.actions()) >= 10
    ventana.aplicar_tema(True)
    assert ventana._tema_oscuro
    assert "#edf3fb" in ventana.styleSheet()
    ventana.aplicar_tema(False)
    assert not ventana._tema_oscuro


def test_resultado_sintetico_nunca_aprueba_y_pose_visible(ventana, tmp_path):
    informe = aplicar_estado(
        {
            "estado_validacion": {
                "origen_datos": "simulado",
                "validacion_experimental_completa": False,
            },
            "registro": {"modo": "fijo", "covarianza": [[0] * 6 for _ in range(6)]},
            "calibracion": {"ng": [1.5, 1.48, 1.5], "ok": True, "banderas": []},
        }
    )
    ventana.mostrar_resultado(
        {
            "informe": informe,
            "informe_path": str(tmp_path / "informe.json"),
            "carpeta": str(tmp_path),
        }
    )
    assert "no concluyente" in ventana.veredicto.text()
    assert "pose sintética exacta" in ventana.veredicto.toolTip()
    assert "predicción exploratoria" in ventana.veredicto.toolTip()
    assert "pose sintética exacta" in ventana.resumen_resultados.toPlainText()
    assert ventana.tabla_a.rowCount() == 3
    assert ventana.tabla_a.item(0, 2).text() == "no disponible"


def test_proyecto_se_guarda_y_restaura_sin_cambiar_receta(ventana, tmp_path):
    ventana.restaurar(tmp_path)
    ventana.receta.setPlainText('{"ejemplo": "local"}')
    ventana.semilla.setValue(42)
    ventana.aplicar_tema(True)
    ventana.guardar()
    assert (tmp_path / "proyecto.vitro.json").is_file()
    ventana.receta.clear()
    ventana.restaurar(tmp_path)
    assert json.loads(ventana.receta.toPlainText()) == {"ejemplo": "local"}
    assert ventana.semilla.value() == 42
    assert ventana._tema_oscuro
    assert ventana._rutas["pose"].text() == ""
    assert ventana._rutas["mascara"].text() == ""


def test_errores_de_entrada_visibles_en_espanol(ventana, tmp_path):
    assert not ventana.validar()
    assert "Falta config.json" in ventana.validacion_texto.toPlainText()
    ruta = tmp_path / "mal.json"
    ruta.write_text("{}")
    ventana._rutas["config"].setText(str(ruta))
    assert not ventana.validar()
    assert ventana.validacion_texto.toPlainText()


def test_trabajador_progreso_cancelacion_y_ventana_responde(ventana, app):
    observado = []

    def trabajo(progreso=None, cancelar=None):
        for i in range(500):
            if cancelar():
                return {"cancelado": True}
            progreso(i / 500, "Calculando")
            time.sleep(0.002)
        return {"cancelado": False}

    ventana._tarea(trabajo, observado.append)
    esperar(app, lambda: ventana.barra.value() > 0)
    ventana.tabs.setCurrentIndex(4)
    assert ventana.tabs.currentIndex() == 4
    ventana.cancelar()
    esperar(app, lambda: bool(observado))
    assert observado == [{"cancelado": True}]
    esperar(app, lambda: not ventana.cancelar_btn.isEnabled())


def test_trabajador_error_con_detalle_sin_bloquear(app):
    errores = []

    def trabajo(progreso=None, cancelar=None):
        raise ValueError("Falta la clave instrumento")

    t = Trabajador(trabajo)
    t.error.connect(lambda texto, detalle: errores.append((texto, detalle)))
    t.start()
    esperar(app, lambda: bool(errores))
    t.wait()
    assert errores[0][0] == "Falta la clave instrumento"
    assert "ValueError" in errores[0][1]


def test_config_efectiva_conserva_pose_y_gradiente(ventana, tmp_path):
    cfg = {
        "mediciones": "m.npz",
        "cad": "cad.stl",
        "registro": {"modo": "fijo"},
        "capas": {"gradiente_nominal": [[0, 0], [0.35, 0], [0, 0]]},
        "calibracion": {"archivo": "a.npz"},
    }
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(cfg))
    ventana.proyecto = tmp_path
    ventana._rutas["config"].setText(str(ruta))
    ventana._cargar_config(ruta)
    destino = ventana._preparar_config()
    efectivo = json.loads(destino.read_text())
    assert efectivo["registro"] == cfg["registro"]
    assert efectivo["capas"]["gradiente_nominal"] == cfg["capas"]["gradiente_nominal"]
    assert Path(efectivo["mediciones"]).is_absolute()


def test_mascara_npz_y_rectangulo_preservan_poligonos_y_unidades(ventana, tmp_path):
    mascara = tmp_path / "soporte.npz"
    mascara.touch()
    existente = [[0, 0], [1, 0], [1, 1], [0, 1]]
    cfg = {
        "calibracion": {},
        "capas": {},
        "mascara_adquisicion": {
            "archivo": "soporte.npz",
            "excluir": [existente],
            "incluir": [existente],
        },
    }
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(cfg))
    ventana._rutas["config"].setText(str(ruta))
    ventana._cargar_config(ruta)
    assert ventana._rutas["mascara"].text() == str(mascara)
    for control, valor in zip(ventana.limites_hueco, [-2, 3, -4, 5]):
        control.setValue(valor)
    ventana.agregar_hueco()
    efectivo = ventana._config_efectiva()["mascara_adquisicion"]
    assert efectivo["archivo"] == str(mascara)
    assert efectivo["incluir"] == [existente]
    assert efectivo["excluir"][0] == existente
    assert efectivo["excluir"][1] == [[-2000, -4000], [3000, -4000], [3000, 5000], [-2000, 5000]]
    ventana.proyecto = tmp_path
    ventana.guardar()
    ventana._huecos = []
    ventana.restaurar(tmp_path)
    assert len(ventana._huecos) == 1


def test_valida_rutas_seleccionadas_aunque_original_no_exista(ventana, tmp_path):
    for nombre in ("medicion_nueva.npz", "a.npz", "cad.stl"):
        (tmp_path / nombre).touch()
    cfg = {
        "mediciones": "archivo_anterior_inexistente.npz",
        "cad": "cad.stl",
        "calibracion": {"archivo": "a.npz"},
        "instrumento": {},
        "capas": {},
        "cuna": {},
        "rango_frontal": [1, 2],
        "registro": {"modo": "fijo"},
    }
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(cfg))
    ventana._rutas["config"].setText(str(ruta))
    ventana._cargar_config(ruta)
    assert not ventana.validar()
    ventana._rutas["mediciones"].setText(str(tmp_path / "medicion_nueva.npz"))
    assert ventana.validar()
    assert not (
        tmp_path / "config_interfaz.json"
    ).exists(), "La revisión inmediata no debe escribir archivos del proyecto"


def test_gemelo_usa_covarianza_medida_sin_inventar_independencia(ventana, monkeypatch):
    cov = [[0.1, -0.02, 0], [-0.02, 0.2, -0.03], [0, -0.03, 0.1]]
    ventana.resultado_actual = {"informe": {"calibracion": {"cov_a_um2": cov}}}
    capturado = []
    monkeypatch.setattr(
        ventana, "_tarea", lambda funcion, callback, config: capturado.append(config)
    )
    ventana.evaluar_gemelo()
    assert capturado[0]["ruido"] == {"covarianza": cov}
    assert capturado[0]["alternativas"][0]["capas"][1]["d"] == 761


def _informe_con_validacion(**extra):
    base = {
        "registro": {"modo": "fijo", "covarianza": [[0.0] * 6 for _ in range(6)]},
        "estado_validacion": {
            "origen_datos": "simulado",
            "validacion_experimental_completa": False,
            "segunda_muestra": "compatible",
            "independencia_documentada": False,
        },
        "calibracion": {
            "ok": True,
            "banderas": [],
            "ng": [1.52, 1.48, 1.52],
            "sigma_a_um": [0.1, 0.1, 0.1],
            "u_rel_sistematica": [1e-4, 1e-4, 1e-4],
            "cov_ng": [
                [3.3e-7, -7e-10, 3.3e-7],
                [-7e-10, 3.8e-6, -7e-10],
                [3.3e-7, -7e-10, 3.3e-7],
            ],
        },
        "validacion_cruzada": {
            "ok": False,
            "z": [0.1, 4.2, -0.3],
            "error_um": [0.1, 5.0, -0.3],
            "sigma_combinada_um": [1.0, 1.2, 1.0],
        },
        "espesor": {},
        "banderas": [],
    }
    base.update(extra)
    return aplicar_estado(base)


def test_calibracion_muestra_compatibilidad_si_no_y_matriz_legible(ventana, app):
    informe = _informe_con_validacion()
    ventana.mostrar_resultado(
        {"informe": informe, "informe_path": "/tmp/x/informe.json", "carpeta": "/tmp/x"}
    )
    app.processEvents()
    compat = [ventana.tabla_b.item(j, 4).text() for j in range(3)]
    assert compat == ["sí", "no", "sí"]
    texto = ventana.covarianza.toPlainText()
    assert "e-07" in texto and "[" not in texto
    assert "no concluyente" in ventana.veredicto.text()
    assert "aprobado" not in ventana.veredicto.text().lower()


def test_estado_indica_motivos_pendientes_en_la_ayuda(ventana, app):
    informe = _informe_con_validacion()
    ventana.mostrar_resultado(
        {"informe": informe, "informe_path": "/tmp/x/informe.json", "carpeta": "/tmp/x"}
    )
    assert "validación experimental incompleta" in ventana.veredicto.toolTip()


def test_gemelo_bloquea_coeficientes_en_catalogo_y_los_habilita_en_cauchy(ventana, app):
    from motor.interfaz.qt import QtCore as C

    tabla = ventana.receta_tabla
    for columna in (1, 2, 3):
        item = tabla.item(0, columna)
        assert item.text() == "(catálogo)" and not item.flags() & C.Qt.ItemFlag.ItemIsEditable
    tabla.cellWidget(0, 0).setCurrentText("Cauchy")
    app.processEvents()
    assert [tabla.item(0, c).text() for c in (1, 2, 3)] == ["1.5", "0", "0"]
    assert all(tabla.item(0, c).flags() & C.Qt.ItemFlag.ItemIsEditable for c in (1, 2, 3))
    tabla.cellWidget(0, 0).setCurrentText("BK7")
    assert tabla.item(0, 1).text() == "(catálogo)"


def test_cada_tipo_de_vista_usa_su_propio_archivo(ventana, tmp_path, monkeypatch):
    informe = _informe_con_validacion()
    ventana.mostrar_resultado(
        {
            "informe": informe,
            "informe_path": str(tmp_path / "informe.json"),
            "carpeta": str(tmp_path),
        }
    )
    destinos = {}
    monkeypatch.setattr(
        ventana,
        "_tarea",
        lambda funcion, callback, *a, **k: destinos.setdefault(
            ventana.tipo_vista.currentIndex(), k["destino"]
        ),
    )
    for indice in range(3):
        ventana.tipo_vista.setCurrentIndex(indice)
        ventana.actualizar_vista()
    assert len(set(destinos.values())) == 3
    assert [Path(destinos[i]).name for i in range(3)] == [
        "vista_interfaz_mapa.png",
        "vista_interfaz_3d.png",
        "vista_interfaz_corte.png",
    ]


def test_exportacion_se_muestra_como_lista_de_archivos(ventana):
    ventana._exportado({"carpeta": "/x", "archivos": ["a.stl", "b.pdf"], "manifest": "/x/m.json"})
    texto = ventana.exportacion_texto.toPlainText().splitlines()
    assert texto[0] == "Carpeta: /x"
    assert "  a.stl" in texto and "  b.pdf" in texto
    assert texto[-1] == "Manifiesto: /x/m.json"


def test_la_rueda_del_raton_no_cambia_los_selectores(ventana):
    from motor.interfaz.qt import QtCore, QtGui
    from motor.interfaz.qt import QtWidgets as W

    combo = ventana.receta_tabla.cellWidget(1, 0)
    antes = combo.currentText()
    evento = QtGui.QWheelEvent(
        QtCore.QPointF(5, 5),
        QtCore.QPointF(5, 5),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, -120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    W.QApplication.sendEvent(combo, evento)
    assert combo.currentText() == antes
