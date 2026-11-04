import sys
import traceback
from pathlib import Path

from motor.estado import ETIQUETA_HUD, condicion_pose, veredicto
from motor.interfaz.componentes import Trabajador
from motor.interfaz.pestana_calibracion import PestanaCalibracion
from motor.interfaz.pestana_calidad import PestanaCalidad
from motor.interfaz.pestana_exportar import PestanaExportar
from motor.interfaz.pestana_gemelo import PestanaGemelo
from motor.interfaz.pestana_proyecto import PestanaProyecto
from motor.interfaz.pestana_resultados import PestanaResultados
from motor.interfaz.qt import QtGui as G
from motor.interfaz.qt import QtWidgets as W
from motor.interfaz.tema import hoja_de_estilo

PESTAÑAS_BLOQUEADAS = (0, 3, 4, 5)


class VentanaPrincipal(
    PestanaProyecto,
    PestanaCalibracion,
    PestanaResultados,
    PestanaCalidad,
    PestanaGemelo,
    PestanaExportar,
    W.QMainWindow,
):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vitro · Gemelo digital")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 920)
        self.proyecto = None
        self.resultado_actual = None
        self.trabajador = None
        self._callback = None
        self._tema_oscuro = False
        self._imagenes_vista = {}
        self._huecos = []
        self._crear_ui()
        self._crear_atajos()
        self.aplicar_tema(False)

    def _crear_ui(self):
        centro = W.QWidget()
        raiz = W.QVBoxLayout(centro)
        raiz.setContentsMargins(20, 14, 20, 12)

        cabecera = W.QHBoxLayout()
        self.veredicto = W.QLabel("Estado: sin análisis")
        self.veredicto.setObjectName("estado_resultado")
        self.tema = W.QPushButton("Modo oscuro")
        self.tema.clicked.connect(lambda: self.aplicar_tema(not self._tema_oscuro))
        cabecera.addStretch()
        cabecera.addWidget(self.veredicto)
        cabecera.addWidget(self.tema)
        raiz.addLayout(cabecera)

        self.tabs = W.QTabWidget()
        self.tabs.setDocumentMode(True)
        pestañas = (
            ("Proyecto", self._crear_proyecto),
            ("Calibración", self._crear_calibracion),
            ("Resultados", self._crear_resultados),
            ("Calidad", self._crear_calidad),
            ("Gemelo", self._crear_gemelo),
            ("Exportar", self._crear_exportar),
        )
        for nombre, construir in pestañas:
            contenido = W.QWidget()
            construir(W.QVBoxLayout(contenido))
            if nombre == "Proyecto":
                desplazable = W.QScrollArea()
                desplazable.setWidgetResizable(True)
                desplazable.setFrameShape(W.QFrame.Shape.NoFrame)
                desplazable.setWidget(contenido)
                contenido = desplazable
            self.tabs.addTab(contenido, nombre)
        raiz.addWidget(self.tabs, 1)

        pie = W.QHBoxLayout()
        self.estado = W.QLabel("Abra una carpeta de proyecto o genere un ejemplo")
        self.barra = W.QProgressBar()
        self.barra.setRange(0, 100)
        self.barra.setMaximumWidth(300)
        self.cancelar_btn = W.QPushButton("Cancelar")
        self.cancelar_btn.setEnabled(False)
        self.cancelar_btn.clicked.connect(self.cancelar)
        pie.addWidget(self.estado, 1)
        pie.addWidget(self.barra)
        pie.addWidget(self.cancelar_btn)
        raiz.addLayout(pie)
        self.setCentralWidget(centro)

    def _crear_atajos(self):
        atajos = (
            ("Ctrl+O", self.abrir_carpeta),
            ("Ctrl+S", self.guardar),
            ("Ctrl+R", self.analizar),
            ("Escape", self.cancelar),
        )
        for secuencia, accion in atajos:
            self._atajo(secuencia, accion)
        for indice in range(self.tabs.count()):
            self._atajo(f"Alt+{indice + 1}", lambda _=None, n=indice: self.tabs.setCurrentIndex(n))

    def _atajo(self, secuencia, accion):
        atajo = G.QAction(self)
        atajo.setShortcut(G.QKeySequence(secuencia))
        atajo.triggered.connect(accion)
        self.addAction(atajo)

    def aplicar_tema(self, oscuro):
        self._tema_oscuro = bool(oscuro)
        self.setStyleSheet(hoja_de_estilo(self._tema_oscuro))
        self.tema.setText("Modo claro" if oscuro else "Modo oscuro")

    def _servicio(self):
        from motor import servicio

        return servicio

    def _error(self, mensaje, detalle=""):
        self.estado.setText("Revise los datos")
        self.validacion_texto.setPlainText(mensaje)
        caja = W.QMessageBox(self)
        caja.setIcon(W.QMessageBox.Icon.Warning)
        caja.setWindowTitle("Revisar datos")
        caja.setText(mensaje)
        if detalle:
            caja.setDetailedText(detalle)
        caja.open()
        self._ultimo_error = caja

    def _tarea(self, funcion, callback, *args, **kwargs):
        if self.trabajador and self.trabajador.isRunning():
            self.estado.setText("Hay un cálculo en curso")
            return
        self.trabajador = Trabajador(funcion, *args, **kwargs)
        self._callback = callback
        self.trabajador.progreso.connect(self._progreso)
        self.trabajador.resultado.connect(self._finalizado)
        self.trabajador.error.connect(self._error)
        self.trabajador.finished.connect(self._liberar)
        self._bloquear(True)
        self.barra.setValue(0)
        self.estado.setText("Calculando…")
        self.trabajador.start()

    def _bloquear(self, bloqueado):
        self.analizar_btn.setEnabled(not bloqueado)
        for indice in PESTAÑAS_BLOQUEADAS:
            self.tabs.widget(indice).setEnabled(not bloqueado)
        self.cancelar_btn.setEnabled(bloqueado)

    def _progreso(self, fraccion, mensaje):
        self.barra.setValue(max(0, min(100, round(fraccion * 100))))
        self.estado.setText(mensaje)

    def _finalizado(self, resultado):
        self.barra.setValue(100)
        self.estado.setText("Listo")
        if self._callback:
            try:
                self._callback(resultado)
            except Exception as exc:
                self._error(str(exc), traceback.format_exc())

    def _liberar(self):
        self._bloquear(False)

    def cancelar(self):
        if self.trabajador and self.trabajador.isRunning():
            self.trabajador.cancelar()
            self.estado.setText("Cancelando…")

    def mostrar_resultado(self, resultado):
        anterior = (self.resultado_actual or {}).get("informe_path")
        if anterior != resultado.get("informe_path"):
            self._imagenes_vista = {}
        self.resultado_actual = resultado
        informe = resultado["informe"]
        self._mostrar_estado(informe)
        self._mostrar_calibracion(informe)
        self._mostrar_resumen(informe)
        self._mostrar_calidad(informe)
        self._mostrar_imagen()

    def _mostrar_estado(self, informe):
        self.veredicto.setText(f"Estado: {veredicto(informe)}")
        detalles = list(informe.get("motivos_no_concluyente") or [])
        detalles += [str(b).replace("_", " ") for b in informe.get("banderas", [])]
        detalles += [f"Desviación CAD: {condicion_pose(informe)}", f"HUD: {ETIQUETA_HUD}"]
        self.veredicto.setToolTip("\n".join(detalles))

    def closeEvent(self, evento):
        if self.trabajador and self.trabajador.isRunning():
            self.cancelar()
            evento.ignore()
            self.estado.setText("Cerrando el cálculo; cierre de nuevo cuando termine")
            return
        evento.accept()


def ejecutar(proyecto=None, informe=None):
    app = W.QApplication.instance() or W.QApplication(sys.argv[:1])
    app.setApplicationName("Vitro · Gemelo digital")
    ventana = VentanaPrincipal()
    if proyecto:
        ruta = Path(proyecto)
        ventana.restaurar(ruta.parent if ruta.is_file() else ruta)
    if informe:
        ventana.cargar_resultado(informe)
    ventana.show()
    return app.exec()
