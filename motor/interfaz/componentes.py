import json
import threading
import traceback
from pathlib import Path

from motor.interfaz.qt import QtCore as C
from motor.interfaz.qt import QtGui as G
from motor.interfaz.qt import QtWidgets as W
from motor.interfaz.qt import Signal

CATALOGO = "(catálogo)"
FUENTE_FIJA = "font-family: Menlo, Consolas, 'DejaVu Sans Mono', monospace;"


def texto(valor, precision=6):
    if valor is None:
        return "no disponible"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, float):
        return f"{valor:.{precision}g}"
    if isinstance(valor, (list, dict)):
        return json.dumps(valor, ensure_ascii=False, indent=2)
    return str(valor)


def matriz_texto(matriz, precision=4):
    es_matriz = (
        isinstance(matriz, (list, tuple))
        and matriz
        and all(isinstance(fila, (list, tuple)) for fila in matriz)
    )
    if not es_matriz:
        return "no disponible"
    filas = [[f"{float(v):+.{precision}e}" for v in fila] for fila in matriz]
    return "\n".join("  ".join(fila) for fila in filas)


def boton(titulo, accion, disposicion):
    b = W.QPushButton(titulo)
    b.clicked.connect(accion)
    disposicion.addWidget(b)
    return b


def tabla(columnas, editable=False):
    t = W.QTableWidget(0, len(columnas))
    t.setHorizontalHeaderLabels(columnas)
    t.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
    t.verticalHeader().setVisible(False)
    t.setAlternatingRowColors(True)
    if not editable:
        t.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
    return t


def agregar_fila(t, valores):
    fila = t.rowCount()
    t.insertRow(fila)
    for columna, valor in enumerate(valores):
        t.setItem(fila, columna, W.QTableWidgetItem(texto(valor)))


def zona_texto(altura=None, fija=False, pista=""):
    z = W.QPlainTextEdit()
    z.setReadOnly(True)
    if altura:
        z.setMaximumHeight(altura)
    if fija:
        z.setStyleSheet(FUENTE_FIJA)
    z.setPlaceholderText(pista)
    return z


def _sin_rueda(clase):
    class SinRueda(clase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(C.Qt.FocusPolicy.StrongFocus)

        def wheelEvent(self, evento):
            if self.hasFocus():
                super().wheelEvent(evento)
            else:
                evento.ignore()

    return SinRueda


Selector = _sin_rueda(W.QComboBox)
Entero = _sin_rueda(W.QSpinBox)
Decimal = _sin_rueda(W.QDoubleSpinBox)


def numero(valor=0.0, minimo=-1e6, maximo=1e6, decimales=3):
    c = Decimal()
    c.setRange(minimo, maximo)
    c.setDecimals(decimales)
    c.setValue(valor)
    return c


def seccion(titulo):
    grupo = W.QGroupBox(titulo)
    disposicion = W.QVBoxLayout(grupo)
    return grupo, disposicion


class Trabajador(C.QThread):
    progreso = Signal(float, str)
    resultado = Signal(object)
    error = Signal(str, str)

    def __init__(self, funcion, *args, **kwargs):
        super().__init__()
        self.funcion, self.args, self.kwargs = funcion, args, kwargs
        self.cancelacion = threading.Event()

    def run(self):
        try:
            resultado = self.funcion(
                *self.args,
                **self.kwargs,
                progreso=lambda fraccion, mensaje: self.progreso.emit(
                    float(fraccion), str(mensaje)
                ),
                cancelar=self.cancelacion.is_set,
            )
            self.resultado.emit(resultado)
        except Exception as exc:
            self.error.emit(str(exc), traceback.format_exc())

    def cancelar(self):
        self.cancelacion.set()


class Imagen(W.QLabel):
    def __init__(self):
        super().__init__("Sin visualización")
        self.setAlignment(C.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(500, 300)
        self.original = None
        self.setWordWrap(True)

    def cargar(self, ruta):
        self.original = G.QPixmap(str(ruta))
        if self.original.isNull():
            self.original = None
            self.setText(f"No hay imagen disponible: {Path(ruta).name}")
        else:
            self._ajustar()

    def vaciar(self, mensaje):
        self.original = None
        self.setText(mensaje)

    def _ajustar(self):
        if self.original:
            self.setPixmap(
                self.original.scaled(
                    self.size(),
                    C.Qt.AspectRatioMode.KeepAspectRatio,
                    C.Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        self._ajustar()
