import json

from motor.interfaz.componentes import boton, zona_texto
from motor.interfaz.qt import QtWidgets as W

FORMATOS = (
    ("stl", "STL · interfaces y sólidos por capa (µm)"),
    ("pdf", "PDF · informe A4 horizontal"),
    ("json", "JSON · resultados y condiciones"),
    ("npz", "NPZ · mapas y máscara de soporte"),
    ("png", "PNG · mapas, vista 3D y corte"),
)


class PestanaExportar:
    def _crear_exportar(self, disposicion):
        self.formatos = {}
        for clave, etiqueta in FORMATOS:
            casilla = W.QCheckBox(etiqueta)
            casilla.setChecked(True)
            disposicion.addWidget(casilla)
            self.formatos[clave] = casilla
        fila = W.QHBoxLayout()
        self.destino = W.QLineEdit()
        self.destino.setPlaceholderText("Carpeta de exportación")
        fila.addWidget(self.destino)
        boton("Elegir…", self.elegir_destino, fila)
        disposicion.addLayout(fila)
        boton("Exportar selección", self.exportar, disposicion)
        self.exportacion_texto = zona_texto(pista="Manifiesto de la exportación")
        disposicion.addWidget(self.exportacion_texto, 1)

    def elegir_destino(self):
        carpeta = W.QFileDialog.getExistingDirectory(
            self, "Carpeta de exportación", self.destino.text() or str(self.proyecto or ".")
        )
        if carpeta:
            self.destino.setText(carpeta)

    def exportar(self):
        if not self.resultado_actual:
            self._error("Primero analice un proyecto o abra un resultado.")
            return
        formatos = tuple(clave for clave, casilla in self.formatos.items() if casilla.isChecked())
        if not formatos:
            self._error("Seleccione al menos un formato.")
            return
        destino = self.destino.text().strip()
        if not destino:
            self.elegir_destino()
            destino = self.destino.text().strip()
        if destino:
            self._tarea(
                self._servicio().exportar_resultados,
                self._exportado,
                self.resultado_actual["informe_path"],
                destino,
                formatos=formatos,
            )

    def _exportado(self, resultado):
        if "archivos" in resultado:
            lineas = [f"Carpeta: {resultado.get('carpeta', '')}", ""]
            lineas += [f"  {archivo}" for archivo in resultado["archivos"]]
            lineas += ["", f"Manifiesto: {resultado.get('manifest', '')}"]
            self.exportacion_texto.setPlainText("\n".join(lineas))
        else:
            self.exportacion_texto.setPlainText(
                json.dumps(resultado, ensure_ascii=False, indent=2, default=str)
            )
        self._mostrar_imagen()
