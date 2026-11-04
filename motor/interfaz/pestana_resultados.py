import traceback
from pathlib import Path

from motor.estado import condicion_pose
from motor.interfaz.componentes import (
    Entero,
    Imagen,
    Selector,
    boton,
    numero,
    texto,
    zona_texto,
)
from motor.interfaz.qt import QtWidgets as W

VISTAS = (("mapa", "mapas.png"), ("3d", "vista_3d.png"), ("corte", "corte.png"))
MAPAS = (
    ("Espesor (µm)", "espesor"),
    ("Desviación CAD (µm)", "desviacion"),
    ("Cuña (mrad)", "cuna"),
    ("Incertidumbre (µm)", "incertidumbre"),
    ("Probabilidad puntual", "probabilidad"),
)


def _coordenadas(entrada):
    cadena = entrada.text().strip()
    if not cadena:
        return None
    valores = [float(x.strip()) * 1000 for x in cadena.split(",")]
    if len(valores) != 2:
        raise ValueError("Escriba las coordenadas x,y en mm, por ejemplo -30,0.")
    return valores


class PestanaResultados:
    def _crear_resultados(self, disposicion):
        fila = W.QHBoxLayout()
        self.tipo_vista = Selector()
        self.tipo_vista.addItems(["Mapas 2D", "Vista 3D", "Corte transversal"])
        self.tipo_vista.currentIndexChanged.connect(self._mostrar_imagen)
        self.mapa_selector = Selector()
        for etiqueta, clave in MAPAS:
            self.mapa_selector.addItem(etiqueta, clave)
        self.capa_selector = Entero()
        self.capa_selector.setRange(1, 3)
        self.capa_selector.setValue(2)
        self.interfaz_selector = Entero()
        self.interfaz_selector.setRange(0, 3)
        self.interfaz_selector.setValue(1)
        for etiqueta, control in (
            ("Vista", self.tipo_vista),
            ("Mapa", self.mapa_selector),
            ("Capa", self.capa_selector),
            ("Interfaz", self.interfaz_selector),
        ):
            fila.addWidget(W.QLabel(etiqueta))
            fila.addWidget(control)
        boton("Actualizar", self.actualizar_vista, fila)
        boton("Abrir 3D interactiva", self.abrir_3d, fila)
        boton("Guardar PNG…", self.guardar_captura, fila)
        fila.addStretch()
        disposicion.addLayout(fila)

        fila = W.QHBoxLayout()
        self.exageracion = numero(1, 1, 1000, 2)
        self.explotada = W.QCheckBox("Separar capas en 3D")
        self.corte_inicio = W.QLineEdit()
        self.corte_inicio.setPlaceholderText("Corte: x,y inicial (mm); vacío = centro")
        self.corte_fin = W.QLineEdit()
        self.corte_fin.setPlaceholderText("Corte: x,y final (mm)")
        fila.addWidget(W.QLabel("Escala vertical ×"))
        fila.addWidget(self.exageracion)
        fila.addWidget(self.explotada)
        fila.addWidget(self.corte_inicio, 1)
        fila.addWidget(self.corte_fin, 1)
        disposicion.addLayout(fila)

        self.imagen = Imagen()
        disposicion.addWidget(self.imagen, 1)
        self.resumen_resultados = zona_texto(altura=130, pista="Espesores, cuña y condiciones")
        disposicion.addWidget(self.resumen_resultados)

    def _mostrar_resumen(self, informe):
        lineas = ["Espesor · media ± σ aleatoria ± σ sistemática (µm)"]
        for nombre, valores in informe.get("espesor", {}).items():
            lineas.append(
                f"{nombre}: {texto(valores.get('media_um'))} ± "
                f"{texto(valores.get('sigma_aleatoria_um'))} ± "
                f"{texto(valores.get('sigma_sistematica_um'))}"
            )
        cuna = informe.get("cuna_pvb_mrad", {})
        lineas.append(
            f"Cuña PVB firmada: {texto(cuna.get('media'))} ± {texto(cuna.get('sigma_media'))} mrad"
        )
        lineas += ["", "Pose: " + condicion_pose(informe)]
        self.resumen_resultados.setPlainText("\n".join(lineas))

    def _mostrar_imagen(self, *_):
        if not self.resultado_actual:
            return
        carpeta = Path(
            self.resultado_actual.get("carpeta", Path(self.resultado_actual["informe_path"]).parent)
        )
        indice = self.tipo_vista.currentIndex()
        ruta = self._imagenes_vista.get(indice, carpeta / VISTAS[indice][1])
        if ruta.exists():
            self.imagen.cargar(ruta)
        else:
            self.imagen.vaciar(f"{ruta.name} no está generado. Exporte PNG o PDF para crearlo.")

    def actualizar_vista(self):
        if not self.resultado_actual:
            self._error("Primero analice un proyecto o abra un resultado.")
            return
        try:
            tipo = VISTAS[self.tipo_vista.currentIndex()][0]
            carpeta = Path(self.resultado_actual["informe_path"]).parent
            self._tarea(
                self._servicio().generar_visualizacion,
                self._vista_generada,
                self.resultado_actual["informe_path"],
                tipo=tipo,
                destino=str(carpeta / f"vista_interfaz_{tipo}.png"),
                mapa=self.mapa_selector.currentData(),
                capa=self.capa_selector.value() - 1,
                interfaz=self.interfaz_selector.value(),
                inicio=_coordenadas(self.corte_inicio),
                fin=_coordenadas(self.corte_fin),
                exageracion=self.exageracion.value(),
                explotada=self.explotada.isChecked(),
            )
        except Exception as exc:
            self._error(str(exc), traceback.format_exc())

    def _vista_generada(self, resultado):
        ruta = Path(resultado["imagen_path"])
        self._imagenes_vista[self.tipo_vista.currentIndex()] = ruta
        self.imagen.cargar(ruta)

    def guardar_captura(self):
        if not self.imagen.original:
            self._error("Primero genere una visualización.")
            return
        ruta, _ = W.QFileDialog.getSaveFileName(self, "Guardar figura", "mapa.png", "PNG (*.png)")
        if ruta:
            self.imagen.original.save(ruta)

    def abrir_3d(self):
        if not self.resultado_actual:
            self._error("Primero analice un proyecto o abra un resultado.")
            return
        try:
            self._servicio().abrir_vista_3d(
                self.resultado_actual["informe_path"],
                mapa=self.mapa_selector.currentData(),
                capa=self.capa_selector.value() - 1,
                interfaz=self.interfaz_selector.value(),
                exageracion=self.exageracion.value(),
                explotada=self.explotada.isChecked(),
            )
            self.estado.setText("Abriendo vista 3D")
        except Exception as exc:
            self._error(str(exc), traceback.format_exc())
