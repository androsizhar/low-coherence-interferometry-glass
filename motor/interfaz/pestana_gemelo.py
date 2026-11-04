import json

from motor import RAIZ
from motor.interfaz.componentes import (
    CATALOGO,
    Selector,
    agregar_fila,
    boton,
    numero,
    tabla,
    texto,
    zona_texto,
)
from motor.interfaz.qt import QtCore as C
from motor.interfaz.qt import QtGui as G
from motor.interfaz.qt import QtWidgets as W

MATERIALES = ("BK7", "PVB_SUPUESTO", "SODOCALCICO_SUPUESTO", "Cauchy")
CAPAS_BASE = (("BK7", 2100), ("PVB_SUPUESTO", 760), ("BK7", 1600))
COEFICIENTES = ((1, "1.5"), (2, "0"), (3, "0"))
RECETA_EJEMPLO = RAIZ / "ejemplos" / "recetas.json"


def _porcentaje(fraccion):
    if fraccion is None:
        return "no disponible"
    return "≈ 0 %" if fraccion == 0 else f"{fraccion * 100:.3g} %"


class PestanaGemelo:
    def _crear_gemelo(self, disposicion):
        self.receta_tabla = tabla(
            ["Material", "A Cauchy", "B (µm²)", "C (µm⁴)", "Base (µm)", "Alternativa (µm)"],
            editable=True,
        )
        self.receta_tabla.setMaximumHeight(165)
        for j, (material, espesor) in enumerate(CAPAS_BASE):
            alternativa = espesor + (1 if j == 1 else 0)
            agregar_fila(
                self.receta_tabla,
                [material, CATALOGO, CATALOGO, CATALOGO, espesor, alternativa],
            )
            combo = Selector()
            combo.addItems(MATERIALES)
            combo.setCurrentText(material)
            combo.currentTextChanged.connect(lambda t, fila=j: self._material_cambiado(fila, t))
            self.receta_tabla.setCellWidget(j, 0, combo)
            self._material_cambiado(j, material)
        disposicion.addWidget(self.receta_tabla)

        fila = W.QHBoxLayout()
        self.w_base = numero(0.35, -100, 100, 5)
        self.w_alt = numero(0.4, -100, 100, 5)
        self.usar_cov_a = W.QCheckBox("Ruido de muestra A")
        self.usar_cov_a.setChecked(True)
        self.usar_json = W.QCheckBox("Editar JSON")
        fila.addWidget(W.QLabel("Cuña base (mrad)"))
        fila.addWidget(self.w_base)
        fila.addWidget(W.QLabel("Cuña alternativa (mrad)"))
        fila.addWidget(self.w_alt)
        fila.addWidget(self.usar_cov_a)
        fila.addWidget(self.usar_json)
        disposicion.addLayout(fila)

        self.receta = W.QPlainTextEdit()
        self.receta.setMaximumHeight(160)
        self.receta.setVisible(False)
        self.receta.setPlaceholderText("Receta en JSON")
        if RECETA_EJEMPLO.exists():
            self.receta.setPlainText(RECETA_EJEMPLO.read_text(encoding="utf-8"))
        self.usar_json.toggled.connect(self.receta.setVisible)
        disposicion.addWidget(self.receta, 1)

        fila = W.QHBoxLayout()
        boton("Cargar receta…", self.cargar_receta, fila)
        boton("Evaluar receta", self.evaluar_gemelo, fila)
        disposicion.addLayout(fila)
        self.gemelo_resumen = zona_texto(pista="Distinguibilidad, sensibilidad e imagen doble")
        disposicion.addWidget(self.gemelo_resumen, 1)

    def _material_cambiado(self, fila, material):
        cauchy = material == "Cauchy"
        editable = C.Qt.ItemFlag.ItemIsEditable
        for columna, defecto in COEFICIENTES:
            item = self.receta_tabla.item(fila, columna)
            if item is None:
                continue
            if cauchy and item.text() == CATALOGO:
                item.setText(defecto)
            elif not cauchy:
                item.setText(CATALOGO)
            banderas = item.flags()
            item.setFlags(banderas | editable if cauchy else banderas & ~editable)
            item.setForeground(G.QBrush() if cauchy else G.QBrush(G.QColor("#8a96a0")))

    def cargar_receta(self):
        ruta, _ = W.QFileDialog.getOpenFileName(
            self, "Cargar receta", str(self.proyecto or "."), "JSON (*.json)"
        )
        if ruta:
            with open(ruta, encoding="utf-8") as archivo:
                self.receta.setPlainText(archivo.read())
            self.usar_json.setChecked(True)

    def _receta_desde_tabla(self):
        base, alternativa = [], []
        for j in range(self.receta_tabla.rowCount()):
            material = self.receta_tabla.cellWidget(j, 0).currentText()
            if material == "Cauchy":
                coeficientes = {
                    nombre: float(self.receta_tabla.item(j, k).text())
                    for k, nombre in ((1, "A"), (2, "B"), (3, "C"))
                }
                material = {"tipo": "cauchy", "nombre": f"Propuesta capa {j + 1}", **coeficientes}
            base.append({"material": material, "d": float(self.receta_tabla.item(j, 4).text())})
            alternativa.append(
                {"material": material, "d": float(self.receta_tabla.item(j, 5).text())}
            )
        return {
            "receta_base": base,
            "alternativas": [
                {
                    "nombre": "Propuesta de la interfaz",
                    "capas": alternativa,
                    "w_mrad": self.w_alt.value(),
                }
            ],
            "w_base_mrad": self.w_base.value(),
        }

    def evaluar_gemelo(self):
        try:
            config = json.loads(self.receta.toPlainText())
            if not self.usar_json.isChecked():
                config.update(self._receta_desde_tabla())
            if self.usar_cov_a.isChecked():
                calibracion = (
                    (self.resultado_actual or {}).get("informe", {}).get("calibracion", {})
                )
                covarianza = calibracion.get("cov_a_um2")
                if covarianza is None:
                    self._error(
                        "No hay covarianza de la muestra A. Analice el proyecto o desmarque "
                        "«Ruido de muestra A»."
                    )
                    return
                config["ruido"] = {"covarianza": covarianza}
            self._tarea(self._servicio().evaluar_gemelo, self._gemelo_evaluado, config)
        except json.JSONDecodeError as exc:
            self._error(f"La receta no es JSON válido (línea {exc.lineno}): {exc.msg}")
        except (ValueError, TypeError) as exc:
            self._error(f"Revise los números de la receta: {exc}")

    def _gemelo_evaluado(self, resultado):
        self._ultimo_gemelo = resultado
        lineas = ["Ruido: " + str(resultado.get("ruido", {}).get("origen", "no declarado"))]
        for receta in [resultado.get("base", {})] + resultado.get("alternativas", []):
            lineas += ["", receta.get("nombre", "Receta base")]
            lineas.append(
                "Espesores (µm): " + ", ".join(texto(x) for x in receta.get("espesores_um", []))
            )
            sensibilidad = receta.get("sensibilidad_indices_exactos", {})
            minimos = sensibilidad.get("cambio_minimo_local_um", [])
            lineas.append(
                "Cambio mínimo detectable por capa, 3σ (µm): "
                + ", ".join(texto(x) for x in minimos)
            )
            comparacion = receta.get("comparacion_con_base", {})
            if comparacion:
                lineas.append(
                    f"Distancia Δ: {texto(comparacion.get('distancia_mahalanobis'))} · "
                    f"distinguible: {texto(comparacion.get('distinguible_condicional'))}"
                )
                lineas.append(
                    "Error de clasificación (modelo gaussiano): "
                    + _porcentaje(comparacion.get("error_clasificacion_ideal"))
                )
            hud = receta.get("hud", {})
            lineas.append(
                f"Imagen doble: {texto(hud.get('separacion_firmada_arcmin'))} minutos de arco"
            )
        espesor = (self.resultado_actual or {}).get("informe", {}).get("espesor", {}) or {}
        sistematica = [v.get("sigma_sistematica_um") for v in espesor.values()]
        if sistematica and all(x is not None for x in sistematica):
            lineas += [
                "",
                "σ sistemática del índice calibrado por capa (µm): "
                + ", ".join(texto(x) for x in sistematica),
                "Δ y los cambios mínimos usan solo la repetibilidad de la muestra A.",
            ]
        self.gemelo_resumen.setPlainText("\n".join(lineas))
