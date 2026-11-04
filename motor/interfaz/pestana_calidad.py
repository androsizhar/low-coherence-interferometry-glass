import json
import traceback

from motor.estado import condicion_pose
from motor.interfaz.componentes import (
    agregar_fila,
    boton,
    numero,
    tabla,
    texto,
    zona_texto,
)
from motor.interfaz.qt import QtWidgets as W


def _porcentaje(fraccion):
    return texto(fraccion * 100 if fraccion is not None else None)


class PestanaCalidad:
    def _crear_calidad(self, disposicion):
        self.tolerancias = tabla(
            ["Capa", "Espesor nominal (µm)", "Tolerancia ± (µm)"], editable=True
        )
        for nombre in ("Vidrio exterior", "PVB", "Vidrio interior"):
            agregar_fila(self.tolerancias, [nombre, "", ""])
        self.tolerancias.setMaximumHeight(150)
        disposicion.addWidget(self.tolerancias)
        formulario = W.QFormLayout()
        self.cuna_nom = numero(0.0, -1000, 1000, 5)
        self.cuna_tol = numero(0.1, 0, 1000, 5)
        for control in (self.cuna_nom, self.cuna_tol):
            control.setMaximumWidth(180)
        formulario.addRow("Cuña nominal firmada (mrad)", self.cuna_nom)
        formulario.addRow("Tolerancia de cuña ± (mrad)", self.cuna_tol)
        disposicion.addLayout(formulario)
        boton("Aplicar y volver a analizar", self.aplicar_tolerancias, disposicion)
        self.calidad_resumen = zona_texto(pista="Fracción de nodos fuera de tolerancia")
        disposicion.addWidget(self.calidad_resumen)

    def _mostrar_calidad(self, informe):
        lineas = []
        for nombre, valores in informe.get("espesor", {}).items():
            fraccion = valores.get("fraccion_zona_fuera_de_tolerancia")
            lineas.append(f"{nombre}: {_porcentaje(fraccion)} % de nodos fuera de tolerancia")
        cuna = informe.get("cuna_pvb_mrad", {})
        lineas.append(
            "Cuña PVB: "
            f"{_porcentaje(cuna.get('fraccion_zona_fuera_de_tolerancia'))} % de nodos fuera de tolerancia"
        )
        for nombre, valores in informe.get("desviacion_vs_cad_um", {}).items():
            lineas.append(
                f"{nombre}: probabilidad puntual media = {texto(valores.get('P_fuera_media'))}"
            )
        lineas += ["", "Pose: " + condicion_pose(informe)]
        self.calidad_resumen.setPlainText("\n".join(lineas))

    def aplicar_tolerancias(self):
        try:
            ruta = self._preparar_config()
            cfg = json.loads(ruta.read_text(encoding="utf-8"))
            capas = cfg.setdefault("capas", {})
            capas["d_nom"] = [float(self.tolerancias.item(j, 1).text()) for j in range(3)]
            capas["tol_d"] = [float(self.tolerancias.item(j, 2).text()) for j in range(3)]
            if any(x <= 0 for x in capas["d_nom"] + capas["tol_d"]):
                raise ValueError("Escriba espesores y tolerancias positivos en µm.")
            cfg.setdefault("cuna", {}).update(
                w_nom=self.cuna_nom.value(), tol=self.cuna_tol.value()
            )
            ruta.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            self._rutas["config"].setText(str(ruta))
            self.analizar()
        except Exception as exc:
            self._error(f"Revise las tolerancias: {exc}", traceback.format_exc())
