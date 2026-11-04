from motor.interfaz.componentes import (
    agregar_fila,
    matriz_texto,
    seccion,
    tabla,
    zona_texto,
)


class PestanaCalibracion:
    def _crear_calibracion(self, disposicion):
        muestra_a, interior = seccion("Muestra A · espesor conocido")
        self.tabla_a = tabla(
            ["Capa", "n de grupo", "Repetibilidad σa (µm)", "σ relativa sistemática"]
        )
        self.tabla_a.setMaximumHeight(150)
        interior.addWidget(self.tabla_a)
        self.covarianza = zona_texto(altura=105, fija=True, pista="Covarianza del índice de grupo")
        interior.addWidget(self.covarianza)
        disposicion.addWidget(muestra_a)

        muestra_b, interior = seccion("Muestra B · contraste sin recalibrar")
        self.tabla_b = tabla(["Capa", "Error (µm)", "σ combinada (µm)", "z", "Compatible"])
        self.tabla_b.setMaximumHeight(150)
        interior.addWidget(self.tabla_b)
        disposicion.addWidget(muestra_b)
        disposicion.addStretch()

    def _mostrar_calibracion(self, informe):
        calibracion = informe.get("calibracion", {})
        ng = calibracion.get("ng", [])
        sigma = calibracion.get("sigma_a_um", [])
        sistematica = calibracion.get("u_rel_sistematica", [])
        self.tabla_a.setRowCount(0)
        for j, valor in enumerate(ng):
            agregar_fila(
                self.tabla_a,
                [
                    j + 1,
                    valor,
                    sigma[j] if j < len(sigma) else None,
                    sistematica[j] if j < len(sistematica) else None,
                ],
            )
        encabezado = "Covarianza de n_g (adimensional²) · filas y columnas: capas 1, 2, 3"
        self.covarianza.setPlainText(encabezado + "\n" + matriz_texto(calibracion.get("cov_ng")))

        validacion = informe.get("validacion", informe.get("validacion_cruzada", {}))
        self.tabla_b.setRowCount(0)
        if not isinstance(validacion, dict):
            return
        errores = validacion.get(
            "error_um", validacion.get("errores_um", validacion.get("error", []))
        )
        if not isinstance(errores, list):
            return
        combinada = validacion.get("sigma_combinada_um", validacion.get("sigma_combinada", []))
        z = validacion.get("z", [])
        compatible = validacion.get("compatible_por_capa")
        if compatible is None:
            compatible = [abs(x) < 3.0 for x in z] if isinstance(z, list) else None

        def elemento(valores, j):
            return valores[j] if isinstance(valores, list) and j < len(valores) else None

        for j, valor in enumerate(errores):
            agregar_fila(
                self.tabla_b,
                [j + 1, valor, elemento(combinada, j), elemento(z, j), elemento(compatible, j)],
            )
