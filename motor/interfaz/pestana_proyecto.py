import json
import tempfile
import traceback
from pathlib import Path

from motor import RAIZ, __version__
from motor.interfaz.componentes import (
    Entero,
    Selector,
    boton,
    numero,
    seccion,
    zona_texto,
)
from motor.interfaz.qt import QtWidgets as W

ARCHIVOS = (
    ("config", "Configuración", "JSON (*.json)"),
    ("mediciones", "Mediciones", "NPZ (*.npz)"),
    ("calibracion", "Calibración A", "NPZ (*.npz)"),
    ("validacion", "Validación B", "NPZ (*.npz)"),
    ("cad", "CAD nominal fijo", "STL (*.stl)"),
    ("pose", "Pose / datums", "JSON (*.json)"),
    ("mascara", "Máscara de adquisición", "NPZ (*.npz)"),
)
RECETA_EJEMPLO = RAIZ / "ejemplos" / "recetas.json"


class PestanaProyecto:
    def _crear_proyecto(self, disposicion):
        fila = W.QHBoxLayout()
        boton("Abrir carpeta…", self.abrir_carpeta, fila)
        boton("Guardar", self.guardar, fila)
        boton("Generar ejemplo", self.generar_ejemplo, fila)
        boton("Abrir resultado…", self.abrir_informe, fila)
        disposicion.addLayout(fila)
        self.carpeta_label = W.QLabel("Carpeta: sin seleccionar")
        disposicion.addWidget(self.carpeta_label)

        archivos, interior = seccion("Archivos")
        cuadricula = W.QGridLayout()
        self._rutas = {}
        for numero_fila, (clave, nombre, filtro) in enumerate(ARCHIVOS):
            entrada = W.QLineEdit()
            entrada.setPlaceholderText("Archivo local")
            entrada.editingFinished.connect(self.validar)
            self._rutas[clave] = entrada
            examinar = W.QPushButton("Buscar…")
            examinar.clicked.connect(lambda _, c=clave, f=filtro: self.buscar_archivo(c, f))
            cuadricula.addWidget(W.QLabel(nombre), numero_fila, 0)
            cuadricula.addWidget(entrada, numero_fila, 1)
            cuadricula.addWidget(examinar, numero_fila, 2)
        self.unidad_stl = Selector()
        self.unidad_stl.addItems(["µm", "mm"])
        cuadricula.addWidget(W.QLabel("Unidades del STL"), len(ARCHIVOS), 0)
        cuadricula.addWidget(self.unidad_stl, len(ARCHIVOS), 1)
        interior.addLayout(cuadricula)
        disposicion.addWidget(archivos)

        huecos, interior = seccion("Huecos del soporte (mm, marco CAD)")
        fila = W.QHBoxLayout()
        self.limites_hueco = []
        for nombre, valor in (("xmin", -10), ("xmax", 10), ("ymin", -10), ("ymax", 10)):
            control = numero(valor)
            self.limites_hueco.append(control)
            fila.addWidget(W.QLabel(nombre))
            fila.addWidget(control)
        boton("Excluir rectángulo", self.agregar_hueco, fila)
        boton("Quitar añadidos", self.quitar_huecos, fila)
        interior.addLayout(fila)
        self.huecos_label = W.QLabel("Huecos añadidos: 0")
        interior.addWidget(self.huecos_label)
        disposicion.addWidget(huecos)

        self.validacion_texto = zona_texto(altura=110)
        self.validacion_texto.setPlainText(
            "Falta config.json. Seleccione la configuración o genere un ejemplo sintético."
        )
        disposicion.addWidget(self.validacion_texto)
        fila = W.QHBoxLayout()
        boton("Revisar archivos", self.validar, fila)
        self.analizar_btn = boton("Analizar proyecto", self.analizar, fila)
        fila.addStretch()
        self.semilla = Entero()
        self.semilla.setRange(0, 2147483647)
        fila.addWidget(W.QLabel("Semilla sintética"))
        fila.addWidget(self.semilla)
        disposicion.addLayout(fila)
        disposicion.addStretch()

    def buscar_archivo(self, clave, filtro):
        ruta, _ = W.QFileDialog.getOpenFileName(
            self, "Seleccione un archivo", str(self.proyecto or Path.cwd()), filtro
        )
        if not ruta:
            return
        self._rutas[clave].setText(ruta)
        try:
            if clave == "config":
                self._cargar_config(Path(ruta))
            self.validar()
        except Exception as exc:
            self._error(f"No se pudo cargar el archivo: {exc}", traceback.format_exc())

    def abrir_carpeta(self):
        carpeta = W.QFileDialog.getExistingDirectory(
            self, "Carpeta del proyecto", str(self.proyecto or Path.cwd())
        )
        if carpeta:
            self.restaurar(carpeta)

    def restaurar(self, carpeta):
        self.proyecto = Path(carpeta).resolve()
        self.proyecto.mkdir(parents=True, exist_ok=True)
        self.carpeta_label.setText(f"Carpeta: {self.proyecto}")
        self.destino.setText(str(self.proyecto / "exportacion"))
        try:
            if (self.proyecto / "proyecto.vitro.json").exists():
                self._restaurar_estado(self._servicio().abrir_proyecto(self.proyecto))
            elif (self.proyecto / "config.json").exists():
                self._rutas["config"].setText(str(self.proyecto / "config.json"))
            config = self._rutas["config"].text()
            if config and Path(config).is_file():
                self._cargar_config(Path(config))
            self.validar()
        except Exception as exc:
            self._error(str(exc), traceback.format_exc())

    def _restaurar_estado(self, estado):
        if estado.get("config_path"):
            self._rutas["config"].setText(estado["config_path"])
        for clave, ruta in estado.get("rutas", {}).items():
            if clave not in self._rutas:
                continue
            if not ruta:
                self._rutas[clave].clear()
                continue
            p = Path(ruta)
            p = p if p.is_absolute() else self.proyecto / p
            opcional_vacia = clave in ("pose", "mascara") and p.is_dir()
            self._rutas[clave].setText("" if opcional_vacia else str(p))
        self.semilla.setValue(int(estado.get("semilla", 0)))
        self.aplicar_tema(bool(estado.get("tema_oscuro", False)))
        self.receta.setPlainText(estado.get("receta", self.receta.toPlainText()))
        for fila, valores in enumerate(estado.get("receta_tabla", [])):
            if fila < self.receta_tabla.rowCount():
                self.receta_tabla.cellWidget(fila, 0).setCurrentText(valores[0])
                for columna, valor in enumerate(valores[1:], 1):
                    self.receta_tabla.setItem(fila, columna, W.QTableWidgetItem(str(valor)))
                self._material_cambiado(fila, valores[0])
        self.w_base.setValue(float(estado.get("w_base", 0.35)))
        self.w_alt.setValue(float(estado.get("w_alt", 0.4)))
        self.usar_json.setChecked(bool(estado.get("usar_json", False)))
        self.usar_cov_a.setChecked(bool(estado.get("usar_cov_a", True)))
        self._huecos = estado.get("huecos_adicionales", [])
        self.huecos_label.setText(f"Huecos añadidos: {len(self._huecos)}")
        if estado.get("gemelo_resultado"):
            self._gemelo_evaluado(estado["gemelo_resultado"])
        self.unidad_stl.setCurrentText(estado.get("unidad_stl", "µm"))
        informe = estado.get("informe_path")
        if informe and Path(informe).is_file():
            self.cargar_resultado(informe)

    def _cargar_config(self, ruta):
        cfg = json.loads(ruta.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("config.json debe contener un objeto JSON con sus claves y valores.")
        for clave in ("calibracion", "capas", "cuna"):
            if clave in cfg and not isinstance(cfg[clave], dict):
                raise ValueError(f"{clave} debe ser un objeto JSON.")
        self._config = cfg
        mascara = cfg.get("mascara_adquisicion")
        valores = {
            "mediciones": cfg.get("mediciones"),
            "calibracion": cfg.get("calibracion", {}).get("archivo"),
            "validacion": cfg.get("calibracion", {}).get("validacion"),
            "cad": cfg.get("cad"),
            "mascara": (mascara or {}).get("archivo") if isinstance(mascara, dict) else mascara,
        }
        for clave, valor in valores.items():
            if isinstance(valor, str):
                self._rutas[clave].setText(str((ruta.parent / valor).resolve()))
        self.unidad_stl.setCurrentText("mm" if cfg.get("escala_cad", 1) == 1000 else "µm")
        capas = cfg.get("capas", {})
        tolerancias = capas.get("tol_d", [])
        for fila in range(min(3, len(capas.get("d_nom", [])))):
            self.tolerancias.setItem(fila, 1, W.QTableWidgetItem(str(capas["d_nom"][fila])))
            valor = str(tolerancias[fila]) if fila < len(tolerancias) else ""
            self.tolerancias.setItem(fila, 2, W.QTableWidgetItem(valor))
        self.cuna_nom.setValue(float(cfg.get("cuna", {}).get("w_nom", 0)))
        self.cuna_tol.setValue(float(cfg.get("cuna", {}).get("tol", 0.1)))
        if not self.receta.toPlainText() and RECETA_EJEMPLO.exists():
            self.receta.setPlainText(RECETA_EJEMPLO.read_text(encoding="utf-8"))

    def validar(self):
        if not self._rutas["config"].text().strip():
            self.validacion_texto.setPlainText(
                "Falta config.json. Seleccione la configuración o genere un ejemplo sintético."
            )
            return False
        try:
            cfg = self._config_efectiva()
            with tempfile.TemporaryDirectory(prefix="vitro-revisar-") as carpeta:
                temporal = Path(carpeta) / "config.json"
                temporal.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
                observaciones = self._servicio().validar_proyecto(temporal)
            self.validacion_texto.setPlainText("Archivos válidos.")
            self.validacion_texto.setToolTip("\n".join(observaciones))
            return True
        except Exception as exc:
            self.validacion_texto.setPlainText(str(exc))
            self.validacion_texto.setToolTip("")
            return False

    def guardar(self):
        if self.proyecto is None:
            self.abrir_carpeta()
        if self.proyecto is None:
            return
        receta_tabla = [
            [self.receta_tabla.cellWidget(j, 0).currentText()]
            + [self.receta_tabla.item(j, k).text() for k in range(1, 6)]
            for j in range(self.receta_tabla.rowCount())
        ]
        estado = {
            "rutas": {clave: entrada.text() for clave, entrada in self._rutas.items()},
            "semilla": self.semilla.value(),
            "tema_oscuro": self._tema_oscuro,
            "unidad_stl": self.unidad_stl.currentText(),
            "receta": self.receta.toPlainText(),
            "receta_tabla": receta_tabla,
            "w_base": self.w_base.value(),
            "w_alt": self.w_alt.value(),
            "usar_json": self.usar_json.isChecked(),
            "usar_cov_a": self.usar_cov_a.isChecked(),
            "gemelo_resultado": getattr(self, "_ultimo_gemelo", None),
            "huecos_adicionales": self._huecos,
            "informe_path": (self.resultado_actual or {}).get("informe_path"),
            "version": __version__,
        }
        try:
            ruta = self._servicio().guardar_proyecto(self.proyecto, estado)
            self.estado.setText(f"Proyecto guardado: {ruta}")
        except Exception as exc:
            self._error(str(exc), traceback.format_exc())

    def generar_ejemplo(self):
        if self.proyecto is None:
            self.abrir_carpeta()
        if self.proyecto is None:
            return
        self._tarea(
            self._servicio().generar_ejemplo,
            self._ejemplo_generado,
            str(self.proyecto / "ejemplo_sintetico"),
            semilla=self.semilla.value(),
        )

    def _ejemplo_generado(self, resultado):
        self._rutas["config"].setText(str(resultado["config_path"]))
        self._cargar_config(Path(resultado["config_path"]))
        self.validar()
        if resultado.get("informe_path"):
            self.cargar_resultado(resultado["informe_path"])
        self.estado.setText("Ejemplo sintético creado")

    def agregar_hueco(self):
        xmin, xmax, ymin, ymax = [c.value() * 1000 for c in self.limites_hueco]
        if xmin >= xmax or ymin >= ymax:
            self._error("Revise los límites: xmin debe ser menor que xmax e ymin menor que ymax.")
            return
        self._huecos.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        self.huecos_label.setText(f"Huecos añadidos: {len(self._huecos)}")
        self.validar()

    def quitar_huecos(self):
        self._huecos = []
        self.huecos_label.setText("Huecos añadidos: 0")
        self.validar()

    def _config_efectiva(self):
        original = Path(self._rutas["config"].text())
        cfg = json.loads(original.read_text(encoding="utf-8"))
        base = original.resolve().parent

        def absoluta(valor):
            p = Path(valor)
            return str((p if p.is_absolute() else base / p).resolve())

        for clave in ("mediciones", "cad"):
            valor = self._rutas[clave].text().strip() or cfg.get(clave)
            if isinstance(valor, str) and valor:
                cfg[clave] = absoluta(valor)
        calibracion = cfg.get("calibracion", {})
        if isinstance(calibracion, dict):
            for entrada, clave in (("calibracion", "archivo"), ("validacion", "validacion")):
                valor = self._rutas[entrada].text().strip() or calibracion.get(clave)
                if isinstance(valor, str) and valor:
                    calibracion[clave] = absoluta(valor)
            cfg["calibracion"] = calibracion
        pose = self._rutas["pose"].text().strip()
        if pose:
            if not Path(pose).is_file():
                raise ValueError(
                    "Seleccione un archivo JSON de pose / datums; una carpeta no es una pose."
                )
            objeto = json.loads(Path(pose).read_text(encoding="utf-8"))
            cfg["registro"] = objeto.get("registro", objeto)
        mascara = self._mascara_efectiva(cfg, absoluta)
        if mascara:
            cfg["mascara_adquisicion"] = mascara
        cfg["escala_cad"] = 1000.0 if self.unidad_stl.currentText() == "mm" else 1.0
        return cfg

    def _mascara_efectiva(self, cfg, absoluta):
        mascara = cfg.get("mascara_adquisicion") or {}
        if isinstance(mascara, str):
            mascara = {"archivo": mascara}
        if not isinstance(mascara, dict):
            raise ValueError("mascara_adquisicion debe contener archivo NPZ, incluir y/o excluir.")
        archivo = self._rutas["mascara"].text().strip() or mascara.get("archivo")
        if archivo:
            mascara["archivo"] = absoluta(archivo)
        if self._huecos:
            exclusiones = mascara.setdefault("excluir", [])
            exclusiones.extend(p for p in self._huecos if p not in exclusiones)
        return mascara

    def _preparar_config(self):
        cfg = self._config_efectiva()
        carpeta = self.proyecto or Path(self._rutas["config"].text()).parent
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / "config_interfaz.json"
        ruta.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return ruta

    def analizar(self):
        if not self.validar():
            return
        try:
            ruta = self._preparar_config()
            self._tarea(
                self._servicio().analizar_proyecto,
                self._analizado,
                str(ruta),
                salida=str((self.proyecto or ruta.parent) / "resultados"),
            )
        except Exception as exc:
            self._error(str(exc), traceback.format_exc())

    def _analizado(self, resultado):
        self.mostrar_resultado(resultado)
        self.tabs.setCurrentIndex(2)
        if self.proyecto:
            self.guardar()

    def abrir_informe(self):
        ruta, _ = W.QFileDialog.getOpenFileName(
            self, "Abrir resultado", str(self.proyecto or Path.cwd()), "JSON (*.json)"
        )
        if not ruta:
            return
        try:
            self.cargar_resultado(ruta)
        except Exception as exc:
            self._error(str(exc), traceback.format_exc())

    def cargar_resultado(self, ruta):
        self.mostrar_resultado(self._servicio().cargar_resultado(ruta))
