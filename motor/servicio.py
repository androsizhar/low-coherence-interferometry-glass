from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from motor import __version__ as VERSION


class ErrorProyecto(ValueError):
    pass


class OperacionCancelada(ErrorProyecto):
    pass


def _leer_json(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise ErrorProyecto(f"No se puede abrir {Path(ruta).name}: {exc}") from exc


def guardar_proyecto(carpeta, estado):
    carpeta = Path(carpeta).resolve()
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / "proyecto.vitro.json"
    contenido = {"version": VERSION, "estado": estado}
    temporal = destino.with_suffix(".tmp")
    temporal.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporal.replace(destino)
    return destino


def abrir_proyecto(carpeta):
    ruta = Path(carpeta)
    if ruta.is_dir():
        ruta = ruta / "proyecto.vitro.json"
    j = _leer_json(ruta)
    if not isinstance(j, dict) or not isinstance(j.get("estado"), dict):
        raise ErrorProyecto("El archivo no es un proyecto Vitro. Abra proyecto.vitro.json.")
    return j["estado"]


def validar_proyecto(config_path):
    p = Path(config_path).resolve()
    c = _leer_json(p)
    if not isinstance(c, dict):
        raise ErrorProyecto("config.json debe contener un objeto JSON.")
    for key in (
        "mediciones",
        "calibracion",
        "cad",
        "instrumento",
        "capas",
        "cuna",
        "rango_frontal",
        "registro",
    ):
        if key not in c:
            raise ErrorProyecto(
                f"Falta la clave {key} en config.json. Complete el campo en Proyecto."
            )
    archivos = [("mediciones", c["mediciones"]), ("cad", c["cad"])]
    if not isinstance(c["calibracion"], dict) or not c["calibracion"].get("archivo"):
        raise ErrorProyecto("Falta calibracion.archivo en config.json. Seleccione la muestra A.")
    archivos.append(("calibracion", c["calibracion"]["archivo"]))
    if c["calibracion"].get("validacion"):
        archivos.append(("validacion", c["calibracion"]["validacion"]))
    for nombre, ruta in archivos:
        if not isinstance(ruta, str) or not ruta:
            raise ErrorProyecto(f"Seleccione el archivo {nombre}.")
        q = Path(ruta)
        q = q if q.is_absolute() else p.parent / q
        if not q.is_file():
            raise ErrorProyecto(
                f"No se encontró el archivo {nombre}: {q}. Seleccione su ubicación actual."
            )
    for key in ("instrumento", "capas", "cuna", "registro"):
        if not isinstance(c[key], dict):
            raise ErrorProyecto(f"{key} debe ser un objeto JSON.")
    if c["registro"].get("modo") not in ("fijo", "icp_forma"):
        raise ErrorProyecto(
            "Seleccione registro.modo fijo con datums o icp_forma para forma relativa."
        )
    avisos = []
    if not c["calibracion"].get("validacion"):
        avisos.append("Falta la muestra B: el veredicto será no concluyente.")
    if c.get("origen_datos") != "real_declarado":
        avisos.append("Origen de datos simulado o no declarado; resultado de investigación.")
    if c["registro"].get("modo") == "fijo" and c["registro"].get("covarianza") is None:
        avisos.append(
            "Covarianza de pose no disponible; no se decide desviación absoluta con incertidumbre completa."
        )
    return avisos


def cargar_resultado(informe_path):
    from motor.estado import aplicar_estado

    p = Path(informe_path).resolve()
    informe = aplicar_estado(_leer_json(p))
    mapas = p.parent / "mapas.npz"
    return {
        "informe_path": str(p),
        "mapas_path": str(mapas),
        "carpeta": str(p.parent),
        "informe": informe,
    }


def _motor(accion, solicitud, progreso=None, cancelar=None):
    ejecutable = os.environ.get("VITRO_PYTHON") or sys.executable
    raiz = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(raiz)
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vitro-matplotlib"))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="vitro-tarea-") as tmp:
        req = Path(tmp) / "solicitud.json"
        respuesta = Path(tmp) / "respuesta.json"
        solicitud = dict(solicitud)
        if accion == "analizar":
            solicitud["salida_temporal"] = str(Path(tmp) / "resultado")
        req.write_text(json.dumps(solicitud, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        cmd = [
            ejecutable,
            "-m",
            "motor.servicio",
            "--worker",
            accion,
            "--solicitud",
            str(req),
            "--respuesta",
            str(respuesta),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(raiz),
            )
        except OSError as exc:
            raise ErrorProyecto(
                "No se puede iniciar Python de cálculo. Revise el intérprete y sus dependencias."
            ) from exc
        cola = queue.Queue()
        registro = []

        def leer():
            for linea in proc.stdout:
                cola.put(linea)
            cola.put(None)

        threading.Thread(target=leer, daemon=True).start()
        try:
            while True:
                if cancelar is not None and cancelar():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise OperacionCancelada(
                        "Operación cancelada; no se ha sustituido el resultado anterior."
                    )
                try:
                    linea = cola.get(timeout=0.08)
                except queue.Empty:
                    continue
                if linea is None:
                    break
                registro.append(linea.rstrip())
                try:
                    evento = json.loads(linea)
                    if isinstance(evento, dict) and "progreso" in evento and progreso:
                        progreso(float(evento["progreso"]), evento.get("mensaje", "Calculando"))
                except (ValueError, TypeError):
                    pass
            codigo = proc.wait()
            respuesta_datos = _leer_json(respuesta) if respuesta.exists() else None
            if codigo or respuesta_datos is None or "error" in respuesta_datos:
                detalle = "\n".join(registro[-150:])
                error = (respuesta_datos or {}).get(
                    "error", "No terminó el cálculo. Revise las dependencias del motor."
                )
                raise ErrorProyecto(error + "\nDetalle técnico:\n" + detalle)
            if accion == "analizar":
                cp = Path(solicitud["config_path"])
                cfg = _leer_json(cp)
                destino = Path(solicitud.get("salida") or cfg.get("salida", "salida"))
                if not destino.is_absolute():
                    destino = cp.parent / destino
                _publicar_resultado(Path(solicitud["salida_temporal"]), destino, cp)
                return cargar_resultado(destino / "informe.json")
            return respuesta_datos
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()


def _publicar_resultado(origen, destino, config_path):
    destino = destino.resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and any(destino.iterdir()) and not (destino / "informe.json").is_file():
        raise ErrorProyecto(
            "La carpeta de salida contiene archivos ajenos al resultado. Elija una carpeta vacía o una salida Vitro anterior."
        )
    nuevo = destino.with_name(destino.name + ".nuevo-" + uuid.uuid4().hex)
    respaldo = destino.with_name(destino.name + ".anterior-" + uuid.uuid4().hex)
    try:
        shutil.copytree(origen, nuevo)
        inf = _leer_json(nuevo / "informe.json")
        for key, ruta in inf.get("cad", {}).get("archivos", {}).items():
            inf["cad"]["archivos"][key] = str(destino / "cad" / Path(ruta).name)
        inf.setdefault("trazabilidad", {})["config"] = str(config_path.resolve())
        (nuevo / "informe.json").write_text(
            json.dumps(inf, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if destino.exists():
            destino.replace(respaldo)
        nuevo.replace(destino)
        if respaldo.exists():
            shutil.rmtree(respaldo)
    except Exception:
        if respaldo.exists() and not destino.exists():
            respaldo.replace(destino)
        raise
    finally:
        if nuevo.exists():
            shutil.rmtree(nuevo)


def generar_ejemplo(carpeta, progreso=None, cancelar=None, semilla=0):
    return _motor(
        "ejemplo",
        {"carpeta": str(Path(carpeta).resolve()), "semilla": int(semilla)},
        progreso,
        cancelar,
    )


def analizar_proyecto(config_path, salida=None, progreso=None, cancelar=None):
    validar_proyecto(config_path)
    return _motor(
        "analizar",
        {
            "config_path": str(Path(config_path).resolve()),
            "salida": str(Path(salida).resolve()) if salida else None,
        },
        progreso,
        cancelar,
    )


def evaluar_gemelo(config, progreso=None, cancelar=None):
    return _motor("gemelo", {"config": config}, progreso, cancelar)


def exportar_resultados(
    informe_path,
    destino,
    formatos=("json", "npz", "png", "stl", "pdf"),
    progreso=None,
    cancelar=None,
):
    return _motor(
        "exportar",
        {
            "informe_path": str(Path(informe_path).resolve()),
            "destino": str(Path(destino).resolve()),
            "formatos": list(formatos),
        },
        progreso,
        cancelar,
    )


def generar_visualizacion(
    informe_path, tipo="mapa", destino=None, progreso=None, cancelar=None, **opciones
):
    return _motor(
        "visualizar",
        {
            "informe_path": str(Path(informe_path).resolve()),
            "tipo": tipo,
            "destino": str(Path(destino).resolve()) if destino else None,
            "opciones": opciones,
        },
        progreso,
        cancelar,
    )


def abrir_vista_3d(informe_path, **opciones):
    raiz = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(raiz)
    env.pop("MPLBACKEND", None)
    exe = os.environ.get("VITRO_PYTHON") or sys.executable
    comprobar = subprocess.run(
        [
            exe,
            "-c",
            "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('pyvista') else 2)",
        ],
        env=env,
        capture_output=True,
    )
    if comprobar.returncode:
        raise ErrorProyecto(
            "La vista 3D interactiva requiere PyVista/VTK en el intérprete de cálculo. Puede usar la captura estática local mientras instala ruedas compatibles."
        )
    return subprocess.Popen(
        [
            os.environ.get("VITRO_PYTHON") or sys.executable,
            "-m",
            "motor.servicio",
            "--vista3d",
            str(Path(informe_path).resolve()),
            "--opciones",
            json.dumps(opciones),
        ],
        env=env,
        cwd=raiz,
    )


def _evento(fraccion, texto):
    print(
        json.dumps({"progreso": float(fraccion), "mensaje": texto}, ensure_ascii=False), flush=True
    )


def _generar_vistas(p, informe, opciones=None, tipo=None, destino=None):
    from motor.salida.visualizacion import (
        cargar_mapas,
        crear_corte,
        crear_mapas,
        crear_vista_3d,
        figura_mapa,
    )

    mapas = cargar_mapas(p.parent / "mapas.npz")
    opciones = opciones or {}
    destino = (
        Path(destino)
        if destino
        else p.parent
        / (
            {"mapa": "mapas.png", "corte": "corte.png", "3d": "vista_3d.png"}.get(
                tipo, "vista_3d.png"
            )
        )
    )
    if tipo == "mapa":
        if opciones.get("mapa"):
            fig = figura_mapa(
                informe,
                mapas,
                mapa=opciones["mapa"],
                capa=opciones.get("capa", 1),
                interfaz=opciones.get("interfaz", 1),
            )
            fig.savefig(destino, dpi=130)
            from matplotlib import pyplot as plt

            plt.close(fig)
            resultado = destino
        else:
            resultado = crear_mapas(informe, mapas, destino)
    elif tipo == "corte":
        resultado = crear_corte(
            informe,
            mapas,
            destino,
            inicio=opciones.get("inicio"),
            fin=opciones.get("fin"),
            exageracion=(
                opciones.get("exageracion") if (opciones.get("exageracion") or 1) > 1 else None
            ),
        )
    else:
        permitidas = {"mapa", "capa", "interfaz", "exageracion", "explotada", "usar_pyvista"}
        resultado = crear_vista_3d(
            informe,
            mapas,
            destino,
            off_screen=True,
            **{k: v for k, v in opciones.items() if k in permitidas},
        )
    return str(resultado or destino)


def _trabajar(accion, s):
    if accion == "ejemplo":
        from motor.analisis.demo import guardar_escenario

        _evento(0.01, "Generando ejemplo sintético: no son mediciones reales")
        p = guardar_escenario(s["carpeta"], n_lado=9, semilla=s.get("semilla", 0))
        cfg = _leer_json(p)
        cfg["semilla"] = s.get("semilla", 0)
        Path(p).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        guardar_proyecto(s["carpeta"], {"config_path": str(Path(p).resolve())})
        return {"config_path": str(Path(p).resolve())}
    if accion == "analizar":
        from motor.analisis.pipeline import ejecutar_desde_config

        cp = Path(s["config_path"])
        c = _leer_json(cp)
        if s.get("salida_temporal"):
            c["salida"] = s["salida_temporal"]
        elif s.get("salida"):
            c["salida"] = s["salida"]
        tmp = None
        try:
            if s.get("salida") or s.get("salida_temporal"):
                f = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix=".vitro-",
                    dir=cp.parent,
                    delete=False,
                    encoding="utf-8",
                )
                json.dump(c, f, ensure_ascii=False)
                f.close()
                tmp = Path(f.name)
                cp_ejec = tmp
            else:
                cp_ejec = cp
            _evento(0.01, "Comprobando datos y calibración")
            _, inf = ejecutar_desde_config(str(cp_ejec), verbose=False, progreso=_evento)
            salida = Path(c.get("salida", "salida"))
            salida = salida if salida.is_absolute() else cp.parent / salida
            p = salida / "informe.json"
            _evento(0.92, "Generando vista 3D estática y corte")
            _generar_vistas(p, inf, tipo="3d")
            _generar_vistas(p, inf, tipo="corte")
            _evento(1, "Finalizado: consulte el veredicto y las condiciones")
            return cargar_resultado(p)
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)
    if accion == "gemelo":
        from motor.analisis.recetas import evaluar_config
        from motor.estado import ETIQUETA_HUD

        _evento(0.1, "Evaluando recetas; predicción exploratoria")
        informe = evaluar_config(s["config"])
        informe["etiqueta_HUD"] = ETIQUETA_HUD
        return informe
    if accion == "visualizar":
        p = Path(s["informe_path"])
        inf = cargar_resultado(p)["informe"]
        _evento(0.1, "Preparando visualización local")
        path = _generar_vistas(p, inf, s.get("opciones"), s.get("tipo"), s.get("destino"))
        return {"imagen_path": path}
    if accion == "exportar":
        from motor.estado import aplicar_estado

        p = Path(s["informe_path"])
        inf = aplicar_estado(_leer_json(p))
        d = Path(s["destino"])
        d.mkdir(parents=True, exist_ok=True)
        formatos = set(s["formatos"])
        permitidos = {"json", "npz", "png", "stl", "pdf"}
        if formatos - permitidos:
            raise ErrorProyecto("Formato de exportación desconocido.")
        _evento(0.1, "Exportando resultados con sus condiciones")
        if "json" in formatos:
            (d / "informe.json").write_text(
                json.dumps(inf, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        if "npz" in formatos:
            (
                shutil.copy2(p.parent / "mapas.npz", d / "mapas.npz")
                if (p.parent / "mapas.npz").resolve() != (d / "mapas.npz").resolve()
                else None
            )
        if "png" in formatos:
            for tipo in ("mapa", "3d", "corte"):
                _generar_vistas(
                    p,
                    inf,
                    tipo=tipo,
                    destino=d
                    / ({"mapa": "mapas.png", "3d": "vista_3d.png", "corte": "corte.png"}[tipo]),
                )
        if "stl" in formatos:
            from motor.geometria.exportar_cad import exportar_stl
            from motor.salida.visualizacion import cargar_mapas

            z = cargar_mapas(p.parent / "mapas.npz")
            exportar_stl(
                str(d / "cad"), z["interfaces"], list(inf["espesor"]), mascara=z["mascara_soporte"]
            )
            (d / "cad" / "UNIDADES.txt").write_text(
                "STL en µm. Al importar en mm aplicar escala 0.001.\n"
                + inf["condicion_CAD"]
                + "\nVeredicto: "
                + inf["veredicto"]
                + "\n",
                encoding="utf-8",
            )
        if "pdf" in formatos:
            from motor.salida.reporte_pdf import generar_pdf

            generar_pdf(inf, p.parent / "mapas.npz", d / "informe.pdf")
        hashes = {
            str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(d.rglob("*"))
            if f.is_file() and f.name != "manifest.json"
        }
        manifest = {
            "version": VERSION,
            "fecha_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "archivos_sha256": hashes,
            "veredicto": inf["veredicto"],
            "condicion_CAD": inf["condicion_CAD"],
            "origen": "local, sin envío de datos",
        }
        (d / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _evento(1, "Exportación terminada")
        return {"carpeta": str(d), "archivos": list(hashes), "manifest": str(d / "manifest.json")}
    raise ErrorProyecto("Operación desconocida.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Servicios locales de Vitro")
    parser.add_argument("--worker")
    parser.add_argument("--solicitud")
    parser.add_argument("--respuesta")
    parser.add_argument("--vista3d")
    parser.add_argument("--opciones", default="{}")
    a = parser.parse_args(argv)
    if a.vista3d:
        import importlib.util

        from motor.salida.visualizacion import cargar_mapas, crear_vista_3d

        if importlib.util.find_spec("pyvista") is None:
            print(
                "La vista interactiva requiere PyVista/VTK. Instale las dependencias GUI desde sus ruedas locales.",
                file=sys.stderr,
            )
            return 2
        p = Path(a.vista3d)
        plot = crear_vista_3d(
            cargar_resultado(p)["informe"],
            cargar_mapas(p.parent / "mapas.npz"),
            off_screen=False,
            **json.loads(a.opciones),
        )
        if hasattr(plot, "show"):
            plot.show()
        return 0
    if not (a.worker and a.solicitud and a.respuesta):
        parser.error("Se requiere worker, solicitud y respuesta.")
    try:
        r = _trabajar(a.worker, _leer_json(a.solicitud))
        code = 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        r = {"error": str(exc), "tipo": type(exc).__name__}
        code = 2
    Path(a.respuesta).write_text(
        json.dumps(r, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
