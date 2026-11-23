# Instalación

Necesita **Python 3.11, 3.12 o 3.13** (python.org/downloads). Con versiones más nuevas (por ejemplo 3.15) pueden faltar paquetes, en particular VTK para macOS, que usa la vista 3D interactiva.

## Con el lanzador

- **macOS:** doble clic en `iniciar.command`. Si el sistema no lo abre, ejecute una vez `chmod +x iniciar.command` en Terminal.
- **Windows:** doble clic en `iniciar.bat`.

El lanzador crea `.venv`, instala `requirements.txt` la primera vez y abre la interfaz.

## A mano

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

En macOS, si `python3.12` no existe, use la ruta completa del instalador de python.org: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`.

## Sin red

En un equipo con internet y el mismo sistema y versión de Python:

```bash
python -m pip download -r requirements.txt -d ruedas
```

Copie `ruedas/` al equipo sin red y ejecute `python -m pip install --no-index --find-links ruedas -r requirements.txt`.

## Lectura de CAD STEP (opcional)

`python main.py importar-cad pieza.step` necesita el paquete `gmsh`, que incluye OpenCASCADE:

```bash
python -m pip install -r requirements-cad.txt
```

En Linux, gmsh necesita también la biblioteca del sistema libGLU (`sudo apt install libglu1-mesa`). Sin gmsh, el importador acepta un STL cerrado por capa (`importar-cad capa1.stl capa2.stl capa3.stl --unidades mm`).

## Sin ventana

El núcleo y el PDF no necesitan Qt ni VTK. Se usan desde la línea de comandos con `python main.py demo`, `ejemplo`, `analizar`, `receta`, `pdf`, `importar-cad` y `ejemplo-cad` (`python main.py --help` las lista).
