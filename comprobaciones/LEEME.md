# Comprobaciones

Salida de `python -m pytest -rs` sobre la versión 0.5.0, en Linux x86-64 con la interfaz en modo `offscreen` y gmsh instalado (`requirements-cad.txt`), de modo que también corre la prueba de lectura STEP.

| Python | Pruebas | Resultado |
| --- | --- | --- |
| 3.11.15 | 338 | todas pasan |
| 3.12.13 | 338 | todas pasan |
| 3.13.18 | 338 | todas pasan |

`entorno_y_suite.json` lista las versiones de las dependencias. El código pasa además `ruff check .` y `ruff format --check .` con la configuración de `pyproject.toml`. Estas pruebas usan datos simulados y no sustituyen la validación con mediciones reales (`documentacion/VALIDACION_REAL.md`).
No se ejecutaron en Windows ni con Python 3.14 o superior.
