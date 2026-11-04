# Comprobaciones

Salida de `python -m pytest -q -rs verificacion` sobre la versión 0.4.1, en Linux x86-64 con la interfaz en modo `offscreen`.

| Python | Pruebas | Resultado |
| --- | --- | --- |
| 3.11.18 | 323 | todas pasan |
| 3.12.3 | 323 | todas pasan |
| 3.13.18 | 323 | todas pasan |

`entorno_y_suite.json` lista las versiones de las dependencias. Estas pruebas usan datos simulados y no sustituyen la validación con mediciones reales (`documentacion/VALIDACION_REAL.md`).
No se ejecutaron en Windows ni con Python 3.14 o superior.
