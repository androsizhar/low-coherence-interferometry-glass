# Vitro · Gemelo digital de vidrio laminado

Reconstruye de forma no destructiva el espesor y el índice de refracción de cada capa de un vidrio laminado (vidrio / PVB / vidrio) a partir de interferometría de baja coherencia, genera la geometría 3D de cada interfaz y la compara con el CAD nominal. Permite evaluar recetas de materiales y espesores sin fabricarlas.

## Inicio

Doble clic en `iniciar.command` (macOS) o `iniciar.bat` (Windows). Requiere Python 3.11 a 3.13; el detalle está en `documentacion/INSTALACION.md`.

## Uso

En la pestaña **Proyecto**, elija una carpeta, pulse **Generar ejemplo** y luego **Analizar proyecto**.

| Pestaña | Contenido |
| --- | --- |
| Proyecto | Archivos de medición, calibración, CAD, pose y máscara |
| Calibración | Índice de grupo con la muestra A y contraste con la muestra B |
| Resultados | Mapas de espesor, desviación contra CAD, cuña del PVB, vista 3D y corte |
| Calidad | Tolerancias y fracción de nodos fuera de tolerancia |
| Gemelo | Recetas virtuales: distinguibilidad, sensibilidad e imagen doble del HUD |
| Exportar | STL (µm), PDF, JSON, NPZ y PNG con manifiesto SHA-256 |

El STL no guarda unidades: está en µm, así que al importarlo en milímetros se aplica el factor 0.001.

## Línea de comandos

```bash
python main.py demo --salida salidas/demo
python main.py ejemplo datos/pieza_ok --variante conforme
python main.py ejemplo datos/pieza_defecto --variante defecto
python main.py analizar config.json
python main.py receta ejemplos/recetas.json --salida salidas/recetas.json
python main.py pdf informe.json mapas.npz informe.pdf
```

## Estructura

```
main.py            punto de entrada
motor/
  medicion/        lectura de señales, picos, modelo óptico y simulador
  metrologia/      calibración, espesor y diagnósticos
  geometria/       CAD, superficies, soporte y exportación STL
  analisis/        pipeline, calidad, gemelo, recetas y escenario de demostración
  salida/          figuras y reporte PDF
  interfaz/        ventana y pestañas (Qt)
ejemplos/          receta de ejemplo, resultados de demostración y dos piezas simuladas (casos/)
verificacion/      pruebas automáticas
comprobaciones/    resultados de las pruebas por versión de Python
documentacion/     instalación, formatos de entrada, alcance y protocolo de validación
```

## Piezas simuladas para presentar

`ejemplos/casos/conforme` y `ejemplos/casos/defecto` contienen el resultado ya analizado de dos piezas sintéticas: una dentro de tolerancia y otra con una burbuja en el vidrio exterior, una depresión en el PVB y una cuña fuera de especificación. Se abren en la ventana con **Abrir resultado** (`informe.json`) y cada carpeta incluye PDF, mapas, vista 3D, corte y STL. Los datos crudos (~38 MB por pieza) se regeneran con `python main.py ejemplo` y luego `python main.py analizar CARPETA/config.json`.

## Pruebas

```bash
python -m pytest -q verificacion
```

## Estado

Todos los datos incluidos son simulados. Hasta validar con mediciones reales (`documentacion/VALIDACION_REAL.md`) el resultado se declara **no concluyente**, y la desviación contra el CAD debe leerse junto con la condición de pose que informa el programa.
