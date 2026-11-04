# Formatos de entrada y contratos

La API anterior conserva argumentos existentes. Los callbacks nuevos son argumentos de palabra clave: `progreso(fraccion,texto)` y `cancelar()->bool`. La GUI usa `servicio.py`, que aísla el núcleo en un proceso y publica el análisis únicamente cuando termina.

## Configuración de diseño y máscara

`registro.modo=fijo|icp_forma`, `R,t`, espesores nominales, gradientes y unidades conservan su significado. `gradiente_nominal` sigue en µm/mm por capa, respecto de `origen_nominal_xy` en µm del marco CAD; no se estima desde la pieza.

```json
{
  "mascara_adquisicion": {
    "archivo": "mascara.npz",
    "clave": "mascara",
    "excluir": [[[ -5000,-3000 ],[ 5000,-3000 ],[ 5000,3000 ],[ -5000,3000 ]]]
  },
  "zonas_tolerancia": [
    {
      "nombre": "zona HUD de diseño",
      "poligono_xy_um": [[-20000,-15000],[20000,-15000],[20000,15000],[-20000,15000]],
      "tol_d": [10,15,10],
      "tol_w": 0.08,
      "tol_delta": [15,15,15,15]
    }
  ]
}
```

Es un fragmento para añadir a una configuración completa. La máscara puede usar únicamente polígonos `incluir`/`excluir`, únicamente `archivo`, o ambos. NPZ contiene `mascara (ny,nx)` booleana y opcionalmente `X,Y` en µm coincidentes con la rejilla de resultados. No hay remuestreo silencioso. Polígonos se expresan siempre en el marco CAD; la GUI convierte sus rectángulos en mm a esas coordenadas.

Las zonas sobrescriben tolerancias base en orden de lista; la última gana si se solapan. `tol_delta` por zona necesita tolerancia base por interfaz. Si un hueco atraviesa una celda sin excluir vértices, se excluyen sus nodos de forma conservadora para que el STL no lo puentee. Esta resolución puede eliminar un margen adicional: refinar rejilla cuando sea necesario.

La máscara limita dominio, exportación y calidad. No sustituye depurar los interferogramas: si se aportan observaciones dentro de regiones excluidas, el informe avisa que el ajuste espacial las recibió; para un hueco real de adquisición no deben aportarse señales inexistentes o ficticias.

## Señales reales

El formato uniforme `k` se conserva. Se admite `lambda_um` o k no uniforme únicamente si se habilita `remuestrear_espectro=true`; se advierte que la interpolación puede correlacionar ruido y no calibra el espectrómetro.

`instrumento.limites_saturacion=[min,max]` declara límites ADC medidos. No se infiere saturación de un valor máximo repetido sin conocer el instrumento. Los mensajes indican corregir exposición/ganancia si se exceden.

Datos `origen_datos=real_declarado` requieren `sigma_pos` explícita; no se asignan los3 µm de ejemplo a una medición real. Una pose real con covarianza cero requiere `registro.pose_exacta_declarada=true` y una `fuente` documental. Si se desconoce, omitir `covarianza` o usar null; no colocar ceros.

`rangos_ng_esperados` opcional contiene `[min,max]` por capa. No se fuerza1.5 como límite del PVB: ese intervalo debe confirmarse con la empresa y puede variar entre índice de fase y de grupo.

## Covarianza de normal y presupuesto

`mediciones.npz` puede incorporar `cov_normal (3,3)` común o `(M,3,3)` por punto, en coordenadas cartesianas de dirección y unidades adimensionales². Se proyecta a la tangente de la normal unitaria y se transforma al marco CAD. Se propaga como contribución geométrica conservadora que no se diluye al suavizar. La incertidumbre de normal CAD y la respuesta espectral oblicua siguen sin caracterizar.

El diagnóstico de incidencia informa ángulos y, si hay índices de fase aportados, una estimación geométrica del exceso de recorrido. No es una corrección de la interferometría de grupo. El sesgo/uncertidumbre experimental de espesor por oblicuidad permanece null.

`presupuesto_incertidumbre` puede conservar declaraciones adicionales por componente con `sigma`, `unidad` y `fuente`. La tabla indica qué contribución está propagada y cuál solo documentada; no suma valores con unidades/modelos incompatibles ni duplica referencia/índice. Incluir una cifra en esa tabla no demuestra haber completado el presupuesto experimental.

## Salidas y compatibilidad

- `veredicto`, `condicion_CAD`, `etiqueta_HUD` se centralizan; los archivos anteriores también se visualizan con las puertas actuales, sin heredar aprobaciones.
- `sigma_total_media_um` se conserva por compatibilidad pero significa **parcial modelada**, no incertidumbre experimental completa. Usar `sigma_parcial_conocida` y consultar `sigma_total_experimental=null`.
- Los campos de rejilla NPZ quedan `NaN` fuera de `mascara_soporte`, incluidas geometría, gradientes, tolerancias y covarianzas; no se deben usar esas posiciones como resultados. JSON usa null para desconocidos.
- La geometría nominal se incluye para exagerar solo las diferencias, preservando la curvatura global. La vista explotada es gráfica y no modifica STL.
- `software` y `trazabilidad.semilla` registran evidencia de ejecución; semilla ausente es null, nunca un número inventado.
- Exportar produce `manifest.json` con versión y SHA-256. Guardar el proyecto no altera el CAD ni envía archivos fuera del equipo.

Confocal: mantener `separar_confocal` como API experimental etiquetada. Para conectarla se necesita acordar formato, apertura, calibración, ruido y correlaciones. STEP: no implementado; mantener STL hasta una solicitud explícita del cliente con ensayos de ida/vuelta y volumen.
