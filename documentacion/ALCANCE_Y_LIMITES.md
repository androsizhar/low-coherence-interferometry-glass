# Alcance y límites

**Estado:** prototipo de investigación endurecido, listo para iniciar validación experimental. No se dispone de mediciones reales ni de validación con una segunda muestra física.

## Alcance implementado

- GUI local de seis pestañas, texto en español, tema claro/oscuro, rutas, proyecto restaurable, selección de unidades, pose, máscaras y tolerancias; tablas de calibración A y contraste B, receta editable catálogo/Cauchy y exportación.
- Hilo Qt y proceso de cálculo separado; progreso y cancelación. Un análisis se publica al terminar; si se cancela conserva el resultado anterior. No hay servidor ni llamadas de red en los módulos del proyecto.
- Etiquetas centrales en `estado.py`: no concluyente, condición de pose junto a desviaciones y HUD como predicción exploratoria. La compatibilidad de A/B nunca convierte por sí sola un resultado en aprobación.
- Máscara explícita NPZ o polígonos, huecos rectangulares desde la GUI, respeto de topología y exclusión conservadora de huecos subrejilla; valores NPZ fuera del soporte son NaN, JSON desconocido es null, STL no rellena esos dominios.
- Nominal variable de diseño conservado, tolerancias por zona y precedencia explícita, mapas divergentes de cuña/desviación centrados en cero; incertidumbre y espesor secuenciales.
- Vista3D opcional PyVista y alternativa Matplotlib rotulada como estática, selector de escalar y unidades, geometría global nominal conservada al exagerar diferencias, vista explotada rotulada; captura PNG.
- Corte de espesores exacto respecto de la interpolación de d_media, sin unir huecos, detalle PVB y fallas puntuales; escala vertical rotulada.
- PDF A4 horizontal de cinco páginas, generable por CLI sin interfaz, con incertidumbre parcial, condiciones, mapas,3D,corte,presupuesto y método/límites. Se renderizó con Poppler y se revisó visualmente.
- Incidencia y covarianza de normal: diagnóstico geométrico cuantificado, rechazo de orientación saliente y propagación de primer orden hacia desviación CAD; no se inventa una corrección de espesor oblicuo.
- Robustez de entrada: remuestreo espectral solo explícito, límites ADC declarados para saturación, unidades, mensajes de qué corregir, documentación del ruido correlacionado no caracterizado.
- GP por lotes para rejillas de10,000 nodos; comprobaciones cronometradas y evidencia en JSON.

## Invariantes metrológicas conservadas

1. Las interfaces salen de frontal+espesores. En cada vértice correspondiente, la separación conserva d_media; no se regresó a cuatro superficies independientes incompatibles.
2. El nominal proviene del diseño. Un ajuste ICP solo describe forma relativa; no declara desviación absoluta.
3. Se mantiene covarianza entre espesores vecinos y entre índices calibrados compartidos. Las contribuciones comunes no disminuyen al suavizar más puntos.
4. La dirección de cuña es física y fija. Invertir el orden de capas para el trazador HUD no invierte por sí solo su signo.
5. Δ=3 no implica certeza; se presenta la hipótesis estadística y la ambigüedad n–d. Un error ideal que desborda la precisión numérica se etiqueta así en GUI, sin mostrarlo como certeza.
6. La desviación de una demo con pose exacta se identifica como escenario ideal y nunca se usa como precisión esperada del montaje real.

## Lo que se verificó y lo que no

`comprobaciones/` contiene las salidas de la suite (308 pruebas en Python 3.11, 3.12 y 3.13 sobre Linux) y las versiones usadas. La interfaz PySide6 y la ventana 3D VTK se ejecutaron de verdad en ese entorno; `documentacion/capturas/` muestra el resultado.

**No se probó** en Windows ni con Python 3.14 o superior. En macOS (Python 3.12.10) la instalación y el zoom de la ventana 3D se comprobaron a mano, no con la suite. Los datos son simulados: la validación experimental sigue pendiente.

## Presupuesto: cobertura completa, cuantificación experimental incompleta

La tabla enumera referencia, índice de grupo, eje espectral, offset, deriva, montaje, normal, repetibilidad, interpolación y modelo. Las contribuciones ya propagadas se identifican, y la referencia no se suma otra vez después del índice. Las faltantes se mantienen null. Una declaración adicional se conserva con fuente y estado de integración, sin sumarla automáticamente a una magnitud cuyo modelo/unidad no se haya definido.

Las bandas siguen siendo parciales y condicionadas al GP, al modelo óptico y a la pose aportada. La incertidumbre experimental total continúa no disponible. La covarianza de normal no calibra el sesgo de espesor por incidencia ni la incertidumbre de la normal CAD. El remuestreo no calibra el eje espectral ni estima automáticamente ruido correlacionado. La exportación con perfiles exactos no certifica ausencia global de autointersecciones de superficies complejas.

## Decisiones abiertas y límites conservados

- **Confocal:** API experimental etiquetada y fuera de la GUI. El equipo debe acordar formato y metrología antes de conectarla.
- **STEP:** no implementado. STL satisface el alcance actual; una futura exigencia requiere prueba de ida/vuelta y volumen.
- **HUD:** trazador local2D de tres capas y dos reflexiones principales; predicción exploratoria hasta contrastar índices de fase visibles, geometría y fotografías reales. No se incluyeron intensidad/polarización ni curvatura global del sistema real.
- **Máscara:** el soporte es a resolución de rejilla y la exclusión subrejilla es conservadora. El ajuste recibe las observaciones válidas suministradas; si se incluyen observaciones dentro de una zona excluida, el informe lo avisa. No inventar señales dentro de huecos reales.
- **Datos físicos:** absorción/recubrimientos, temperatura, datums, eje espectral, offset, ruido, rango/resolución y muestra B siguen pendientes del equipo y de Vitro.

El objetivo de esta iteración es preparar una herramienta utilizable y transparente para recibir esos datos; no demostrar una metrología de producción mediante software o simulación.
