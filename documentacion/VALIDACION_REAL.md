# Protocolo propuesto para validar con señales reales

Estado al entregar esta revisión: **no ejecutado**. No se recibieron interferogramas reales, certificados de espesor, adquisición de una segunda muestra física, datums del montaje ni geometría del HUD de Vitro. Un ensayo simulado no sustituye estos datos.

## 1. Congelar lo que se evalúa

Registrar versión del programa, configuración, equipo/serie, espectrómetro, eje espectral calibrado, temperatura, fecha, operador, procesamiento previo y tolerancias acordadas. Conservar señales crudas, oscuridad, fondo, fuente y archivos de referencia sin sobrescribirlos.

Definir el CAD nominal y el marco de datums antes de medir. Obtener `R`, `t` y covarianza desde referencias independientes de las superficies que se inspeccionan. Si solo existe ICP, declarar evaluación de forma relativa; no interpretar como error absoluto de posición o inclinación. No introducir covarianza cero por falta de información.

Precisar qué espesor se compara: distancia a lo largo del rayo casi normal, espesor normal a la superficie o distancia según el dibujo. En una zona muy inclinada estos mensurandos difieren.

## 2. Referencia para separar índice y espesor

Identificar físicamente muestra A, materiales/lotes, historia de laminación y espesores **después del proceso**. No usar el espesor nominal de receta como verdad. Obtener espesores independientes con incertidumbre estándar trazable, incluyendo correlaciones de medición (`cov_d_ref` si existen).

Como piloto propuesto, adquirir al menos treinta repeticiones en el mismo punto de A; el mínimo de cinco del software es solo una restricción operativa. Analizar estabilidad temporal, saturación, ruido espectral, picos secundarios, contraste de cada interfaz, espesor aparente y covarianza de los caminos vecinos.

Medir además variación al repetir montaje, operador, día y temperatura. Una desviación estándar de repeticiones sin desmontar caracteriza repetibilidad de corto plazo; no incluye todos esos efectos. Si hay deriva, no tratar toda la señal como ruido blanco independiente.

Comprobar la compatibilidad de las dos capas de vidrio antes de agruparlas. Si lotes/composición/temperatura son distintos, usar grupos separados; no forzar un único índice para mejorar artificialmente la precisión.

## 3. Segunda muestra de validación

Usar muestra B, identificador y señales distintos, sin reajustar la calibración obtenida con A. Elegir espesores suficientemente diferentes para probar transferencia, dentro del rango del equipo, y materiales representativos del uso previsto. Tener metrología independiente con incertidumbre, en varios puntos o zonas, no solo una media nominal.

Reportar por capa `d_estimado`, `d_ref2`, error, incertidumbre combinada y error normalizado z. El programa usa `|z|<3` por capa como contraste normal aproximado. Una muestra compatible no demuestra cobertura universal ni descarta un error de modelo compensado.

Si falla: conservar el resultado fallido, revisar eje espectral, ventanas, SNR, grupos, incidencia, temperatura, absorción y referencia; no añadir un factor de corrección usando B y seguir llamándola muestra de validación. Si B se usa para ajustar, reservar una tercera muestra independiente para probar el ajuste.

## 4. Validar reconstrucción espacial y CAD

Medir una malla de puntos con cobertura real registrada, conservar máscara de huecos y no aceptar extrapolación como evidencia de calidad. Comparar forma de las interfaces con metrología geométrica independiente cuando sea accesible. Separar incertidumbre del montaje, posición y orientación de sonda, calibración óptica y ajuste espacial.

El programa acepta una máscara explícita NPZ o polígonos y huecos rectangulares desde la GUI. La envolvente convexa sola no identifica huecos. Verificar máscara, unidades y resolución; los huecos subrejilla se excluyen conservadoramente y pueden eliminar un margen adicional. No aportar señales ficticias dentro de huecos reales.

Revisar sesgo, RMS, máximo absoluto, residuos espaciales, diferencias entre interfaces y conservación de espesores en el STL. Evaluar sesgo de la interpolación/triangulación del CAD: un CAD discretizado tiene error geométrico propio. Aumentar densidad de triangulación si es necesario, sin ajustar el nominal a la pieza medida.

La dirección de sonda debe ser casi normal. El programa cuantifica un diagnóstico geométrico y acepta covarianza de normal para propagación hacia geometría CAD; no corrige la respuesta óptica oblicua ni cuantifica su sesgo de espesor. Caracterizar el ángulo, la normal CAD y su efecto antes de aceptar incertidumbres submicrométricas. La cuña reportada es una derivada proyectada en el plano CAD, con aproximación de ángulo pequeño.

## 5. Validar incertidumbre y regla de calidad

Construir presupuesto que incluya referencia, grupo óptico, eje espectral, offset, deriva, posición/orientación, montaje, repetibilidad/reproducibilidad, interpolación y discrepancia del modelo. Las correlaciones comunes no desaparecen al promediar puntos. Este tratamiento sigue el principio de propagación con covarianzas de [JCGM 100:2008, §5.2](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).

Examinar coberturas empíricas de intervalos en muestras reservadas y reportar el número efectivo de ensayos, dependencia espacial y limitación del tamaño muestral. No estimar cobertura de población contando miles de píxeles correlacionados de una misma pieza. Para parámetros GP, considerar validación espacial reservada y sensibilidad a hiperparámetros.

Acordar con Vitro si se controla riesgo por punto, por zona o por pieza y los costos de aceptación/rechazo erróneos. La fracción de nodos con `P_fuera>0.05` no es una probabilidad global de fallo de la pieza. La incertidumbre y la regla de aceptación deben tratarse conjuntamente, conforme al marco de [JCGM 106:2012](https://www.bipm.org/en/doi/10.59161/jcgm106-2012); la implementación actual no constituye certificación conforme a esa guía.

## 6. Validar recetas y HUD

Medir ruido/covarianza, resolución axial, rango sin alias y amplitud mínima para todas las interfaces previstas. La sensibilidad paramétrica local no basta si los picos se fusionan o desaparecen por falta de contraste/absorción. Para un cambio mínimo detectable operativo, fijar además probabilidades de falsa alarma y detección y contrastarlas con cambios físicos o patrones conocidos.

Para HUD, obtener índice de fase y dispersión visible de cada material, geometría real fuente–parabrisas–ojo, dirección de cuña, longitud de onda, curvatura y polarización. Calibrar n_g en infrarrojo no determina esos índices de fase visibles. Comparar separación angular predicha con imágenes reales en varias posiciones del ojo; registrar intensidad del fantasma si es una exigencia de calidad.

La API propaga covarianza paramétrica si se suministra, pero no añade automáticamente discrepancia del trazador 2D respecto al parabrisas real. Hasta completar esa comparación, informar separación HUD como exploratoria con incertidumbre total desconocida.

## Evidencia mínima para cerrar el objetivo

- Señales reales crudas y procesadas con trazabilidad, ruido, resolución y rango caracterizados.
- Referencia A con espesores independientes, incertidumbres y covarianzas justificadas.
- Validación reservada B compatible, sin usarla para recalibrar, con errores por capa y repetibilidad entre montajes.
- Marco CAD independiente y presupuesto de incertidumbre geométrica, incluida dirección de sonda.
- Cobertura y regla de decisión acordadas y verificadas para el uso previsto.
- Índices visibles/geometría HUD y comparación óptica si se pretende tomar decisiones sobre imagen doble.

Entregable experimental esperado: informe por pieza y por muestra de referencia, valores e incertidumbres con unidades, errores contra la referencia, fallos sin ocultar, archivos STL con escala declarada y dictamen limitado al dominio realmente validado.
