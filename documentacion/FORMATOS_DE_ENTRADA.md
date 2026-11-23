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

Confocal: mantener `separar_confocal` como API experimental etiquetada. Para conectarla se necesita acordar formato, apertura, calibración, ruido y correlaciones. STEP: lectura con `importar-cad` (sección siguiente); la salida sigue en STL.

## CAD importado (STEP o STL por capa)

```bash
python main.py importar-cad pieza.step --salida modelo_cad
python main.py importar-cad capa_a.stl capa_b.stl capa_c.stl --unidades mm --zona HUD=hud.stl --salida modelo_cad
```

El importador no usa nombres, armadora, ejes de vehículo ni número de capas fijo:

1. **Capas por geometría.** Un sólido es candidato a capa si es laminar (espesor equivalente 2V/A menor a 25 mm y a 5 % de su tamaño). Las capas son los laminares que se tocan cubriendo al menos 60 % de la cara de su vecina (`cobertura_min`) y forman una pila simple. Cintas, perfiles, soportes o películas parciales se reportan como ignorados.
2. **Lado de la sonda.** `--lado-sonda convexo` (por defecto), `concavo` o el nombre de una capa extrema. La interfaz 0 es la cara de ese lado.
3. **Marco local.** Por defecto se arma un marco donde la pieza es un campo de altura z(x,y), con z hacia la sonda (`P_local = R·P_cad + t`, guardado en `cad_importado.json`). `--marco original` lo evita si el CAD ya cumple esa condición.
4. **Espesor nominal por nodo.** Cada interfaz se representa por mínimos cuadrados cuadráticos locales sobre los vértices del teselado (que están sobre la superficie exacta), y el espesor de cada capa se mide a lo largo de la normal de la cara frontal, igual que en el motor.
5. **Zonas.** Las superficies auxiliares que están sobre el vidrio (a menos de 1 mm de una interfaz) se guardan con su nombre, la interfaz donde están y sus lazos en el marco local. Una región con huecos se evalúa con la regla par-impar.
6. **Diagnóstico.** Radio de curvatura mínimo, validez del modelo local plano, inclinación máxima, cuña por capa, separación mínima de picos, error de cuerda del STL frontal y avisos.

Salidas en la carpeta: `cad_frontal.stl` (µm, marco local), `nominal.npz` (`X`, `Y`, `z0`, `d_nom (capas,ny,nx)`, `mascara`), `cad_importado.json`, `config_fragmento.json` y `vista_cad.png`. Los nombres que vienen del STEP se asignan por orden de archivo y son solo etiquetas.

Claves de `config.json` que usan el modelo:

```json
{
  "cad": "modelo_cad/cad_frontal.stl",
  "cad_importado": "modelo_cad/cad_importado.json",
  "capas": {"d_nom_mapa": "modelo_cad/nominal.npz", "idx_pvb": 1, "...": "..."},
  "cuna": {"w_nom": "cad", "tol": 0.1},
  "direccion_cuna": "cad",
  "mascara_adquisicion": {"excluir_zonas_cad": ["serigrafia_negra"]},
  "zonas_tolerancia": [{"zona_cad": "campo_HUD", "tol_w": 0.05}]
}
```

- Con `cad_importado`, `registro.R,t` sigue siendo la pose del sensor en el marco **original** del CAD; el motor compone el marco local y transporta la covarianza. `marco_cad: {R,t}` permite declararlo sin el JSON del importador.
- `d_nom_mapa` reemplaza a `gradiente_nominal` (no se aceptan ambos). Si falta `d_nom`, se usa la media del mapa. Las ventanas de picos de cada adquisición se centran en el nominal local de su posición.
- `cuna.w_nom = "cad"` compara la cuña medida contra el mapa de cuña nominal; `direccion_cuna = "cad"` toma la dirección media del gradiente nominal de la capa `idx_pvb`.
- `zona_cad` y `excluir_zonas_cad`/`incluir_zonas_cad` usan los nombres de `cad_importado.json`; un nombre inexistente es un error. Las coordenadas de polígonos escritos a mano quedan en el marco local.

`python main.py ejemplo-cad modelo_cad/cad_importado.json pieza --variante defecto` genera mediciones sintéticas sobre esa geometría (con `--zona` para restringirlas) y un `config.json` listo para `analizar`; `verdad.npz` guarda la verdad usada.
