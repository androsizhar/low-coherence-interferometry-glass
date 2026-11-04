from __future__ import annotations

import argparse
import json
import tempfile
from html import escape
from pathlib import Path


def _numero(valor, sufijo="", decimales=5):
    import math

    if valor is None:
        return "no disponible"
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return "no disponible"
    return f"{valor:.{decimales}g}{sufijo}" if math.isfinite(valor) else "no disponible"


def _vector_numerico(valor):
    if hasattr(valor, "tolist"):
        valor = valor.tolist()
    if isinstance(valor, (list, tuple)):
        return "[" + ", ".join(_vector_numerico(v) for v in valor) + "]"
    return _numero(valor)


def generar_pdf(informe, mapas, destino, inicio_corte=None, fin_corte=None, exageracion=1):
    if isinstance(informe, (str, Path)):
        informe = json.loads(Path(informe).read_text(encoding="utf-8"))
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:
        raise RuntimeError(
            "Para exportar PDF instala reportlab (requirements.txt); no se requiere interfaz gráfica"
        ) from error
    from matplotlib.font_manager import findfont

    from motor.estado import ETIQUETA_HUD, condicion_pose, veredicto
    from motor.salida.visualizacion import crear_corte, crear_mapas, crear_vista_3d

    if "VitroSans" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("VitroSans", findfont("DejaVu Sans")))
        pdfmetrics.registerFont(TTFont("VitroSansBold", findfont("DejaVu Sans:bold")))
    estilo = ParagraphStyle(
        "Vitro",
        fontName="VitroSans",
        fontSize=8.3,
        leading=11.3,
        textColor=colors.HexColor("#183044"),
        alignment=TA_LEFT,
    )
    titulo = ParagraphStyle(
        "VitroTitulo",
        parent=estilo,
        fontName="VitroSansBold",
        fontSize=17,
        leading=21,
        spaceAfter=12,
    )
    subtitulo = ParagraphStyle(
        "VitroSubtitulo",
        parent=estilo,
        fontName="VitroSansBold",
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )
    pequeno = ParagraphStyle("VitroPequeno", parent=estilo, fontSize=7.1, leading=9.4)

    def p(texto, sty=estilo):
        return Paragraph(escape(str(texto)), sty)

    def tabla(filas, anchos):
        contenido = [[p(c, pequeno) for c in fila] for fila in filas]
        t = Table(contenido, colWidths=anchos, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d2d9")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return t

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    ancho, alto = landscape(A4)
    util = ancho - 64
    estado, pose = veredicto(informe), condicion_pose(informe)
    texto_pie = p(
        pose + ". HUD: " + ETIQUETA_HUD + ". σ parcial modelada; total experimental no disponible.",
        pequeno,
    )
    _, alto_pie = texto_pie.wrap(util, 200)
    tope_pie = max(68, alto_pie + 38)
    alto_contenido = alto - 45 - max(80, tope_pie + 12) - 12
    ancho_mapa = min(util * 0.91, max(120, alto_contenido - 64) * 16 / 9)

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#183044"))
        canvas.setFont("VitroSansBold", 8)
        canvas.drawString(32, alto - 25, "VITRO | Prototipo de investigación")
        canvas.drawRightString(ancho - 32, alto - 25, f"{estado} | {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#c5d2d9"))
        canvas.line(32, tope_pie, ancho - 32, tope_pie)
        texto_pie.drawOn(canvas, 32, tope_pie - 9 - alto_pie)
        canvas.setFont("VitroSans", 6.8)
        canvas.drawString(
            32,
            16,
            "100 % local | Sin validación experimental demostrada por esta exportación | STL en µm: dividir por 1000 para mm",
        )
        canvas.restoreState()

    with tempfile.TemporaryDirectory(prefix="vitro_pdf_") as tmp:
        tmp = Path(tmp)
        imagen_mapas = crear_mapas(informe, mapas, tmp / "mapas.png")
        imagen_3d = crear_vista_3d(
            informe, mapas, tmp / "vista_3d.png", usar_pyvista=False, exageracion=exageracion
        )
        imagen_corte = crear_corte(
            informe, mapas, tmp / "corte.png", inicio=inicio_corte, fin=fin_corte
        )
        historia = [
            p("Resultados de investigación", titulo),
            p("Veredicto: " + estado, subtitulo),
            p(pose),
            Spacer(1, 8),
        ]
        origen = informe.get("estado_validacion", {}).get("origen_datos", "no declarado")
        historia += [
            p(
                f"Origen de datos: {origen}. La compatibilidad de una segunda muestra y las pruebas de software no equivalen a validación experimental completa."
            ),
            Spacer(1, 10),
        ]
        filas = [
            [
                "Capa",
                "Espesor medio (µm)",
                "σ aleatoria (µm)",
                "σ sistemática (µm)",
                "Zona fuera de tolerancia (%)",
            ]
        ]
        for nombre, valores in informe.get("espesor", {}).items():
            fraccion = valores.get("fraccion_zona_fuera_de_tolerancia")
            filas.append(
                [
                    nombre,
                    _numero(valores.get("media_um")),
                    _numero(valores.get("sigma_aleatoria_um")),
                    _numero(valores.get("sigma_sistematica_um")),
                    _numero(None if fraccion is None else 100 * fraccion),
                ]
            )
        historia += [
            tabla(filas, [util * 0.22, util * 0.20, util * 0.18, util * 0.18, util * 0.22]),
            Spacer(1, 9),
        ]
        w = informe.get("cuna_pvb_mrad", {})
        fw = w.get("fraccion_zona_fuera_de_tolerancia")
        historia += [
            p(
                "Cuña PVB con signo: "
                + _numero(w.get("media"))
                + " ± "
                + _numero(w.get("sigma_media"))
                + " mrad (1 σ). Eje fuente→ojo: "
                + str(w.get("direccion", "no disponible"))
                + ". Zona fuera de tolerancia: "
                + _numero(None if fw is None else 100 * fw)
                + " %."
            ),
            Spacer(1, 6),
        ]
        gemelo = informe.get("gemelo", {})
        historia += [
            p(
                "HUD - "
                + ETIQUETA_HUD
                + ": imagen doble "
                + _numero(gemelo.get("imagen_doble_arcmin"))
                + " minutos de arco; incertidumbre total "
                + _numero(gemelo.get("incertidumbre_total_hud_arcmin"))
                + ". No contrastada con imágenes medidas."
            ),
            Spacer(1, 8),
        ]
        avisos = list(informe.get("banderas", []))
        avisos += list(informe.get("calibracion", {}).get("banderas", []))
        avisos += list(informe.get("registro", {}).get("banderas", []))
        historia += [
            p("Avisos y condiciones", subtitulo),
            p(
                (
                    "; ".join(map(str, avisos))
                    if avisos
                    else "Sin banderas registradas; permanecen los límites del método y los requisitos experimentales."
                ),
                pequeno,
            ),
            Spacer(1, 5),
            p(
                informe.get("regla_calidad", {}).get(
                    "interpretacion",
                    "Probabilidades por punto condicionadas al modelo; no expresan el riesgo global de la pieza.",
                ),
                pequeno,
            ),
        ]
        historia += [
            PageBreak(),
            p("Mapas y soporte de adquisición", titulo),
            p(
                "Cada barra conserva todo el rango observado. Desviación y cuña: escala simétrica en cero; zonas sin soporte: vacías.",
                pequeno,
            ),
            Spacer(1, 6),
            Image(str(imagen_mapas), width=ancho_mapa, height=ancho_mapa * 9 / 16),
        ]
        historia += [
            PageBreak(),
            p("Geometría y corte transversal", titulo),
            p(
                "Captura 3D estática con Matplotlib, sin requerir PyVista/VTK. La vista interactiva usa PyVista cuando está instalado. La separación de interfaces conserva unidades físicas.",
                pequeno,
            ),
            Spacer(1, 4),
        ]
        historia += [
            Table(
                [
                    [
                        Image(str(imagen_3d), width=util * 0.50, height=util * 0.50 * 8 / 12),
                        Image(str(imagen_corte), width=util * 0.50, height=util * 0.50 * 8 / 12),
                    ]
                ],
                colWidths=[util * 0.5] * 2,
            ),
            Spacer(1, 7),
        ]
        historia += [
            p(
                "El corte aplanado elimina la curvatura del frente para hacer visibles los espesores. La diferencia entre sus perfiles coincide con d_media interpolado; los huecos interrumpen el perfil. Las cruces indican P puntual por encima del umbral, no una decisión de liberación.",
                pequeno,
            ),
            Spacer(1, 8),
        ]
        filas_delta = [
            ["Interfaz", "Media vs CAD (µm)", "RMS vs CAD (µm)", "σ parcial modelada (µm)"]
        ]
        for nombre, valores in informe.get("desviacion_vs_cad_um", {}).items():
            filas_delta.append(
                [
                    nombre,
                    _numero(valores.get("media")),
                    _numero(valores.get("rms")),
                    _numero(
                        valores.get(
                            "sigma_parcial_conocida_media", valores.get("sigma_total_media")
                        )
                    ),
                ]
            )
        historia += [
            tabla(filas_delta, [util * 0.25] * 4),
            p(
                "Condición de pose completa en el pie de esta página. Ningún valor sintético es precisión esperada de un montaje real.",
                pequeno,
            ),
        ]
        historia += [
            PageBreak(),
            p("Presupuesto de incertidumbre", titulo),
            p(
                "Contribuciones comunes conservadas entre puntos; una incertidumbre desconocida permanece no disponible. No se suman arbitrariamente componentes ya contenidos en otra estimación.",
                pequeno,
            ),
            Spacer(1, 8),
        ]
        presupuesto = informe.get("presupuesto_incertidumbre", [])
        if isinstance(presupuesto, dict):
            presupuesto = [
                {"componente": k, **(v if isinstance(v, dict) else {"descripcion": v})}
                for k, v in presupuesto.items()
            ]
        filas_pres = [["Componente", "Estado / tratamiento", "Incertidumbre y límite"]]
        for item in presupuesto:
            nombre = item.get("componente", item.get("fuente", "componente"))
            estado_item = item.get("estado", item.get("tratamiento", "no disponible"))
            detalle = item.get(
                "detalle", item.get("descripcion", item.get("limite", item.get("nota", "")))
            )
            valor = item.get(
                "sigma", item.get("sigma_um", item.get("valor", item.get("incertidumbre")))
            )
            unidad = item.get("unidad", "µm")
            texto = (
                _vector_numerico(valor) + " " + unidad + "; "
                if valor is not None
                else "σ escalar no disponible; "
            ) + str(detalle)
            if item.get("covarianza") is not None:
                texto += "; covarianza: " + _vector_numerico(item["covarianza"])
            if item.get("declaracion_adicional"):
                texto += "; evidencia adicional no integrada al cálculo: " + str(
                    item["declaracion_adicional"]
                )
            filas_pres.append([nombre, estado_item, texto])
        if len(filas_pres) == 1:
            filas_pres.append(
                [
                    "Presupuesto completo",
                    "No disponible",
                    "No se inventan contribuciones ni se reemplazan por cero.",
                ]
            )
        historia += [tabla(filas_pres, [util * 0.20, util * 0.25, util * 0.55]), Spacer(1, 8)]
        historia += [PageBreak(), p("Método y límites", titulo)]
        metodo = [
            "Espesor = camino óptico / índice de grupo calibrado con muestra A. La muestra B debe ser distinta y no recalibrarse para el contraste. El índice calibrado es común a la pieza y los espesores vecinos comparten el error de un pico.",
            "Interfaces construidas desde frente + espesores, coherentes por vértice. La máscara de adquisición excluye huecos de mapas, decisiones y STL. Interpolación condicionada al modelo y al soporte; no demuestra información en zonas no medidas.",
            "CAD absoluto requiere datums independientes y covarianza de pose declarada. Mejor ajuste informa solo forma relativa. La pose sintética exacta es un supuesto idealizado y no una caracterización del equipo real.",
            "Distinguibilidad y mínimo detectable dependen del modelo y del ruido aportado. Δ=3 implica aproximadamente 6.7 % de error de clasificación en hipótesis gaussianas simples; no garantiza potencia de detección. Ambigüedad n–d: sin referencia no se separan índice y espesor.",
            "HUD: "
            + ETIQUETA_HUD
            + ". Faltan geometría óptica real, índices de fase visibles, imágenes medidas y presupuesto completo. Confocal no integrado; STEP no implementado. Exportación STL en µm: al importar en mm, dividir las coordenadas por 1000.",
            "Pendiente experimental: muestra A con al menos 30 repeticiones, muestra B independiente, eje espectral, offset, ruido, resolución, rango sin alias, datums y geometría del HUD. Estado: prototipo de investigación endurecido, listo para iniciar validación experimental.",
        ]
        for texto in metodo:
            historia += [p(texto, pequeno), Spacer(1, 5)]
        incidencia = informe.get("incidencia", {})
        if incidencia:
            historia += [
                p(
                    "Incidencia oblicua: ángulo máximo "
                    + _numero(incidencia.get("angulo_max_deg"))
                    + " grados; exceso geométrico máximo por capa "
                    + _vector_numerico(incidencia.get("exceso_camino_geometrico_max_um_por_capa"))
                    + " µm. Corrección aplicada: "
                    + ("sí" if incidencia.get("correccion_aplicada", False) else "no")
                    + ". "
                    + str(incidencia.get("advertencia", ""))
                    + ". Incertidumbre del sesgo de espesor por incidencia: no disponible.",
                    pequeno,
                ),
                Spacer(1, 5),
            ]
        for nombre, texto in informe.get("incertidumbre_superficies", {}).items():
            legible = (
                "no disponible"
                if texto is None
                else (("sí" if texto else "no") if isinstance(texto, bool) else str(texto))
            )
            historia += [p(str(nombre).replace("_", " ") + ": " + legible, pequeno), Spacer(1, 4)]
        documento = SimpleDocTemplate(
            str(destino),
            pagesize=landscape(A4),
            rightMargin=32,
            leftMargin=32,
            topMargin=45,
            bottomMargin=max(80, tope_pie + 12),
            title="Vitro - informe de investigación",
            author="Motor Vitro local",
        )
        documento.build(historia, onFirstPage=pie, onLaterPages=pie)
    return destino


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Genera un informe PDF local sin abrir la interfaz"
    )
    parser.add_argument("informe", help="informe.json")
    parser.add_argument("mapas", help="mapas.npz")
    parser.add_argument("destino", help="archivo PDF de salida")
    opciones = parser.parse_args(argv)
    generar_pdf(opciones.informe, opciones.mapas, opciones.destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
