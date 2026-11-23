"""Punto de entrada único: ventana y órdenes de línea de comandos."""

import argparse
import sys
from pathlib import Path


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Gemelo digital de vidrio laminado"
    )
    ordenes = parser.add_subparsers(dest="orden")

    interfaz = ordenes.add_parser("interfaz", help="abre la ventana (opción por defecto)")
    interfaz.add_argument("--proyecto", help="carpeta o archivo proyecto.vitro.json")
    interfaz.add_argument("--informe", help="informe.json ya calculado")

    demo = ordenes.add_parser("demo", help="ejecuta el escenario sintético completo")
    demo.add_argument("--salida", default="salidas/demo")
    demo.add_argument("--semilla", type=int, default=0)
    demo.add_argument("--ruido", type=float, default=0.01)
    demo.add_argument("--n-lado", type=int, default=13)

    ejemplo = ordenes.add_parser("ejemplo", help="genera datos sintéticos de una pieza")
    ejemplo.add_argument("carpeta")
    ejemplo.add_argument("--variante", choices=("base", "conforme", "defecto"), default="conforme")
    ejemplo.add_argument("--n-lado", type=int, default=17)
    ejemplo.add_argument("--semilla", type=int, default=0)
    ejemplo.add_argument("--ruido", type=float, default=0.01)

    analizar = ordenes.add_parser("analizar", help="analiza un config.json")
    analizar.add_argument("config")

    receta = ordenes.add_parser("receta", help="evalúa recetas sin fabricar")
    receta.add_argument("config")
    receta.add_argument("--salida", default="salidas/recetas.json")

    imp = ordenes.add_parser(
        "importar-cad", help="importa un CAD (STEP, o un STL cerrado por capa) al formato del motor"
    )
    imp.add_argument(
        "entrada", nargs="+", help="archivo STEP/IGES/BREP, o 2+ STL (una capa cada uno)"
    )
    imp.add_argument("--salida", default="cad_importado")
    imp.add_argument("--unidades", default="mm", help="unidades de los STL: um, mm, cm, m, in")
    imp.add_argument(
        "--lado-sonda", default="convexo", help="convexo, concavo o nombre de capa extrema"
    )
    imp.add_argument("--marco", choices=("local", "original"), default="local")
    imp.add_argument("--tam-malla-mm", type=float, help="tamaño de teselado del STEP")
    imp.add_argument("--paso-nominal-mm", type=float, help="paso de la rejilla de espesores")
    imp.add_argument(
        "--zona",
        action="append",
        default=[],
        metavar="NOMBRE=ARCHIVO.stl",
        help="superficie auxiliar (solo con STL por capa); se puede repetir",
    )

    ejc = ordenes.add_parser(
        "ejemplo-cad", help="genera una pieza sintética sobre un CAD importado"
    )
    ejc.add_argument("modelo", help="cad_importado.json")
    ejc.add_argument("carpeta")
    ejc.add_argument("--variante", choices=("conforme", "defecto"), default="conforme")
    ejc.add_argument("--n-lado", type=int, default=15)
    ejc.add_argument("--semilla", type=int, default=0)
    ejc.add_argument("--ruido", type=float, default=0.01)
    ejc.add_argument("--materiales", help="un nombre de catálogo por capa, separados por comas")
    ejc.add_argument("--zona", help="restringe los puntos a una zona del CAD importado")

    pdf = ordenes.add_parser("pdf", help="genera el informe PDF")
    pdf.add_argument("informe")
    pdf.add_argument("mapas")
    pdf.add_argument("destino")
    return parser


def _interfaz(args) -> int:
    try:
        from motor.interfaz.ventana import ejecutar
    except ImportError as exc:
        print(f"No se pudo abrir la interfaz. {exc}", file=sys.stderr)
        return 2
    return ejecutar(proyecto=args.proyecto, informe=args.informe)


def _demo(args) -> int:
    from motor.analisis.pipeline import correr_demo

    correr_demo(args.salida, args.ruido, args.semilla, args.n_lado)
    print(f"\nResultados en: {Path(args.salida).resolve()}")
    return 0


def _ejemplo(args) -> int:
    from motor.analisis.demo import guardar_escenario

    ruta = guardar_escenario(args.carpeta, args.ruido, args.semilla, args.n_lado, args.variante)
    print(f"Ejemplo escrito. Corre:  python main.py analizar {ruta}")
    return 0


def _analizar(args) -> int:
    from motor.analisis.pipeline import ejecutar_desde_config

    ejecutar_desde_config(args.config)
    return 0


def _receta(args) -> int:
    from motor.analisis.recetas import ejecutar_config

    ejecutar_config(args.config, args.salida)
    print(f"Informe exploratorio escrito en: {Path(args.salida).resolve()}")
    return 0


def _importar_cad(args) -> int:
    from motor.geometria.importar_cad import OpcionesImportacion, importar
    from motor.medicion.lectura import ErrorDeEntrada

    zonas = {}
    for z in args.zona:
        nombre, separador, ruta = z.partition("=")
        if not separador:
            raise ErrorDeEntrada("--zona usa NOMBRE=ARCHIVO.stl")
        zonas[nombre] = ruta
    op = OpcionesImportacion(
        lado_sonda=args.lado_sonda,
        marco=args.marco,
        paso_nominal_um=None if args.paso_nominal_mm is None else args.paso_nominal_mm * 1e3,
    )
    modelo, ruta = importar(
        args.entrada, args.salida, op, args.unidades, args.tam_malla_mm, zonas or None
    )
    print(f"Capas (desde el lado de la sonda): {', '.join(modelo.nombres_capas)}")
    for nombre, c in modelo.diagnostico["capas"].items():
        print(
            f"  {nombre}: {c['espesor_min_um']:.1f}–{c['espesor_max_um']:.1f} µm, "
            f"cuña p95 {c['cuna_p95_mrad']:.3f} mrad"
        )
    print(f"Zonas sobre el vidrio: {len(modelo.zonas)}")
    for aviso in modelo.avisos:
        print(f"  aviso: {aviso}")
    print(f"Modelo escrito en {ruta}; fragmento de configuración en config_fragmento.json")
    return 0


def _ejemplo_cad(args) -> int:
    from motor.analisis.sintetico_cad import generar_desde_cad

    ruta = generar_desde_cad(
        args.modelo,
        args.carpeta,
        n_lado=args.n_lado,
        ruido=args.ruido,
        semilla=args.semilla,
        materiales=args.materiales.split(",") if args.materiales else None,
        variante=args.variante,
        zona=args.zona,
    )
    print(f"Ejemplo escrito. Corre:  python main.py analizar {ruta}")
    return 0


def _pdf(args) -> int:
    from motor.salida.reporte_pdf import generar_pdf

    generar_pdf(args.informe, args.mapas, args.destino)
    return 0


ORDENES = {
    None: _interfaz,
    "interfaz": _interfaz,
    "demo": _demo,
    "ejemplo": _ejemplo,
    "analizar": _analizar,
    "receta": _receta,
    "importar-cad": _importar_cad,
    "ejemplo-cad": _ejemplo_cad,
    "pdf": _pdf,
}


def main(argv=None) -> int:
    args = construir_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.orden is None:
        args.proyecto = args.informe = None
    from motor.medicion.lectura import ErrorDeEntrada

    try:
        return ORDENES[args.orden](args)
    except (ErrorDeEntrada, OSError) as exc:
        print(f"ERROR DE ENTRADA: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
