import argparse
import sys


def construir_parser():
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

    analizar = ordenes.add_parser("analizar", help="analiza un config.json")
    analizar.add_argument("config")

    receta = ordenes.add_parser("receta", help="evalúa recetas sin fabricar")
    receta.add_argument("config")
    receta.add_argument("--salida", default="salidas/recetas.json")

    pdf = ordenes.add_parser("pdf", help="genera el informe PDF")
    pdf.add_argument("informe")
    pdf.add_argument("mapas")
    pdf.add_argument("destino")
    return parser


def main(argv=None):
    args = construir_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.orden in (None, "interfaz"):
        from motor.app import main as abrir

        parametros = []
        for nombre in ("proyecto", "informe"):
            valor = getattr(args, nombre, None)
            if valor:
                parametros += [f"--{nombre}", valor]
        return abrir(parametros)
    if args.orden == "demo":
        from motor.analisis.pipeline import main as correr

        return correr(
            ["--demo", "--salida", args.salida, "--semilla", str(args.semilla)]
            + ["--ruido", str(args.ruido), "--n-lado", str(args.n_lado)]
        )
    if args.orden == "ejemplo":
        from motor.analisis.pipeline import main as correr

        return correr(
            ["--generar-ejemplo", args.carpeta, "--variante", args.variante]
            + ["--n-lado", str(args.n_lado), "--semilla", str(args.semilla)]
        )
    if args.orden == "analizar":
        from motor.analisis.pipeline import main as correr

        return correr(["--config", args.config])
    if args.orden == "receta":
        from motor.analisis.recetas import main as evaluar

        return evaluar(["--config", args.config, "--salida", args.salida])
    from motor.salida.reporte_pdf import main as generar

    return generar([args.informe, args.mapas, args.destino])


if __name__ == "__main__":
    sys.exit(main())
