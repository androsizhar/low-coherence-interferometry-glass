from __future__ import annotations

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vitro · gemelo digital local")
    parser.add_argument("--proyecto", help="Archivo proyecto.json para restaurar")
    parser.add_argument("--informe", help="Abrir informe.json ya calculado")
    args = parser.parse_args(argv)
    try:
        from motor.interfaz.ventana import ejecutar
    except ImportError as exc:
        print(f"No se pudo abrir la interfaz. {exc}", file=sys.stderr)
        return 2
    return ejecutar(proyecto=args.proyecto, informe=args.informe)


if __name__ == "__main__":
    raise SystemExit(main())
