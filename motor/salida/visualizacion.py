from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Mapping

import numpy as np


def cargar_mapas(mapas):
    if isinstance(mapas, (str, Path)):
        with np.load(mapas, allow_pickle=False) as archivo:
            datos = {k: archivo[k].copy() for k in archivo.files}
    elif isinstance(mapas, Mapping):
        datos = {k: np.asarray(v) for k, v in mapas.items()}
    else:
        raise ValueError("Mapas debe ser un diccionario o la ruta de mapas.npz")
    faltan = [k for k in ("X", "Y", "d_media", "mascara_soporte") if k not in datos]
    if faltan:
        raise ValueError("Faltan mapas requeridos: " + ", ".join(faltan))
    X, Y = datos["X"], datos["Y"]
    mask = np.asarray(datos["mascara_soporte"], bool)
    if (
        X.ndim != 2
        or X.shape != Y.shape
        or X.shape != mask.shape
        or datos["d_media"].shape[1:] != X.shape
    ):
        raise ValueError("X, Y, espesores y máscara deben compartir la misma rejilla")
    if not mask.any():
        raise ValueError("La máscara no contiene zonas medidas")
    datos["mascara_soporte"] = mask
    return datos


def _etiquetas(informe):
    from motor.estado import ETIQUETA_HUD, condicion_pose, veredicto

    return veredicto(informe), condicion_pose(informe), ETIQUETA_HUD


def _figura(tamano):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=tamano, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _encabezado(fig, informe, titulo, extra=""):
    estado, pose, hud = _etiquetas(informe)
    fig.suptitle(f"{titulo} | {estado}", fontsize=14, color="#16344a", y=0.99)
    ancho_texto = int(fig.get_figwidth() * 14)
    subtitulo = textwrap.fill(f"{pose}. HUD: {hud}. {extra}".strip(), width=ancho_texto)
    fig.text(0.02, 0.945, subtitulo, fontsize=8, va="top")
    altura_linea = 11 / (72 * fig.get_figheight())
    superior = min(0.86, 0.945 - altura_linea * (subtitulo.count("\n") + 1) - 0.045)
    return superior, 0.08


def _campo(datos, mapa, capa=1, interfaz=1):
    opciones = {
        "espesor": ("d_media", capa, "Espesor", "µm", False, "viridis"),
        "desviacion": ("delta_media", interfaz, "Desviación vs CAD", "µm", True, "RdBu_r"),
        "cuna": ("w", None, "Cuña del PVB con signo", "mrad", True, "RdBu_r"),
        "incertidumbre": ("d_std", capa, "σ parcial modelada del espesor", "µm", False, "magma"),
        "probabilidad": (
            "P_fuera_d",
            capa,
            "Probabilidad puntual fuera de tolerancia",
            "1",
            False,
            "viridis",
        ),
    }
    if mapa not in opciones:
        raise ValueError("Mapa desconocido: " + str(mapa))
    clave, indice, titulo, unidad, diverge, cmap = opciones[mapa]
    if clave not in datos:
        raise ValueError(f"No disponible el mapa {clave}")
    valor = np.asarray(datos[clave], float)
    if indice is not None:
        if not 0 <= indice < len(valor):
            raise ValueError("Capa o interfaz fuera de rango")
        valor = valor[indice]
        titulo += f" {indice + 1 if mapa != 'desviacion' else indice}"
    valido = datos["mascara_soporte"] & np.isfinite(valor)
    if not valido.any():
        raise ValueError(f"El mapa {clave} no tiene resultados disponibles dentro del soporte")
    if diverge:
        limite = max(float(np.max(np.abs(valor[valido]))), 1e-12)
        clim = (-limite, limite)
    elif mapa == "probabilidad":
        if np.any((valor[valido] < 0) | (valor[valido] > 1)):
            raise ValueError("Las probabilidades deben estar entre 0 y 1")
        clim = (0, 1)
    else:
        clim = (float(np.min(valor[valido])), float(np.max(valor[valido])))
        if clim[0] == clim[1]:
            clim = (clim[0] - 0.5, clim[1] + 0.5)
    return np.ma.masked_where(~valido, valor), titulo, unidad, cmap, clim


def figura_mapa(informe, mapas, mapa="espesor", capa=1, interfaz=1):
    datos = cargar_mapas(mapas)
    fig = _figura((10, 6.7))
    ax = fig.add_subplot(111)
    valor, titulo, unidad, cmap, clim = _campo(datos, mapa, capa, interfaz)
    im = ax.pcolormesh(
        datos["X"] / 1000,
        datos["Y"] / 1000,
        valor,
        shading="nearest",
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    fig.colorbar(im, ax=ax, label=unidad)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=titulo)
    ax.set_aspect("equal")
    superior, inferior = _encabezado(fig, informe, titulo)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=max(inferior, 0.1), top=superior)
    return fig


def crear_mapas(informe, mapas, destino):
    datos = cargar_mapas(mapas)
    n = len(datos["d_media"])
    fig = _figura((16, 9))
    opciones = [("espesor", j, 0) for j in range(n)]
    opciones += [
        ("cuna", 0, 0),
        ("desviacion", 0, 0),
        ("desviacion", 0, n),
        ("incertidumbre", min(1, n - 1), 0),
        ("probabilidad", min(1, n - 1), 0),
    ]
    cols = max(4, int(np.ceil(len(opciones) / 2)))
    for numero, (mapa, capa, interfaz) in enumerate(opciones):
        ax = fig.add_subplot(2, cols, numero + 1)
        try:
            valor, titulo, unidad, cmap, clim = _campo(datos, mapa, capa, interfaz)
            im = ax.pcolormesh(
                datos["X"] / 1000,
                datos["Y"] / 1000,
                valor,
                shading="nearest",
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
            )
            fig.colorbar(im, ax=ax, label=unidad, shrink=0.85)
            ax.set_title(textwrap.fill(titulo, 29), fontsize=9)
        except ValueError as error:
            ax.text(
                0.5, 0.5, str(error), ha="center", va="center", transform=ax.transAxes, wrap=True
            )
        ax.set(xlabel="x (mm)", ylabel="y (mm)")
        ax.set_aspect("equal")
    superior, inferior = _encabezado(fig, informe, "Mapas de reconstrucción")
    fig.subplots_adjust(
        left=0.05, right=0.97, bottom=max(inferior, 0.09), top=superior, hspace=0.48, wspace=0.55
    )
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=145, metadata={"Description": " | ".join(_etiquetas(informe))})
    return destino


def _interpolar_en_rejilla(datos, campos, xy):
    X, Y = datos["X"], datos["Y"]
    xs, ys = X[0], Y[:, 0]
    if not (
        np.allclose(X, xs[None, :])
        and np.allclose(Y, ys[:, None])
        and np.all(np.diff(xs) > 0)
        and np.all(np.diff(ys) > 0)
    ):
        raise ValueError("El corte requiere una rejilla rectangular creciente")
    nx, ny = len(xs), len(ys)
    ix = np.clip(np.searchsorted(xs, xy[:, 0], side="right") - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(ys, xy[:, 1], side="right") - 1, 0, ny - 2)
    tx = (xy[:, 0] - xs[ix]) / (xs[ix + 1] - xs[ix])
    ty = (xy[:, 1] - ys[iy]) / (ys[iy + 1] - ys[iy])
    dentro = (xy[:, 0] >= xs[0]) & (xy[:, 0] <= xs[-1]) & (xy[:, 1] >= ys[0]) & (xy[:, 1] <= ys[-1])
    valido = dentro.copy()
    resultado = np.zeros((len(xy),) + campos.shape[:-2])
    for dx, dy, peso in (
        (0, 0, (1 - tx) * (1 - ty)),
        (1, 0, tx * (1 - ty)),
        (0, 1, (1 - tx) * ty),
        (1, 1, tx * ty),
    ):
        aporta = peso > 1e-12
        valores = np.moveaxis(campos[..., iy + dy, ix + dx], -1, 0)
        finitos = np.isfinite(valores).reshape(len(xy), -1).all(axis=1)
        valido &= ~aporta | (datos["mascara_soporte"][iy + dy, ix + dx] & finitos)
        resultado += np.where(np.isfinite(valores), valores, 0) * peso.reshape(
            (-1,) + (1,) * (valores.ndim - 1)
        )
    resultado[~valido] = np.nan
    return resultado, valido


def perfil_corte(mapas, inicio=None, fin=None, n=200):
    datos = cargar_mapas(mapas)
    X, Y = datos["X"], datos["Y"]
    if inicio is None:
        fila = X.shape[0] // 2
        inicio = (X[fila, 0], Y[fila, 0])
    if fin is None:
        fila = X.shape[0] // 2
        fin = (X[fila, -1], Y[fila, -1])
    inicio, fin = np.asarray(inicio, float), np.asarray(fin, float)
    if (
        inicio.shape != (2,)
        or fin.shape != (2,)
        or not np.isfinite([*inicio, *fin]).all()
        or np.array_equal(inicio, fin)
    ):
        raise ValueError("Elige dos puntos distintos y finitos para el corte (x, y en µm)")
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("El corte requiere al menos dos muestras")
    xy = np.linspace(inicio, fin, n)
    d, mask = _interpolar_en_rejilla(datos, datos["d_media"], xy)
    interfaces = np.column_stack([np.zeros(n), np.cumsum(d, axis=1)])
    interfaces[~mask] = np.nan
    p = None
    if "P_fuera_d" in datos:
        p, _ = _interpolar_en_rejilla(datos, datos["P_fuera_d"], xy)
    w = None
    if "w" in datos:
        w, _ = _interpolar_en_rejilla(datos, datos["w"][None], xy)
        w = w[:, 0]
    pw = None
    if "P_fuera_w" in datos:
        pw, _ = _interpolar_en_rejilla(datos, datos["P_fuera_w"][None], xy)
        pw = pw[:, 0]
    return {
        "xy_um": xy,
        "s_um": np.linalg.norm(xy - inicio, axis=1),
        "d_media_um": d,
        "interfaces_um": interfaces,
        "soporte": mask,
        "P_fuera_d": p,
        "P_fuera_w": pw,
        "w_mrad": w,
        "inicio_um": inicio,
        "fin_um": fin,
    }


def figura_corte(informe, mapas, inicio=None, fin=None, n=200, exageracion=None):
    if exageracion is not None and (not np.isfinite(exageracion) or exageracion <= 0):
        raise ValueError("La exageración vertical debe ser positiva")
    datos = cargar_mapas(mapas)
    perfil = perfil_corte(datos, inicio, fin, n)
    fig = _figura((12, 8))
    grilla = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.60)
    ax = fig.add_subplot(grilla[0])
    detalle = fig.add_subplot(grilla[1], sharex=ax)
    x = perfil["s_um"] / 1000
    limites = perfil["interfaces_um"]
    colores = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
    nombres = list(informe.get("espesor", {}))
    umbral = informe.get("regla_calidad", {}).get("umbral_P_fuera", 0.05)
    for j in range(limites.shape[1] - 1):
        nombre = nombres[j] if j < len(nombres) else f"capa {j+1}"
        ax.fill_between(
            x,
            limites[:, j],
            limites[:, j + 1],
            color=colores[j % len(colores)],
            alpha=0.65,
            label=nombre,
        )
        ax.plot(x, limites[:, j + 1], color="#263238", linewidth=0.8)
        p = perfil["P_fuera_d"]
        if p is not None:
            falla = (p[:, j] > umbral) & perfil["soporte"]
            ax.scatter(
                x[falla],
                0.5 * (limites[falla, j] + limites[falla, j + 1]),
                marker="x",
                color="#8B0000",
                s=15,
            )
    ax.set(xlabel="Distancia (mm)", ylabel="Espesor acumulado (µm)")
    if exageracion is not None:
        ax.set_aspect(exageracion / 1000, adjustable="box")
    ax.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(alpha=0.15)
    w = perfil["w_mrad"]
    texto_w = (
        "Cuña no disponible"
        if w is None or not np.isfinite(w).any()
        else f"Cuña en el corte: {np.nanmin(w):+.5g} a {np.nanmax(w):+.5g} mrad"
    )
    titulo_perfil = "Perfiles aplanados; cruces = P puntual mayor al umbral"
    if exageracion is None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.18 * (ymax - ymin))
    ax.set_title(titulo_perfil, fontsize=9)
    indice_pvb = informe.get("cuna_pvb_mrad", {}).get(
        "indice_capa", min(1, perfil["d_media_um"].shape[1] - 1)
    )
    if (
        not isinstance(indice_pvb, (int, np.integer))
        or not 0 <= indice_pvb < perfil["d_media_um"].shape[1]
    ):
        raise ValueError("El índice de la capa PVB está fuera de rango en informe.json")
    pvb = perfil["d_media_um"][:, indice_pvb]
    detalle.plot(x, pvb, color="#805500", linewidth=1.5)
    if perfil["P_fuera_w"] is not None:
        falla_cuna = (perfil["P_fuera_w"] > umbral) & perfil["soporte"]
        detalle.scatter(
            x[falla_cuna],
            pvb[falla_cuna],
            color="#8B0000",
            marker="x",
            s=14,
            label="Cuña: P puntual mayor al umbral",
        )
        if falla_cuna.any():
            detalle.legend(loc="best", fontsize=7)
    finitos = pvb[np.isfinite(pvb)]
    rango = max(float(np.ptp(finitos)), 1) if len(finitos) else 1
    escala_detalle = max(float(np.ptp(perfil["s_um"])) / rango * 0.10, 1)
    detalle.set_aspect(escala_detalle / 1000, adjustable="box")
    detalle.set(xlabel="Distancia sobre el corte (mm)", ylabel="Espesor PVB (µm)")
    detalle.set_title(texto_w + f" | detalle: escala vertical ×{escala_detalle:.3g}", fontsize=9)
    detalle.grid(alpha=0.2)
    superior, inferior = _encabezado(fig, informe, "Corte transversal aplanado")
    fig.subplots_adjust(left=0.1, right=0.97, bottom=max(inferior, 0.12), top=min(superior, 0.82))
    fig.canvas.draw()
    caja = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ancho_in, alto_in = caja.width * fig.get_figwidth(), caja.height * fig.get_figheight()
    if exageracion is not None:
        efectivo = exageracion
    else:
        efectivo = (alto_in / abs(y1 - y0)) / (ancho_in / (abs(x1 - x0) * 1000))
    ax.set_title(
        titulo_perfil
        + f" | escala vertical ×{efectivo:.3g}"
        + (" (ajuste automático)" if exageracion is None else ""),
        fontsize=9,
    )
    fig.exageracion_vertical = float(efectivo)
    return fig


def crear_corte(informe, mapas, destino, **opciones):
    fig = figura_corte(informe, mapas, **opciones)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150, metadata={"Description": " | ".join(_etiquetas(informe))})
    return destino


def geometria_vista_3d(mapas, exageracion=1, explotada=False):
    datos = cargar_mapas(mapas)
    if "interfaces" not in datos:
        raise ValueError("Faltan las interfaces 3D en mapas.npz")
    S = np.asarray(datos["interfaces"], float).copy()
    if S.shape[1:3] != datos["X"].shape or S.shape[-1] != 3:
        raise ValueError("Las interfaces 3D deben tener forma (K,ny,nx,3)")
    if not np.isfinite(exageracion) or exageracion <= 0:
        raise ValueError("La exageración debe ser positiva")
    if exageracion != 1:
        if "interfaces_nominales" not in datos:
            raise ValueError(
                "Para exagerar diferencias hace falta interfaces_nominales; usa escala vertical ×1"
            )
        nominal = datos["interfaces_nominales"]
        if nominal.shape != S.shape:
            raise ValueError("Las interfaces nominales y medidas deben compartir forma")
        S[..., 2] = nominal[..., 2] + exageracion * (S[..., 2] - nominal[..., 2])
    S[:, ~datos["mascara_soporte"]] = np.nan
    return S


def _triangulos_mask(mask):
    ny, nx = mask.shape
    celdas = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    iy, ix = np.nonzero(celdas)
    a = iy * nx + ix
    return np.concatenate(
        [np.column_stack([a, a + 1, a + nx + 1]), np.column_stack([a, a + nx + 1, a + nx])]
    )


def _capas_vista(S, datos, explotada):
    separacion = float(np.nanmedian(np.sum(datos["d_media"], axis=0))) * 0.6 if explotada else 0
    capas = []
    for j in range(len(S) - 1):
        capa = S[j : j + 2].copy()
        capa[..., 2] += j * separacion
        capas.append(capa)
    return capas


def _aristas_borde(triangulos):
    aristas = np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]])
    aristas.sort(axis=1)
    unicas, cuenta = np.unique(aristas, axis=0, return_counts=True)
    return unicas[cuenta == 1]


def crear_vista_3d(
    informe,
    mapas,
    destino=None,
    off_screen=True,
    usar_pyvista=True,
    exageracion=1,
    explotada=False,
    mapa="espesor",
    capa=1,
    interfaz=1,
):
    datos = cargar_mapas(mapas)
    S = geometria_vista_3d(datos, exageracion, explotada)
    capas_geometria = _capas_vista(S, datos, explotada)
    valor, titulo, unidad, cmap, clim = _campo(datos, mapa, capa, interfaz)
    estado, pose, hud = _etiquetas(informe)
    extra = " | vista explotada (separación artificial)" if explotada else ""
    rotulo = f"{estado}\n{pose}\nHUD: {hud}\nescala vertical ×{exageracion:g} de diferencias respecto al CAD{extra}"
    pv = None
    if usar_pyvista:
        try:
            import pyvista as pv
        except ImportError:
            pv = None
    if pv is not None:
        plotter = pv.Plotter(off_screen=off_screen, window_size=(1400, 950))
        try:
            pv.set_new_attribute(plotter, "backend_vitro", "pyvista")
        except AttributeError:
            plotter.backend_vitro = "pyvista"
        mask = datos["mascara_soporte"]
        validas = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
        for capa_3d in capas_geometria:
            volumen = np.moveaxis(capa_3d, 0, 2)
            seguro = np.where(np.isfinite(volumen), volumen, 0) / 1000
            grid = pv.StructuredGrid(seguro[..., 0], seguro[..., 1], seguro[..., 2])
            grid.point_data["resultado"] = np.repeat(
                np.asarray(valor.filled(np.nan))[..., None], 2, axis=2
            ).ravel(order="F")
            celdas = grid.extract_cells(validas.ravel(order="F"))
            try:
                malla = celdas.extract_surface(algorithm="dataset_surface")
            except TypeError:
                malla = celdas.extract_surface()
            plotter.add_mesh(
                malla,
                scalars="resultado",
                cmap=cmap,
                clim=clim,
                opacity=0.72,
                scalar_bar_args={
                    "title": f"{titulo} ({unidad})",
                    "vertical": True,
                    "position_x": 0.90,
                    "position_y": 0.18,
                    "height": 0.62,
                    "width": 0.07,
                    "title_font_size": 14,
                    "label_font_size": 12,
                    "n_labels": 5,
                    "fmt": "%.3g",
                },
                show_scalar_bar=True,
            )
        plotter.add_text(rotulo, position="upper_left", font_size=9, color="#10293b")
        plotter.show_grid(
            xtitle="x (mm)",
            ytitle="y (mm)",
            ztitle="z (mm)",
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=2,
            fmt="%.1f",
            font_size=10,
        )
        plotter.set_background("white")
        plotter.view_isometric()
        if destino is None:
            return plotter
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(destino))
        plotter.close()
        return destino
    from matplotlib import colormaps, colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = _figura((12, 8))
    fig.backend_vitro = "matplotlib_estatico"
    ax = fig.add_subplot(111, projection="3d")
    triangulos = _triangulos_mask(datos["mascara_soporte"])
    bordes = _aristas_borde(triangulos)
    normalizar = colors.Normalize(*clim)
    paleta = colormaps[cmap]
    planos = np.asarray(valor.filled(np.nan)).ravel()[triangulos].mean(axis=1)
    for capa_3d in capas_geometria:
        for superficie in capa_3d:
            vertices = superficie.reshape(-1, 3)[triangulos] / 1000
            finitos = np.isfinite(vertices).all(axis=(1, 2)) & np.isfinite(planos)
            coleccion = Poly3DCollection(
                vertices[finitos],
                facecolors=paleta(normalizar(planos[finitos])),
                edgecolors="none",
                alpha=0.72,
            )
            ax.add_collection3d(coleccion)
        ab = capa_3d[0].reshape(-1, 3)[bordes] / 1000
        ar = capa_3d[1].reshape(-1, 3)[bordes] / 1000
        caras_laterales = np.stack([ab[:, 0], ab[:, 1], ar[:, 1], ar[:, 0]], axis=1)
        color_borde = np.asarray(valor.filled(np.nan)).ravel()[bordes].mean(axis=1)
        finitos = np.isfinite(caras_laterales).all(axis=(1, 2)) & np.isfinite(color_borde)
        ax.add_collection3d(
            Poly3DCollection(
                caras_laterales[finitos],
                facecolors=paleta(normalizar(color_borde[finitos])),
                edgecolors="none",
                alpha=0.65,
            )
        )
    puntos = (
        np.concatenate(
            [capa[:, datos["mascara_soporte"]].reshape(-1, 3) for capa in capas_geometria]
        )
        / 1000
    )
    ax.auto_scale_xyz(puntos[:, 0], puntos[:, 1], puntos[:, 2])
    rango = np.maximum(np.ptp(puntos, axis=0), 1e-9)
    ax.set_box_aspect(rango)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    from matplotlib.ticker import MaxNLocator

    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis="z", labelsize=7)
    mappable = colormaps[cmap]
    from matplotlib.cm import ScalarMappable

    fig.colorbar(
        ScalarMappable(norm=normalizar, cmap=mappable),
        ax=ax,
        label=f"{titulo} ({unidad})",
        shrink=0.57,
        pad=0.08,
    )
    superior, inferior = _encabezado(
        fig,
        informe,
        "Capas 3D - captura estática local",
        f"escala vertical ×{exageracion:g} de diferencias respecto al CAD{extra}. Backend: matplotlib.",
    )
    fig.subplots_adjust(left=0.04, right=0.88, bottom=inferior, top=min(superior, 0.83))
    if destino is None:
        return fig
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destino, dpi=140, metadata={"Description": rotulo + " | Backend: matplotlib_estatico"}
    )
    return destino
