import numpy as np
import pytest

from motor.salida.visualizacion import (
    _capas_vista,
    _triangulos_mask,
    cargar_mapas,
    crear_corte,
    crear_mapas,
    crear_vista_3d,
    figura_mapa,
    geometria_vista_3d,
    perfil_corte,
)


def test_corte_espesores_coinciden_con_d_media(datos_visual):
    perfil = perfil_corte(datos_visual, n=7)
    np.testing.assert_allclose(
        perfil["d_media_um"], datos_visual["d_media"][:, 2, :].T, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        np.diff(perfil["interfaces_um"], axis=1), perfil["d_media_um"], atol=1e-6, rtol=0
    )


def test_corte_no_puentea_hueco(datos_visual):
    datos_visual["mascara_soporte"][2, 3] = False
    perfil = perfil_corte(datos_visual, n=61)
    centro = 30
    assert not perfil["soporte"][centro]
    assert np.isnan(perfil["interfaces_um"][centro]).all()
    assert not perfil["soporte"][25:36].any()
    assert perfil["soporte"][0] and perfil["soporte"][-1]


def test_corte_incluye_zonas_cuna_fuera_tolerancia(datos_visual):
    datos_visual["P_fuera_w"] = np.where(datos_visual["X"] > 0, 0.9, 0.01)
    perfil = perfil_corte(datos_visual, n=7)
    np.testing.assert_allclose(perfil["P_fuera_w"], datos_visual["P_fuera_w"][2])
    assert (perfil["P_fuera_w"] > 0.05).sum() == 3


def test_malla_no_cubre_huecos(datos_visual):
    mask = datos_visual["mascara_soporte"]
    mask[2, 3] = False
    tri = _triangulos_mask(mask)
    assert mask.ravel()[tri].all()
    assert not np.isin(2 * 7 + 3, tri).any()


def test_exageracion_conserva_curvatura_nominal(datos_visual):
    datos_visual["interfaces_nominales"][..., 0] += 10
    S = geometria_vista_3d(datos_visual, exageracion=5)
    np.testing.assert_allclose(S[..., 2], datos_visual["interfaces"][..., 2] + 8)
    np.testing.assert_allclose(S[..., :2], datos_visual["interfaces"][..., :2])
    np.testing.assert_allclose(
        np.diff(S[0, :, :, 2], axis=1),
        np.diff(datos_visual["interfaces_nominales"][0, :, :, 2], axis=1),
    )
    capas = _capas_vista(S, datos_visual, True)
    for j, capa in enumerate(capas):
        np.testing.assert_allclose(capa[1] - capa[0], S[j + 1] - S[j])


def test_exageracion_no_inventa_baseline(datos_visual):
    datos_visual.pop("interfaces_nominales")
    with pytest.raises(ValueError, match="interfaces_nominales"):
        geometria_vista_3d(datos_visual, exageracion=2)


def test_mapa_divergente_simetrico_y_mascara(datos_visual, informe_visual):
    datos_visual["mascara_soporte"][2, 3] = False
    datos_visual["w"][0, 0] = -0.1
    fig = figura_mapa(informe_visual, datos_visual, mapa="cuna")
    imagen = fig.axes[0].collections[0]
    assert imagen.get_clim() == (-0.35, 0.35)
    assert np.ma.getmaskarray(imagen.get_array())[2, 3]
    texto = " ".join(t.get_text() for t in fig.texts)
    assert (
        "no concluyente" in texto
        and "pose sintética exacta" in texto
        and "predicción exploratoria" in texto
    )


def test_capturas_locales_reales_no_uniformes(tmp_path, datos_visual, informe_visual):
    from PIL import Image

    for nombre, funcion in (("mapas", crear_mapas), ("corte", crear_corte)):
        ruta = funcion(informe_visual, datos_visual, tmp_path / f"{nombre}.png")
        imagen = np.asarray(Image.open(ruta))
        assert imagen.std() > 5
        assert "pose sintética exacta" in Image.open(ruta).info["Description"]
    fig = crear_vista_3d(informe_visual, datos_visual, usar_pyvista=False)
    assert fig.backend_vitro == "matplotlib_estatico"
    ruta = crear_vista_3d(
        informe_visual, datos_visual, tmp_path / "vista3d.png", usar_pyvista=False
    )
    assert np.asarray(Image.open(ruta)).std() > 5


def test_npz_seguro(tmp_path, datos_visual):
    ruta = tmp_path / "mapas.npz"
    np.savez(ruta, **datos_visual)
    assert cargar_mapas(ruta)["X"].shape == (5, 7)


def test_pyvista_offscreen_no_uniforme(tmp_path, datos_visual, informe_visual):
    pytest.importorskip("pyvista")
    from PIL import Image

    ruta = crear_vista_3d(informe_visual, datos_visual, tmp_path / "vtk.png", off_screen=True)
    assert np.asarray(Image.open(ruta)).std() > 5


def test_pyvista_escalares_fortran(datos_visual, informe_visual):
    pytest.importorskip("pyvista")
    plotter = crear_vista_3d(informe_visual, datos_visual, mapa="espesor", capa=1, off_screen=True)
    try:
        for actor in plotter.actors.values():
            mapper = getattr(actor, "mapper", None)
            if (
                mapper is not None
                and mapper.dataset is not None
                and "resultado" in mapper.dataset.point_data
            ):
                mesh = mapper.dataset
                np.testing.assert_allclose(
                    mesh.point_data["resultado"], 760 + mesh.points[:, 0] * 1000 * 0.00035
                )
    finally:
        plotter.close()


def test_corte_automatico_declara_factor_y_fijo_respeta_escala(
    datos_visual, informe_visual, tmp_path
):
    from motor.salida.visualizacion import figura_corte

    for clave in ("X", "Y"):
        datos_visual[clave] = datos_visual[clave] * 15
    datos_visual["interfaces"][..., :2] *= 15
    datos_visual["interfaces_nominales"][..., :2] *= 15
    auto = figura_corte(informe_visual, datos_visual, n=7)
    assert auto.exageracion_vertical > 1
    titulos = [a.get_title() for a in auto.axes]
    assert any(
        f"escala vertical ×{auto.exageracion_vertical:.3g}" in t and "automático" in t
        for t in titulos
    )
    real = figura_corte(informe_visual, datos_visual, n=7, exageracion=1)
    assert real.exageracion_vertical == 1
    assert any(
        "escala vertical ×1" in a.get_title() and "automático" not in a.get_title()
        for a in real.axes
    )
    with pytest.raises(ValueError):
        figura_corte(informe_visual, datos_visual, exageracion=0)
    png = crear_corte(informe_visual, datos_visual, tmp_path / "c.png")
    assert png.stat().st_size > 10_000


def test_pyvista_no_emite_aviso_de_algoritmo(datos_visual, informe_visual, tmp_path):
    pytest.importorskip("pyvista")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ruta = crear_vista_3d(informe_visual, datos_visual, tmp_path / "v.png", off_screen=True)
    assert ruta.stat().st_size > 5_000
