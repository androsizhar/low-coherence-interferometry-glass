import numpy as np
import pytest

from motor.geometria.cad import leer_stl
from motor.geometria.exportar_cad import (
    es_cerrada,
    exportar_stl,
    malla_superficie,
    solido_capa,
)
from motor.medicion.lectura import ErrorDeEntrada

UM = 1e3


def _grilla(nx, ny, L=100 * UM, curvatura=0.0, z0=0.0):
    xs, ys = np.linspace(0, L, nx), np.linspace(0, L, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = z0 + curvatura * (X - L / 2) ** 2 / 2
    return np.stack([X, Y, Z], axis=-1)


def test_capa_plana_volumen_exacto_y_cerrada():
    top = _grilla(11, 9, z0=0.0)
    bot = _grilla(11, 9, z0=-760.0)
    m = solido_capa(top, bot)
    assert es_cerrada(m)
    assert m.volumen() == pytest.approx(760.0 * (100 * UM) ** 2, rel=1e-12)


def test_capa_curva_con_cuna_volumen_coincide_con_integral():
    top = _grilla(61, 61, curvatura=1 / (1500 * UM), z0=0.0)
    bot = top.copy()
    X = bot[..., 0]
    esp = 760.0 + 0.35 * X / UM
    bot[..., 2] = top[..., 2] - esp
    m = solido_capa(top, bot)
    assert es_cerrada(m) and m.volumen() > 0
    esperado = (100 * UM) ** 2 * (760.0 + 0.35 * 50.0)
    assert m.volumen() == pytest.approx(esperado, rel=1e-9)


def test_exportar_stl_ida_y_vuelta(tmp_path):
    ints = [_grilla(15, 15, z0=-z) for z in (0.0, 2100.0, 2860.0, 4460.0)]
    rep = exportar_stl(str(tmp_path), ints, ["vidrio_ext", "pvb", "vidrio_int"])
    assert len(rep.archivos) == 4 + 3
    assert all(rep.cerradas.values())
    esperados = {"vidrio_ext_1": 2100.0, "pvb_2": 760.0, "vidrio_int_3": 1600.0}
    for nombre, esp in esperados.items():
        assert rep.volumenes_um3[nombre] == pytest.approx(esp * (100 * UM) ** 2, rel=1e-9)
    m = leer_stl(rep.archivos["capa_2_pvb"])
    assert m.volumen() == pytest.approx(760.0 * (100 * UM) ** 2, rel=1e-4)


def test_validaciones():
    with pytest.raises(ErrorDeEntrada):
        malla_superficie(np.zeros((1, 5, 3)))
    bad = _grilla(5, 5)
    bad[2, 2, 2] = np.nan
    with pytest.raises(ErrorDeEntrada):
        malla_superficie(bad)
    with pytest.raises(ErrorDeEntrada):
        solido_capa(_grilla(5, 5), _grilla(6, 5))
    with pytest.raises(ErrorDeEntrada):
        exportar_stl("x", [_grilla(5, 5)] * 3, ["a", "b", "c"])


def test_malla_abierta_no_es_cerrada():
    assert not es_cerrada(malla_superficie(_grilla(6, 6)))


def test_solido_rechaza_cruces_nan_y_espesor_nulo():
    a = _grilla(4, 4)
    for b in (a.copy(), np.full_like(a, np.nan)):
        with pytest.raises(ErrorDeEntrada):
            solido_capa(a, b)
    b = a.copy()
    b[..., 2] = -10.0
    b[2, 2, 2] = 10.0
    with pytest.raises(ErrorDeEntrada):
        solido_capa(a, b)


def test_exportacion_con_hueco_solo_incluye_soporte_y_cierra_paredes(tmp_path):
    a = _grilla(7, 7, L=6.0)
    b = a.copy()
    b[..., 2] = -2.0
    mascara = np.ones((7, 7), dtype=bool)
    mascara[3, 3] = False
    a[~mascara] = np.nan
    b[~mascara] = np.nan
    rep = exportar_stl(str(tmp_path), [a, b], ["pvb"], mascara=mascara)
    m = leer_stl(rep.archivos["capa_1_pvb"])
    sup = malla_superficie(a, mascara)
    assert es_cerrada(m) and rep.cerradas["pvb_1"]
    assert m.volumen() == pytest.approx(2.0 * sup.areas().sum())
    assert m.volumen() < 2.0 * 36.0


def test_exportacion_valida_antes_de_dejar_archivos(tmp_path):
    a = _grilla(4, 4)
    with pytest.raises(ErrorDeEntrada):
        exportar_stl(str(tmp_path), [a, a], ["pvb"])
    assert not list(tmp_path.iterdir())
    with pytest.raises(ErrorDeEntrada):
        exportar_stl(str(tmp_path), [a, a - [0, 0, 2]], ["../escape"])


def test_cierre_rechaza_dos_solidos_que_solo_comparten_un_vertice():
    from motor.geometria.cad import Malla

    a = solido_capa(_grilla(2, 2, L=1.0, z0=0.0), _grilla(2, 2, L=1.0, z0=-1.0))
    b = solido_capa(-_grilla(2, 2, L=1.0, z0=-1.0), -_grilla(2, 2, L=1.0, z0=0.0))
    coords = np.vstack([a.v, b.v])
    caras = np.vstack([a.f, b.f + len(a.v)])
    unicos, inv = np.unique(coords, axis=0, return_inverse=True)
    m = Malla(unicos, inv[caras])
    assert not es_cerrada(m)
