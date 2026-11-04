import json
import os

import numpy as np
import pytest

from motor.analisis import demo as D
from motor.analisis.pipeline import (
    calibrar_desde_interferogramas,
    configuracion_demo,
    correr_demo,
    ejecutar,
    ejecutar_desde_config,
)
from motor.medicion.lectura import ErrorDeEntrada

UM = 1e3


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    carpeta = str(tmp_path_factory.mktemp("salida"))
    return correr_demo(carpeta, ruido=0.01, semilla=0, n_lado=11, verbose=False) + (carpeta,)


def test_sin_defectos_calibracion_y_validacion_ok(corrida):
    esc, cfg, cal, res, inf, _ = corrida
    assert not res.ids_defecto
    assert cal.ok and inf["validacion_cruzada"]["ok"]
    assert inf["registro"]["convergio"]
    assert all(not b.startswith("puntos_descartados") for b in res.banderas)


def test_espesor_por_punto_contra_la_verdad(corrida):
    esc, cfg, cal, res, inf, _ = corrida
    err = res.d_puntos - esc.verdad_d[res.ids_ok]
    assert np.all(np.sqrt((err**2).mean(axis=0)) < 0.5), np.sqrt((err**2).mean(axis=0))
    z = err / np.sqrt(res.sigma_aleatoria**2 + res.sigma_sistematica**2)
    assert np.all(np.sqrt((z**2).mean(axis=0)) < 1.5)


def test_mapas_de_espesor_y_cuna_contra_la_verdad(corrida):
    esc, cfg, cal, res, inf, _ = corrida
    X, Y = res.Xg, res.Yg
    dtrue = D.espesores_verdaderos(X, Y)
    for j in range(3):
        e = res.d_media[j] - dtrue[..., j]
        assert np.sqrt((e**2).mean()) < 0.3
        assert (np.abs(e) < 2 * res.d_std[j]).mean() > 0.9
    s = res.cuna.direccion
    assert abs(np.degrees(np.arctan2(s[1], s[0]))) < 5.0
    xm, ym = X / UM, Y / UM
    onda = 2.0 * (-np.sin(2 * np.pi * (xm + ym) / 90)) * (2 * np.pi / 90)
    w_true = (0.35 + onda) * s[0] + onda * s[1]
    e = res.w_mapa - w_true
    assert (np.abs(e) < 2 * res.sigma_w_mapa).mean() > 0.9
    assert abs(res.w_mapa.mean() - 0.35) < 0.03


def test_desviacion_contra_cad_limitada_por_el_registro(corrida):
    esc, cfg, cal, res, inf, _ = corrida
    err = res.delta_puntos - esc.verdad_delta[:, res.ids_ok]
    assert np.all(np.sqrt((err**2).mean(axis=1)) < 4.0), np.sqrt((err**2).mean(axis=1))
    dif = err[3] - err[0]
    assert np.sqrt((dif**2).mean()) < 0.5


def test_stl_cerrados_y_volumenes_coherentes(corrida):
    esc, cfg, cal, res, inf, _ = corrida
    cad = inf["cad"]
    assert all(cad["solidos_cerrados"].values())
    v = cad["volumenes_um3"]
    esp = inf["espesor"]
    r_pvb = v["pvb_2"] / v["vidrio_ext_1"]
    esperado = esp["pvb"]["media_um"] / esp["vidrio_ext"]["media_um"]
    assert r_pvb == pytest.approx(esperado, rel=0.02)
    for ruta in cad["archivos"].values():
        assert os.path.getsize(ruta) > 84


def test_gemelo_identifica_y_calcula_imagen_doble(corrida):
    esc, cfg, cal, res, inf, _ = corrida
    g = inf["gemelo"]
    assert all(g["sensibilidad"]["identificables"]) and g["sensibilidad"]["rango"] == 5
    assert np.isfinite(g["imagen_doble_arcmin"]) and np.isfinite(g["cuna_optima_mrad"])
    assert all(s > 0 for s in g["sensibilidad"]["sigma"])


def test_archivos_de_salida(corrida):
    *_, carpeta = corrida
    for f in ("informe.json", "mapas.npz", "mapas.png"):
        assert os.path.exists(os.path.join(carpeta, f))
    json.load(open(os.path.join(carpeta, "informe.json"), encoding="utf-8"))


def test_punto_danado_se_marca_y_se_excluye():
    cfg, demo = configuracion_demo()
    esc = D.generar_escenario(n_lado=9, ruido=0.01, semilla=3)
    rng = np.random.default_rng(0)
    esc.mediciones[4].I = 0.01 * rng.standard_normal(esc.mediciones[4].I.size)
    k, Is = D.muestra_referencia(np.array([2100.0, 760.0, 1600.0]), 30, 0.01, 7)
    cal = calibrar_desde_interferogramas(k, Is, [2100.0, 760.0, 1600.0], [1.0, 1.0, 1.0], cfg)
    res = ejecutar(esc.mediciones, cfg, cal, D.malla_cad())
    assert 4 in res.ids_defecto and 4 not in res.ids_ok
    assert len(res.ids_ok) == 80


def test_muy_pocos_puntos_validos_detiene_el_programa():
    cfg, demo = configuracion_demo()
    esc = D.generar_escenario(n_lado=4, ruido=0.01, semilla=3)
    k, Is = D.muestra_referencia(np.array([2100.0, 760.0, 1600.0]), 30, 0.01, 7)
    cal = calibrar_desde_interferogramas(k, Is, [2100.0, 760.0, 1600.0], [1.0, 1.0, 1.0], cfg)
    with pytest.raises(ErrorDeEntrada):
        ejecutar(esc.mediciones, cfg, cal, D.malla_cad())


def test_ruta_de_datos_reales_por_archivos_coincide_con_la_directa(tmp_path):
    ruta = D.guardar_escenario(str(tmp_path), ruido=0.01, semilla=0, n_lado=11)
    res, inf = ejecutar_desde_config(ruta, verbose=False)
    esc = D.generar_escenario(n_lado=11, ruido=0.01, semilla=0)
    err = res.d_puntos - esc.verdad_d[res.ids_ok]
    assert np.all(np.sqrt((err**2).mean(axis=0)) < 0.5)
    assert inf["validacion_cruzada"]["ok"]


def test_config_incompleta_da_error_claro(tmp_path):
    ruta = tmp_path / "c.json"
    ruta.write_text(json.dumps({"mediciones": "x.npz"}), encoding="utf-8")
    with pytest.raises(ErrorDeEntrada):
        ejecutar_desde_config(str(ruta), verbose=False)
