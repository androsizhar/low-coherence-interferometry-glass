import numpy as np
import pytest

from motor.analisis.imagen_doble import (
    cuna_optima,
    cuna_optima_laminado,
    separacion_imagen_doble,
    separacion_laminado,
    separacion_por_cuna_analitica,
    trazar_fantasma,
    trazar_fantasma_laminado,
)
from motor.medicion.lectura import ErrorDeEntrada

N, TH = 1.52, np.deg2rad(62.0)
T = 4460.0
LEJOS = 1e11


def test_sin_cuna_y_muy_lejos_no_hay_separacion_angular():
    assert abs(separacion_imagen_doble(T, 0.0, N, TH, LEJOS, LEJOS)) < 1e-7


@pytest.mark.parametrize("Rp,Rv", [(2e6, 2e6), (1e6, 4e6), (4e6, 1e6), (3e5, 9e5)])
def test_efecto_de_la_cuna_coincide_con_formula_analitica(Rp, Rv):
    base = separacion_imagen_doble(T, 0.0, N, TH, Rp, Rv)
    for w in (0.2e-3, 0.5e-3, -0.35e-3):
        num = separacion_imagen_doble(T, w, N, TH, Rp, Rv) - base
        ana = separacion_por_cuna_analitica(w, N, TH, Rp, Rv)
        assert num == pytest.approx(ana, rel=1e-2), (Rp, Rv, w, num, ana)


def test_el_rayo_fantasma_llega_al_ojo():
    tr = trazar_fantasma(T, 0.4e-3, N, TH, 600e3, 900e3)
    assert tr.fallo_en_ojo < 1e-6


def test_separacion_lineal_en_la_cuna():
    a = separacion_imagen_doble(T, 0.2e-3, N, TH, 600e3, 900e3)
    b = separacion_imagen_doble(T, 0.4e-3, N, TH, 600e3, 900e3)
    c = separacion_imagen_doble(T, 0.0, N, TH, 600e3, 900e3)
    assert (b - c) == pytest.approx(2 * (a - c), rel=5e-3)


def test_cuna_optima_anula_y_escala_con_el_espesor():
    w1 = cuna_optima(T, N, TH, 600e3, 900e3)
    assert abs(separacion_imagen_doble(T, w1, N, TH, 600e3, 900e3)) < 1e-11
    w2 = cuna_optima(2 * T, N, TH, 600e3, 900e3)
    assert w2 == pytest.approx(2 * w1, rel=3e-2)
    assert 0.05e-3 < abs(w1) < 2e-3


def test_entradas_invalidas():
    with pytest.raises(ErrorDeEntrada):
        separacion_imagen_doble(-1.0, 0.0, N, TH, 1e5, 1e5)
    with pytest.raises(ErrorDeEntrada):
        separacion_imagen_doble(T, 0.0, 0.9, TH, 1e5, 1e5)
    with pytest.raises(ErrorDeEntrada):
        separacion_imagen_doble(T, 0.0, N, np.deg2rad(95), 1e5, 1e5)


@pytest.mark.parametrize("w", [0.0, 0.4e-3, -0.3e-3])
def test_multicapa_de_indices_iguales_se_reduce_a_placa_homogenea(w):
    d = [2100.0, 760.0, 1600.0]
    laminado = separacion_laminado(d, [N] * 3, w, TH, 600e3, 900e3)
    placa = separacion_imagen_doble(sum(d), w, N, TH, 600e3, 900e3)
    assert laminado == pytest.approx(placa, abs=1e-12)


def test_laminado_paralelo_contra_ecuacion_independiente_de_desplazamiento_lateral():
    from scipy.optimize import brentq

    d, n = np.array([2100.0, 760.0, 1600.0]), np.array([1.6, 1.3, 1.7])
    Rp, Rv = 600e3, 900e3
    dx = (Rp + Rv) * np.sin(TH)
    dy = (Rp + Rv) * np.cos(TH)

    def cierre(beta):
        s = np.sin(beta)
        return dy * np.tan(beta) + 2 * np.sum(d * s / np.sqrt(n**2 - s**2)) - dx

    beta = brentq(cierre, 0, TH)
    resultado = separacion_laminado(d, n, 0.0, TH, Rp, Rv)
    assert resultado == pytest.approx(TH - beta, abs=1e-12)


def test_cuna_multicapa_y_trayectoria_fisica_hasta_ojo():
    d, n = [2100.0, 760.0, 1600.0], [1.52, 1.48, 1.52]
    w = cuna_optima_laminado(d, n, TH, 600e3, 900e3)
    assert abs(separacion_laminado(d, n, w, TH, 600e3, 900e3)) < 1e-11
    tr = trazar_fantasma_laminado(d, n, 0.4e-3, TH, 600e3, 900e3)
    assert tr.fallo_en_ojo < 1e-6
    assert tr.d_salida[1] > 0


@pytest.mark.parametrize("w", [np.nan, np.inf, np.pi / 2])
def test_cuna_no_finita_o_vertical_se_rechaza(w):
    with pytest.raises(ErrorDeEntrada):
        trazar_fantasma(T, w, N, TH, 600e3, 900e3)


def test_hud_laminado_no_acepta_recetas_de_otro_numero_de_capas():
    with pytest.raises(ErrorDeEntrada):
        separacion_laminado([2000, 2000], [N, N], 0, TH, 600e3, 900e3)
    with pytest.raises(ErrorDeEntrada):
        separacion_por_cuna_analitica(0.001, N, TH, Rp=600e3)
