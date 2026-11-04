import numpy as np
import pytest

from motor.geometria.cad import malla_desde_grilla
from motor.geometria.superficies import (
    ProcesoGaussiano2D,
    desviacion_contra_cad,
    puntos_interfaces,
)
from motor.medicion.lectura import ErrorDeEntrada

UM = 1e3


def _campo(x, y):
    return 3.0 * np.sin(2 * np.pi * x / 60) * np.cos(2 * np.pi * y / 80) + 0.35 * x


def _gradiente(x, y):
    gx = 3.0 * (2 * np.pi / 60) * np.cos(2 * np.pi * x / 60) * np.cos(2 * np.pi * y / 80) + 0.35
    gy = -3.0 * (2 * np.pi / 80) * np.sin(2 * np.pi * x / 60) * np.sin(2 * np.pi * y / 80)
    return gx, gy


@pytest.fixture(scope="module")
def gp():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 100, (160, 2))
    sig = np.full(len(X), 0.8)
    y = _campo(X[:, 0], X[:, 1]) + rng.normal(0, sig)
    return ProcesoGaussiano2D().ajustar(X, y, sig)


def test_gp_reduce_el_error_y_esta_calibrado(gp):
    gx, gy = np.meshgrid(np.linspace(10, 90, 25), np.linspace(10, 90, 25))
    Xs = np.column_stack([gx.ravel(), gy.ravel()])
    m, s = gp.predecir(Xs)
    err = m - _campo(Xs[:, 0], Xs[:, 1])
    assert np.sqrt(np.mean(err**2)) < 0.5
    cobertura1 = np.mean(np.abs(err) < s)
    cobertura2 = np.mean(np.abs(err) < 2 * s)
    assert 0.55 < cobertura1 < 0.88, cobertura1
    assert cobertura2 > 0.90, cobertura2


def test_gradiente_es_derivada_de_la_media(gp):
    Xs = np.array([[30.0, 40.0], [55.0, 62.0], [80.0, 20.0]])
    _, _, g, cov = gp.predecir(Xs, gradiente=True)
    h = 1e-4
    for a in range(2):
        e = np.zeros(2)
        e[a] = h
        num = (gp.predecir(Xs + e)[0] - gp.predecir(Xs - e)[0]) / (2 * h)
        assert np.allclose(g[:, a], num, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_gradiente_cubre_la_verdad(gp):
    gx, gy = np.meshgrid(np.linspace(15, 85, 15), np.linspace(15, 85, 15))
    Xs = np.column_stack([gx.ravel(), gy.ravel()])
    _, _, g, cov = gp.predecir(Xs, gradiente=True)
    tx, ty = _gradiente(Xs[:, 0], Xs[:, 1])
    zx = (g[:, 0] - tx) / np.sqrt(cov[:, 0, 0])
    zy = (g[:, 1] - ty) / np.sqrt(cov[:, 1, 1])
    assert np.mean(np.abs(zx) < 2.5) > 0.9 and np.mean(np.abs(zy) < 2.5) > 0.9


def test_hiperparametros_razonables(gp):
    assert 8.0 < gp.ell < 80.0
    assert 0.5 < gp.s < 20.0


def test_tendencia_lineal_exacta():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 100, (60, 2))
    y = 0.35 * X[:, 0] - 0.1 * X[:, 1] + 760.0
    g = ProcesoGaussiano2D(ajustar_nugget=False).ajustar(X, y, np.full(60, 0.05))
    _, _, grad, cov = g.predecir(np.array([[50.0, 50.0]]), gradiente=True)
    assert np.allclose(grad[0], [0.35, -0.1], atol=2e-4)
    assert np.sqrt(cov[0, 0, 0]) < 1e-2


def test_gp_valida_entradas():
    X = np.random.default_rng(0).uniform(0, 1, (20, 2))
    with pytest.raises(ErrorDeEntrada):
        ProcesoGaussiano2D().ajustar(X[:5], np.zeros(5), np.ones(5))
    with pytest.raises(ErrorDeEntrada):
        ProcesoGaussiano2D().ajustar(X, np.zeros(20), np.zeros(20))
    with pytest.raises(ErrorDeEntrada):
        ProcesoGaussiano2D().ajustar(X, np.zeros(19), np.ones(20))


def test_puntos_interfaces_y_desviacion_recuperan_delta_conocida():
    xs = np.linspace(-50 * UM, 50 * UM, 61)
    X, Y = np.meshgrid(xs, xs)
    Z = X**2 / (2 * 1500 * UM)
    exterior = malla_desde_grilla(X, Y, Z)
    from motor.geometria.cad import interfaces_nominales

    d_nom = [2100.0, 760.0, 1600.0]
    cad = interfaces_nominales(exterior, d_nom)
    rng = np.random.default_rng(0)
    q = exterior.muestrear(80, rng)
    _, _, ti = exterior.proyectar(q)
    n_out = exterior.normales_cara()[ti]
    n_in = -n_out
    d_true = np.array(d_nom) + np.array([3.0, -2.0, 1.5])
    P = puntos_interfaces(q, n_in, np.tile(d_true, (len(q), 1)))
    assert P.shape == (4, 80, 3)
    dev, _ = desviacion_contra_cad(P[3], cad[3])
    esperado = -(3.0 - 2.0 + 1.5)
    assert abs(np.mean(dev) - esperado) < 0.5, np.mean(dev)


def test_puntos_interfaces_valida():
    with pytest.raises(ErrorDeEntrada):
        puntos_interfaces(np.zeros((3, 3)), np.ones((3, 3)), np.ones((3, 2)))


def test_gp_rechaza_sigma_infinita_colinealidad_y_prediccion_sin_ajuste():
    X = np.column_stack([np.arange(10.0), np.arange(10.0)])
    for coords, sigma in [
        (X, np.ones(10)),
        (np.random.default_rng(2).normal(size=(10, 2)), np.full(10, np.inf)),
    ]:
        with pytest.raises(ErrorDeEntrada):
            ProcesoGaussiano2D().ajustar(coords, np.zeros(10), sigma)
    with pytest.raises(ErrorDeEntrada):
        ProcesoGaussiano2D().predecir([[0.0, 0.0]])


def test_puntos_interfaces_rechaza_espesores_invalidos():
    q = np.zeros((3, 3))
    n = np.tile([0.0, 0.0, -1.0], (3, 1))
    for d in (np.ones(3), np.full((3, 2), np.nan), np.full((3, 2), -1.0)):
        with pytest.raises(ErrorDeEntrada):
            puntos_interfaces(q, n, d)
