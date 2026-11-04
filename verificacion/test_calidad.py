import numpy as np
import pytest
from scipy.stats import norm

from motor.analisis.calidad import cuna_pvb, decidir, probabilidad_fuera_de_tolerancia
from motor.geometria.superficies import ProcesoGaussiano2D
from motor.medicion.lectura import ErrorDeEntrada


def test_probabilidad_valores_conocidos():
    p = probabilidad_fuera_de_tolerancia(760.0, 2.0, 760.0, 3.0)
    assert p == pytest.approx(2 * (1 - norm.cdf(1.5)), rel=1e-12)
    p = probabilidad_fuera_de_tolerancia(763.0, 1.0, 760.0, 3.0)
    assert p == pytest.approx(0.5 + norm.cdf(-6.0), abs=1e-9)
    assert probabilidad_fuera_de_tolerancia(762.9, 0.0, 760.0, 3.0) == 0.0
    assert probabilidad_fuera_de_tolerancia(763.1, 0.0, 760.0, 3.0) == 1.0


def test_sistematica_aumenta_la_probabilidad():
    p0 = probabilidad_fuera_de_tolerancia(760.0, 0.5, 760.0, 1.5)
    p1 = probabilidad_fuera_de_tolerancia(760.0, 0.5, 760.0, 1.5, sigma_sistematica=1.0)
    assert p1 > p0


def test_tolerancia_asimetrica_y_validaciones():
    p = probabilidad_fuera_de_tolerancia(758.0, 0.1, 760.0, (1.0, 5.0))
    assert p > 0.99
    with pytest.raises(ErrorDeEntrada):
        probabilidad_fuera_de_tolerancia(760.0, -1.0, 760.0, 3.0)
    with pytest.raises(ErrorDeEntrada):
        probabilidad_fuera_de_tolerancia(760.0, 1.0, 760.0, -3.0)
    with pytest.raises(ErrorDeEntrada):
        decidir(np.array([0.1]), umbral=1.5)


def test_decision():
    d = decidir(np.array([0.01, 0.2, 0.049, 0.9]), 0.05)
    assert list(d.falla) == [False, True, False, True] and d.fraccion_falla == 0.5


def _gp_cuna(w_true_mrad, ang_deg, seed, n=15, sigma=0.3):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.linspace(0, 100, n), np.linspace(0, 100, n))
    X = np.column_stack([gx.ravel(), gy.ravel()])
    s = np.array([np.cos(np.deg2rad(ang_deg)), np.sin(np.deg2rad(ang_deg))])
    d = (
        760.0
        + w_true_mrad * (X @ s)
        + 1.5 * np.cos(2 * np.pi * (X[:, 0] + X[:, 1]) / 90)
        + rng.normal(0, sigma, len(X))
    )
    return ProcesoGaussiano2D().ajustar(X, d, np.full(len(X), sigma)), s


def test_cuna_recuperada_y_sigma_calibrada():
    z, ratios = [], []
    for seed in range(12):
        gp, s = _gp_cuna(0.35, 0.0, seed)
        Xs = np.array([[25.0, 50.0], [50.0, 50.0], [75.0, 50.0]])
        c = cuna_pvb(gp, Xs)
        tx = 0.35 + 1.5 * (-np.sin(2 * np.pi * (Xs[:, 0] + Xs[:, 1]) / 90)) * (2 * np.pi / 90)
        z.append((c.w - tx) / c.sigma_w)
        ratios.append(c.sigma_w.mean())
    z = np.array(z)
    assert np.mean(np.abs(z) < 3) > 0.9
    assert 0.5 < np.sqrt(np.mean(z**2)) < 1.8


def test_direccion_de_cuna_se_estima():
    gp, s = _gp_cuna(0.40, 30.0, seed=1)
    Xs = np.column_stack([np.linspace(20, 80, 7), np.linspace(20, 80, 7)])
    c = cuna_pvb(gp, Xs)
    ang = np.degrees(np.arctan2(c.direccion[1], c.direccion[0]))
    assert abs(ang - 30.0) < 8.0
    assert abs(np.mean(c.w) - 0.40) < 0.05


def test_direccion_invalida():
    gp, _ = _gp_cuna(0.3, 0.0, 0)
    with pytest.raises(ErrorDeEntrada):
        cuna_pvb(gp, np.array([[50.0, 50.0]]), direccion=[0.0, 0.0])


@pytest.mark.parametrize(
    "mu,sigma,sis,tol",
    [
        (np.nan, 1.0, 0.0, 1.0),
        (0.0, np.inf, 0.0, 1.0),
        (0.0, 1.0, -1.0, 1.0),
        (0.0, 1.0, 0.0, np.nan),
    ],
)
def test_calidad_rechaza_no_finitos_y_sistematicas_negativas(mu, sigma, sis, tol):
    with pytest.raises(ErrorDeEntrada):
        probabilidad_fuera_de_tolerancia(mu, sigma, 0.0, tol, sis)


def test_probabilidad_conserva_colas_muy_pequenas():
    p = probabilidad_fuera_de_tolerancia(0.0, 1.0, 0.0, 10.0)
    assert p > 0
    assert p == pytest.approx(2 * norm.sf(10.0), rel=1e-12)


@pytest.mark.parametrize("p", [[], [np.nan], [np.inf], [-0.1], [1.1]])
def test_decision_no_aprueba_probabilidades_invalidas(p):
    with pytest.raises(ErrorDeEntrada):
        decidir(p)


def test_cuna_con_direccion_fija_conserva_signo():
    X = np.random.default_rng(9).uniform(0, 100, (40, 2))
    gp = ProcesoGaussiano2D(ajustar_nugget=False).ajustar(
        X, 760.0 - 0.35 * X[:, 0], np.full(40, 0.1)
    )
    c = cuna_pvb(gp, np.array([[50.0, 50.0]]), direccion=[1.0, 0.0])
    assert c.w[0] == pytest.approx(-0.35, abs=0.001)
    assert not c.direccion_estimada
