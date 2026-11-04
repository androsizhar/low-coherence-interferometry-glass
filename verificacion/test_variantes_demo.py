import numpy as np
import pytest

from motor.analisis import demo

UM = 1000.0
X, Y = np.meshgrid(np.linspace(-43, 43, 41) * UM, np.linspace(-43, 43, 41) * UM)
NOMINAL = np.array([2100.0, 760.0, 1600.0])
TOLERANCIA = np.array([10.0, 25.0, 10.0])


def test_variantes_disponibles():
    assert set(demo.VARIANTES) == {"base", "conforme", "defecto"}


def test_pieza_conforme_cumple_tolerancias():
    d = np.array(demo.espesores_conforme(X, Y))
    assert (np.abs(d - NOMINAL).max(axis=(0, 1)) < TOLERANCIA).all()


def test_pieza_conforme_mantiene_la_cuna_cerca_de_lo_nominal():
    d2 = demo.espesores_conforme(X, Y)[..., 1]
    pendiente = np.gradient(d2, X[0] / UM, axis=1)
    assert 0.25 < pendiente.mean() < 0.45


def test_pieza_con_defecto_supera_la_tolerancia_en_el_vidrio_exterior():
    d = np.array(demo.espesores_defecto(X, Y))
    assert abs(d[..., 0] - NOMINAL[0]).max() > TOLERANCIA[0]


def test_defecto_deja_sin_alterar_parte_de_la_pieza():
    conforme = np.array(demo.espesores_conforme(X, Y))
    defecto = np.array(demo.espesores_defecto(X, Y))
    cambia = np.abs(defecto - conforme).max(axis=-1) > 0.5
    assert 0 < cambia.mean() < 0.8


@pytest.mark.parametrize("variante", ["conforme", "defecto"])
def test_los_perfiles_son_deterministas(variante):
    espesores, delta = demo.VARIANTES[variante]
    assert np.array_equal(np.array(espesores(X, Y)), np.array(espesores(X, Y)))
    assert np.array_equal(delta(X, Y), delta(X, Y))


def test_variante_defecto_agrega_desviacion_frente_al_cad():
    assert np.abs(demo.delta_defecto(X, Y)).max() > np.abs(demo.delta_conforme(X, Y)).max()
