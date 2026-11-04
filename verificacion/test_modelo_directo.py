import numpy as np
import pytest

from motor.medicion.materiales import (
    BK7,
    PVB_SUPUESTO,
    SODOCALCICO_SUPUESTO,
    Desplazado,
)
from motor.medicion.modelo_directo import (
    Capa,
    amplitudes_pico,
    g_conf,
    jacobiano,
    observables,
    r_interfaces,
)

LAM = 1.55


@pytest.mark.parametrize("mat", [BK7, PVB_SUPUESTO, SODOCALCICO_SUPUESTO, Desplazado(BK7, 0.01)])
def test_n_grupo_es_n_menos_lam_dn_dlam(mat):
    h = 1e-5
    dn = (mat.n(LAM + h) - mat.n(LAM - h)) / (2 * h)
    esperado = mat.n(LAM) - LAM * dn
    assert abs(mat.n_grupo(LAM) - esperado) < 1e-8


def test_dispersion_normal_ng_mayor_que_n():
    for mat in (BK7, PVB_SUPUESTO, SODOCALCICO_SUPUESTO):
        assert mat.delta_g(LAM) > 0


def test_bk7_valor_conocido():
    assert abs(float(BK7.n(0.58756)) - 1.5168) < 1e-4


def test_g_conf_limite_paraxial_y_apertura():
    n = 1.5
    assert abs(g_conf(n, 0.0) - 1 / n) < 1e-14
    assert g_conf(n, np.deg2rad(5)) > 1 / n * 0.99
    assert abs(g_conf(n, np.deg2rad(5)) - 1 / n) > 1e-4
    with pytest.raises(ValueError):
        g_conf(1.0, np.deg2rad(95))
    with pytest.raises(ValueError):
        g_conf(1.0, np.pi / 2 - 1e-12)


def test_observables_forma():
    capas = [Capa(BK7, 2100.0), Capa(PVB_SUPUESTO, 760.0), Capa(BK7, 2100.0)]
    a, c = observables(capas, LAM, np.deg2rad(3))
    assert a.shape == (3,) and c.shape == (3,)
    assert np.allclose(a, [float(m.material.n_grupo(LAM)) * m.d for m in capas])
    assert np.all(a > c)


def test_reflexion_fresnel_y_signos():
    capas = [Capa(BK7, 2100.0), Capa(PVB_SUPUESTO, 760.0), Capa(BK7, 2100.0)]
    r = r_interfaces(capas, LAM)
    n = float(BK7.n(LAM))
    assert abs(r[0] - (1 - n) / (1 + n)) < 1e-12 and r[0] < 0
    assert r[1] > 0 and r[2] < 0 and abs(r[1] + r[2]) < 1e-12
    A = amplitudes_pico(capas, LAM)
    assert abs(A[0]) == pytest.approx(abs(r[0]))
    assert abs(A[1]) < abs(A[0]) / 10


def test_jacobiano_numerico_contra_analitico():
    n, d = 1.5, 2000.0
    f = lambda p: np.array([(p[1] + 0.005) * p[0], p[0] * float(g_conf(p[1], 0.0))])
    J = jacobiano(f, np.array([d, n]))
    esperado = np.array([[n + 0.005, d], [1 / n, -d / n**2]])
    assert np.allclose(J, esperado, rtol=1e-6)


def test_capa_rechaza_espesor_no_positivo():
    with pytest.raises(ValueError):
        Capa(BK7, 0.0)


@pytest.mark.parametrize("d", [np.inf, np.nan, -1])
def test_capa_rechaza_espesor_no_finito(d):
    with pytest.raises(ValueError):
        Capa(BK7, d)


@pytest.mark.parametrize("lam", [0.0, -1.0, np.nan, np.inf])
def test_materiales_no_calculan_indices_con_lambda_invalida(lam):
    for m in (BK7, PVB_SUPUESTO):
        with pytest.raises(ValueError):
            m.n(lam)
        with pytest.raises(ValueError):
            m.n_grupo(lam)


def test_sellmeier_no_trunca_coeficientes_y_rechaza_polos():
    from motor.medicion.materiales import Sellmeier

    with pytest.raises(ValueError, match="igual longitud"):
        Sellmeier(Bs=(1, 2), Cs=(0.1,))
    with pytest.raises(ValueError, match="polo"):
        Sellmeier(Bs=(1,), Cs=(0.25,)).n(0.5)
    with pytest.raises(ValueError):
        Desplazado(BK7, np.inf)


def test_modelo_confocal_rechaza_indice_no_fisico():
    for n in (np.nan, np.inf, -1):
        with pytest.raises(ValueError):
            g_conf(n, 0)
