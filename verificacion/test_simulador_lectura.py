import numpy as np
import pytest

from motor.medicion.lectura import (
    ErrorDeEntrada,
    Instrumento,
    Medicion,
    remuestrear_a_k_uniforme,
)
from motor.medicion.materiales import BK7, PVB_SUPUESTO
from motor.medicion.modelo_directo import Capa
from motor.medicion.simulador import (
    fuente_gauss,
    interferograma,
    reflexion_tmm,
    rejilla_k,
)

CAPAS = [Capa(BK7, 2100.0), Capa(PVB_SUPUESTO, 760.0), Capa(BK7, 1600.0)]


def test_tmm_una_interfaz_coincide_con_fresnel():
    k = rejilla_k(N=4096)
    r = reflexion_tmm([Capa(BK7, 3000.0)], k)
    n = BK7.n(2 * np.pi / k)
    r_front = (1 - n) / (1 + n)
    assert abs(np.mean(np.abs(r) ** 2) - np.mean(r_front**2) * 2) < 0.02


def test_tmm_conserva_energia_sin_absorcion():
    k = rejilla_k(N=2048)
    r = reflexion_tmm(CAPAS, k)
    assert np.all(np.abs(r) < 1.0)


def test_tmm_capa_cuarto_de_onda_antirreflejante():
    from motor.medicion.materiales import Cauchy

    lam0 = 1.55
    m = Cauchy(A=1.5)
    k = np.array([2 * np.pi / lam0])
    d_media_onda = lam0 / 2 / 1.5
    r = reflexion_tmm([Capa(m, d_media_onda)], k)
    assert abs(r[0]) < 1e-12


def test_interferograma_forma_y_validez():
    k = rejilla_k(N=2**13)
    S = fuente_gauss(k)
    I = interferograma(CAPAS, k, S, offset=12000.0)
    assert I.shape == k.shape and np.all(np.isfinite(I))
    assert S.max() == 1.0


def test_remuestreo_lambda_a_k():
    lam = np.linspace(1.5, 1.6, 4000)
    k_ref = 2 * np.pi / lam
    I = np.cos(k_ref * 500.0)
    k, Iu = remuestrear_a_k_uniforme(lam, I)
    assert np.allclose(np.diff(k), np.diff(k).mean(), rtol=1e-9)
    assert np.max(np.abs(Iu - np.cos(k * 500.0))) < 1e-3


def test_remuestreo_rechaza_basura():
    with pytest.raises(ErrorDeEntrada):
        remuestrear_a_k_uniforme(np.array([1.5, 1.5, 1.6]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ErrorDeEntrada):
        remuestrear_a_k_uniforme(np.array([-1.0, 1.5, 1.6]), np.array([1.0, 2.0, 3.0]))


def test_validacion_medicion_y_instrumento():
    k = rejilla_k(N=2048)
    ok = Medicion(k=k, I=np.ones_like(k), posicion=np.zeros(3), normal=np.array([0, 0, 1.0]))
    ok.validar()
    malo = Medicion(
        k=k, I=np.ones_like(k) * np.nan, posicion=np.zeros(3), normal=np.array([0, 0, 1.0])
    )
    with pytest.raises(ErrorDeEntrada):
        malo.validar()
    with pytest.raises(ErrorDeEntrada):
        Medicion(
            k=k, I=np.ones_like(k), posicion=np.zeros(3), normal=np.array([0, 0, 2.0])
        ).validar()
    with pytest.raises(ErrorDeEntrada):
        Instrumento(lam0=1550.0, offset=1.0).validar()


def test_tmm_pelicula_unica_contra_serie_fabry_perot_analitica():
    from motor.medicion.materiales import Cauchy

    k = rejilla_k(N=101)
    n, d = 1.6, 3.1
    r01, r12 = (1 - n) / (1 + n), (n - 1) / (n + 1)
    fase = np.exp(-2j * k * n * d)
    analitico = (r01 + r12 * fase) / (1 + r01 * r12 * fase)
    assert np.allclose(reflexion_tmm([Capa(Cauchy(A=n), d)], k), analitico, atol=1e-13)


def test_simulador_no_silencia_ruido_negativo_ni_espectro_invalido():
    k = rejilla_k(N=128)
    with pytest.raises(ValueError):
        interferograma(CAPAS, k, fuente_gauss(k), offset=12000, ruido=-1)
    with pytest.raises(ValueError):
        interferograma(CAPAS, k, np.ones(4), offset=12000)
    with pytest.raises(ValueError):
        fuente_gauss(np.array([1.0]))
    with pytest.raises(ValueError):
        rejilla_k(N=1)
    with pytest.raises(ValueError):
        reflexion_tmm(CAPAS, np.array([1.0, np.nan]))


def test_ruido_gaussiano_es_reproducible_y_usa_sigma_de_intensidad():
    k = rejilla_k(N=16384)
    S = fuente_gauss(k)
    limpio = interferograma(CAPAS, k, S, offset=12000)
    sigma = 0.01
    ruidoso = interferograma(CAPAS, k, S, offset=12000, ruido=sigma, rng=np.random.default_rng(123))
    repetido = interferograma(
        CAPAS, k, S, offset=12000, ruido=sigma, rng=np.random.default_rng(123)
    )
    assert np.array_equal(ruidoso, repetido)
    assert np.std(ruidoso - limpio) == pytest.approx(sigma, rel=0.03)
