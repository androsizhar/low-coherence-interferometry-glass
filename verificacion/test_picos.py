import numpy as np
import pytest

from motor.medicion.materiales import BK7, PVB_SUPUESTO, Desplazado
from motor.medicion.modelo_directo import Capa, camino_optico_grupo
from motor.medicion.picos import detectar_picos
from motor.medicion.simulador import fuente_gauss, interferograma, rejilla_k

LAM0 = 1.55
CAPAS = [Capa(BK7, 2100.0), Capa(PVB_SUPUESTO, 760.0), Capa(BK7, 1600.0)]
OFFSET = 12000.0


def _ventanas(capas, tol_rel=0.05):
    a = camino_optico_grupo(capas, LAM0)
    Z = np.concatenate([[0.0], np.cumsum(a)])
    L = 2 * (OFFSET + Z)
    return [(l - tol_rel * L[0] * 0.3, l + tol_rel * L[0] * 0.3) for l in L]


def _medir(capas, ruido=0.0, seed=0, ventanas=True):
    k = rejilla_k(N=2**14)
    S = fuente_gauss(k, LAM0)
    I = interferograma(capas, k, S, OFFSET, ruido=ruido, rng=np.random.default_rng(seed))
    v = _ventanas(CAPAS) if ventanas else None
    return detectar_picos(k, I, n_picos=len(capas) + 1, ventanas=v, L_min=2 * OFFSET - 3000)


def test_sin_ruido_recupera_caminos_con_ventanas():
    r = _medir(CAPAS)
    assert r.ok, r.banderas
    esperado = camino_optico_grupo(CAPAS, LAM0)
    err = r.caminos_capas() - esperado
    assert np.max(np.abs(err)) < 0.05, err


def test_sin_ruido_sin_ventanas_en_zona_de_cruce():
    r = _medir(CAPAS, ventanas=False)
    assert r.ok, r.banderas
    err = r.caminos_capas() - camino_optico_grupo(CAPAS, LAM0)
    assert np.max(np.abs(err)) < 0.05


def test_ruido_1pct_montecarlo_y_sigma_predicha():
    errs, sigs = [], []
    for s in range(20):
        r = _medir(CAPAS, ruido=0.01, seed=s)
        assert r.ok, r.banderas
        errs.append(r.caminos_capas() - camino_optico_grupo(CAPAS, LAM0))
        sigs.append(r.sigma_caminos())
    errs, sigs = np.array(errs), np.array(sigs)
    emp = errs.std(axis=0)
    assert np.all(emp < 0.6), emp
    ratio = sigs.mean(axis=0) / emp
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_interfaz_inexistente_se_marca_no_se_inventa():
    pvb_igual = Desplazado(BK7, 0.0)
    capas = [Capa(BK7, 2100.0), Capa(pvb_igual, 760.0), Capa(BK7, 1600.0)]
    k = rejilla_k(N=2**14)
    I = interferograma(
        capas, k, fuente_gauss(k, LAM0), OFFSET, ruido=0.01, rng=np.random.default_rng(1)
    )
    r = detectar_picos(k, I, n_picos=4, ventanas=_ventanas(CAPAS))
    assert not r.ok
    assert any(b.startswith("snr_bajo") for b in r.banderas)


def test_ruido_alto_se_marca():
    r = _medir(CAPAS, ruido=0.2, seed=3)
    assert not r.ok


def test_ventana_mal_puesta_se_marca():
    k = rejilla_k(N=2**14)
    I = interferograma(CAPAS, k, fuente_gauss(k, LAM0), OFFSET)
    v = [(1.0e5, 1.1e5)] * 4
    r = detectar_picos(k, I, n_picos=4, ventanas=v)
    assert not r.ok


def test_entrada_invalida():
    from motor.medicion.lectura import ErrorDeEntrada

    k = rejilla_k(N=2**14)
    with pytest.raises(ErrorDeEntrada):
        detectar_picos(k[:10], np.ones(10), 4)
    with pytest.raises(ErrorDeEntrada):
        detectar_picos(k, np.ones_like(k), 4, ventanas=[(0, 1)])
