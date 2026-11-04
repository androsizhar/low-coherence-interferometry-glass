import numpy as np
import pytest

from motor.medicion.lectura import (
    ErrorDeEntrada,
    Instrumento,
    Medicion,
    remuestrear_a_k_uniforme,
)
from motor.medicion.picos import ResultadoPicos, detectar_picos
from motor.metrologia.calibracion import Calibracion, calibrar, validar_con_muestra
from motor.metrologia.espesor import espesor_calibrado, separar_confocal


def _picos(a, sigma=0):
    L = 100 + 2 * np.r_[0, np.cumsum(a)]
    n = len(L)
    return ResultadoPicos(L, L.copy(), np.ones(n), np.ones(n) * 100, np.full(n, sigma, dtype=float))


def _cal():
    return Calibracion(
        np.array([1.5, 1.6]),
        np.array([0.001, 0.002]),
        np.array([0.2, 0.3]),
        ["g", "p"],
        cov_a=np.array([[0.04, -0.03], [-0.03, 0.09]]),
    )


@pytest.mark.parametrize("campo", ["normal", "posicion"])
@pytest.mark.parametrize("valor", [np.nan, np.inf])
def test_pose_no_finita_nunca_es_aceptada(campo, valor):
    k = np.linspace(3.9, 4.1, 1024)
    kwargs = dict(k=k, I=np.ones(1024), normal=np.array([0.0, 0.0, 1.0]), posicion=np.zeros(3))
    kwargs[campo][0] = valor
    with pytest.raises(ErrorDeEntrada):
        Medicion(**kwargs).validar()


@pytest.mark.parametrize(
    "nombre,valor", [("offset", np.nan), ("sigma_ruido", -1), ("alpha", np.inf)]
)
def test_instrumento_invalido(nombre, valor):
    kwargs = dict(lam0=1.55, offset=100.0)
    kwargs[nombre] = valor
    with pytest.raises(ErrorDeEntrada):
        Instrumento(**kwargs).validar()


@pytest.mark.parametrize("modo", ["k_no_uniforme", "nan", "dos_dimensiones", "k_negativo"])
def test_detector_directo_valida_rejilla_y_finitud(modo):
    k, I = np.linspace(3.9, 4.1, 1024), np.ones(1024)
    if modo == "k_no_uniforme":
        k[100] += 1e-5
    elif modo == "nan":
        I[200] = np.nan
    elif modo == "dos_dimensiones":
        k, I = k[None], I[None]
    else:
        k -= 5
    with pytest.raises(ErrorDeEntrada):
        detectar_picos(k, I, 1)


def test_espectro_constante_no_inventa_interfaces_ni_pasa_por_snr_nan():
    k = np.linspace(3.9, 4.1, 2048)
    r = detectar_picos(k, np.ones_like(k), 2, ventanas=[(1000, 1500), (2500, 3000)])
    assert not r.ok
    assert np.all(np.isfinite(r.snr)) and np.all(r.snr == 0)


@pytest.mark.parametrize(
    "lam,I,N",
    [
        ([], [], None),
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, np.nan, 4.0], None),
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 0),
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 100.5),
    ],
)
def test_remuestreo_rechaza_vacio_nan_y_tamanos_ambiguos(lam, I, N):
    with pytest.raises(ErrorDeEntrada):
        remuestrear_a_k_uniforme(lam, I, N)


def test_caminos_adyacentes_comparten_error_con_signo_opuesto():
    C = _picos([100, 200, 300], sigma=2).cov_caminos()
    assert np.allclose(np.diag(C), 2)
    assert C[0, 1] == C[1, 2] == -1
    assert np.sum(C) == pytest.approx(2)


def test_calibracion_repeticiones_exactas_no_produce_nan():
    c = calibrar([_picos([1500, 800, 2250])] * 5, [1000, 500, 1500], [0, 0, 0], ["g", "p", "g"])
    assert c.ok and np.array_equal(c.ng, [1.5, 1.6, 1.5])
    assert np.all(np.isfinite(c.u_rel)) and np.all(c.u_rel == 0)


def test_referencias_exactas_incompatibles_se_marcan():
    c = calibrar([_picos([1500, 800, 2400])] * 5, [1000, 500, 1500], [0, 0, 0], ["g", "p", "g"])
    assert not c.ok and any(b.startswith("inconsistencia_grupo") for b in c.banderas)
    assert np.all(np.isfinite(c.ng))


def test_calibracion_covarianza_material_compartido():
    rng = np.random.default_rng(71)
    refs = [_picos(np.array([1500, 800, 2250]) + fila) for fila in rng.normal(0, 0.2, (30, 3))]
    c = calibrar(refs, [1000, 500, 1500], [1, 1, 1], ["g", "p", "g"])
    assert c.ok
    assert c.cov_ng[0, 2] == pytest.approx(c.cov_ng[0, 0])
    assert c.cov_ng[2, 2] == pytest.approx(c.cov_ng[0, 0])
    e = espesor_calibrado(np.array([1500.0, 800.0, 2250.0]), c.sigma_a, c)
    assert e.cov_sistematica[0, 2] > 0
    assert np.sum(e.cov_sistematica) > np.sum(e.sigma_sistematica**2)


def test_certificado_correlacionado_no_desaparece_al_agrupar():
    c = calibrar(
        [_picos([1500, 3000])] * 5,
        [1000, 2000],
        [1, 2],
        ["g", "g"],
        cov_d_ref=np.array([[1, 2], [2, 4]]),
    )
    assert c.ok and c.u_rel == pytest.approx([0.001, 0.001])


def test_espesor_cov_por_punto():
    c = _cal()
    a = np.array([[1500, 800], [1650, 880]])
    sigma = np.array([[0.2, 0.3], [0.4, 0.6]])
    e = espesor_calibrado(a, sigma, c, cov_a=np.stack([c.cov_a, 4 * c.cov_a]))
    assert e.cov_aleatoria.shape == (2, 2, 2)
    assert np.allclose(e.cov_aleatoria[1], 4 * e.cov_aleatoria[0])
    assert np.allclose(e.sigma_aleatoria, sigma / c.ng) and e.cov_aleatoria[0, 0, 1] < 0


@pytest.mark.parametrize(
    "a,sigma",
    [
        (np.array([np.nan, 2]), np.ones(2)),
        (np.ones(2), np.array([1.0, -1.0])),
        (np.ones(2), np.array([1.0, np.inf])),
    ],
)
def test_espesor_invalido_no_alimenta_mapas(a, sigma):
    with pytest.raises(ErrorDeEntrada):
        espesor_calibrado(a, sigma, _cal())


def test_validacion_una_adquisicion_usa_repetibilidad():
    c = _cal()
    v = validar_con_muestra(c, [_picos([1500, 800])], [1000, 500], [0.2, 0.2])
    assert v.ok and np.all(v.error == 0) and np.all(np.isfinite(v.z)) and np.all(v.sigma_d > 0)
    assert not validar_con_muestra(c, [_picos([1500, 800])], [1010, 500], [0.2, 0.2]).ok


@pytest.mark.parametrize(
    "refs,d,sd",
    [
        ([], [1000, 500], [1, 1]),
        ([_picos([1500, 800])], [np.nan, 500], [1, 1]),
        ([_picos([1500, 800])], [1000], [1]),
    ],
)
def test_validacion_rechaza_falta_de_evidencia_y_dimensiones(refs, d, sd):
    with pytest.raises(ErrorDeEntrada):
        validar_con_muestra(_cal(), refs, d, sd)


@pytest.mark.parametrize("sigma_a,sigma_c", [(0.3, 0), (0, 1.0)])
def test_confocal_propaga_si_solo_un_canal_tiene_ruido(sigma_a, sigma_c):
    r = separar_confocal(1500, 1000 / 1.5, 0, 0, sigma_a, sigma_c)
    assert r.d == pytest.approx(1000) and r.n == pytest.approx(1.5)
    assert np.all(np.diag(r.cov) > 0) and np.linalg.det(r.cov) == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize(
    "kwargs", [dict(a=np.nan), dict(sigma_c=-1.0), dict(alpha=-0.1), dict(delta_g=np.inf)]
)
def test_confocal_rechaza_entradas_no_fisicas(kwargs):
    args = dict(a=1500.0, c=1000 / 1.5, delta_g=0.0, alpha=0.0)
    args.update(kwargs)
    with pytest.raises(ErrorDeEntrada):
        separar_confocal(**args)
