import numpy as np
import pytest

from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.materiales import BK7, PVB_SUPUESTO
from motor.medicion.modelo_directo import Capa, camino_optico_grupo, g_conf
from motor.medicion.picos import detectar_picos
from motor.medicion.simulador import fuente_gauss, interferograma, rejilla_k
from motor.metrologia.calibracion import calibrar, validar_con_muestra
from motor.metrologia.espesor import espesor_calibrado, separar_confocal

LAM0, OFFSET = 1.55, 12000.0
GRUPOS = ["vidrio", "pvb", "vidrio"]
K = rejilla_k(N=2**14)
S = fuente_gauss(K, LAM0)


def _ventanas(d):
    capas = [Capa(BK7, d[0]), Capa(PVB_SUPUESTO, d[1]), Capa(BK7, d[2])]
    a = camino_optico_grupo(capas, LAM0)
    L = 2 * (OFFSET + np.concatenate([[0.0], np.cumsum(a)]))
    return [(l - 250, l + 250) for l in L]


def _medir(d, ruido, seed, mats=(BK7, PVB_SUPUESTO, BK7), tol_ventana_d=None):
    capas = [Capa(m, x) for m, x in zip(mats, d)]
    I = interferograma(capas, K, S, OFFSET, ruido=ruido, rng=np.random.default_rng(seed))
    return detectar_picos(
        K, I, 4, ventanas=_ventanas(d if tol_ventana_d is None else tol_ventana_d)
    )


D_REF = np.array([2100.0, 760.0, 1600.0])
SD_REF = np.array([1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def cal():
    reps = [_medir(D_REF, 0.01, s) for s in range(30)]
    return calibrar(reps, D_REF, SD_REF, GRUPOS)


def test_calibracion_n_g_cercano_al_verdadero(cal):
    ng_true = np.array([float(m.n_grupo(LAM0)) for m in (BK7, PVB_SUPUESTO, BK7)])
    assert cal.ok, cal.banderas
    assert np.all(np.abs(cal.ng - ng_true) < 5 * cal.sigma_ng), (cal.ng, ng_true, cal.sigma_ng)
    assert cal.ng[0] == cal.ng[2]
    assert np.all(cal.sigma_a < 0.6)


def test_espesor_pieza_nueva_dentro_de_incertidumbre(cal):
    d_true = np.array([2060.0, 790.0, 1640.0])
    errs = []
    for s in range(40, 60):
        r = _medir(d_true, 0.01, s)
        e = espesor_calibrado(r.caminos_capas(), cal.sigma_a, cal)
        errs.append(e.d - d_true)
        assert np.all(e.sigma_total > e.sigma_aleatoria)
    errs = np.array(errs)
    assert np.all(np.abs(errs.mean(axis=0)) < 3 * e.sigma_sistematica)
    assert np.all(errs.std(axis=0) < 0.6)


def test_sistematico_se_propaga_si_d_ref_esta_mal():
    reps = [
        _medir(D_REF + np.array([5.0, 0.0, 0.0]), 0.01, s, tol_ventana_d=D_REF) for s in range(30)
    ]
    cal_mala = calibrar(reps, D_REF, SD_REF, GRUPOS)
    d_true = np.array([2060.0, 790.0, 1640.0])
    r = _medir(d_true, 0.01, 99)
    e = espesor_calibrado(r.caminos_capas(), cal_mala.sigma_a, cal_mala)
    err_rel = (e.d[0] - d_true[0]) / d_true[0]
    w = (D_REF[[0, 2]] / SD_REF[[0, 2]]) ** 2
    esperado = -(5.0 / D_REF[0]) * w[0] / w.sum()
    assert abs(err_rel - esperado) < 3e-4


def test_grupo_inconsistente_se_marca():
    reps = [
        _medir(D_REF + np.array([0.0, 0.0, 30.0]), 0.01, s, tol_ventana_d=D_REF) for s in range(30)
    ]
    c = calibrar(reps, D_REF, SD_REF, GRUPOS)
    assert not c.ok and any(b.startswith("inconsistencia_grupo") for b in c.banderas)


def test_validacion_cruzada_pasa_y_falla(cal):
    d2 = np.array([2050.0, 800.0, 1650.0])
    reps2 = [_medir(d2, 0.01, 200 + s) for s in range(10)]
    v = validar_con_muestra(cal, reps2, d2, SD_REF)
    assert v.ok, v.z
    v_mal = validar_con_muestra(cal, reps2, d2 + np.array([10.0, 0, 0]), SD_REF)
    assert not v_mal.ok


def test_calibracion_rechaza_entradas_malas():
    reps = [_medir(D_REF, 0.01, s) for s in range(3)]
    with pytest.raises(ErrorDeEntrada):
        calibrar(reps, D_REF, SD_REF, GRUPOS)
    reps = [_medir(D_REF, 0.01, s) for s in range(6)]
    with pytest.raises(ErrorDeEntrada):
        calibrar(reps, [2100.0, -1.0, 1600.0], SD_REF, GRUPOS)
    malo = _medir(D_REF, 0.2, 1)
    with pytest.raises(ErrorDeEntrada):
        calibrar([malo] * 6, D_REF, SD_REF, GRUPOS)


@pytest.mark.parametrize("alpha_deg", [0.0, 3.0, 7.0])
def test_confocal_recupera_n_y_d_exactos(alpha_deg):
    alpha = np.deg2rad(alpha_deg)
    for mat, d in ((BK7, 2100.0), (PVB_SUPUESTO, 760.0)):
        n = float(mat.n(LAM0))
        dg = float(mat.delta_g(LAM0))
        a = (n + dg) * d
        c = d * float(g_conf(n, alpha))
        r = separar_confocal(a, c, dg, alpha, sigma_a=0.3, sigma_c=1.0)
        assert abs(r.d - d) < 1e-6 and abs(r.n - n) < 1e-9
        assert r.cov.shape == (2, 2) and np.all(np.diag(r.cov) > 0)


def test_confocal_covarianza_coincide_con_montecarlo():
    mat, d, alpha = BK7, 2100.0, np.deg2rad(5)
    n = float(mat.n(LAM0))
    dg = float(mat.delta_g(LAM0))
    a0, c0 = (n + dg) * d, d * float(g_conf(n, alpha))
    rng = np.random.default_rng(0)
    est = np.array(
        [
            [
                separar_confocal(a0 + rng.normal(0, 0.3), c0 + rng.normal(0, 1.0), dg, alpha).d,
                separar_confocal(a0 + rng.normal(0, 0.3), c0 + rng.normal(0, 1.0), dg, alpha).n,
            ]
            for _ in range(1500)
        ]
    )
    cov = separar_confocal(a0, c0, dg, alpha, 0.3, 1.0).cov
    assert abs(est[:, 0].std() / np.sqrt(cov[0, 0]) - 1) < 0.1
    assert abs(est[:, 1].std() / np.sqrt(cov[1, 1]) - 1) < 0.1


def test_confocal_entradas_invalidas():
    with pytest.raises(ErrorDeEntrada):
        separar_confocal(-1.0, 1000.0, 0.005, 0.0)
