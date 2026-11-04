import numpy as np
import pytest

from motor.analisis.gemelo import (
    ConfigGemelo,
    ParametrosHUD,
    covarianza_observables,
    cuna_optima_receta,
    distinguibilidad,
    imagen_doble_receta,
    predecir_imagen_doble,
    sensibilidad,
    vector_observables,
)
from motor.medicion.lectura import ErrorDeEntrada
from motor.medicion.materiales import BK7, PVB_SUPUESTO, Desplazado
from motor.medicion.modelo_directo import Capa, camino_optico_grupo

LAM0 = 1.55
RECETA = [Capa(BK7, 2100.0), Capa(PVB_SUPUESTO, 760.0), Capa(BK7, 1600.0)]


def test_vector_observables_forma_y_valores():
    cfg = ConfigGemelo(sigma_lnA=0.05)
    y = vector_observables(RECETA, cfg)
    assert y.size == 3 + 4
    assert np.allclose(y[:3], camino_optico_grupo(RECETA, LAM0))
    assert vector_observables(RECETA, ConfigGemelo(sigma_lnA=None)).size == 3


def test_recetas_identicas_no_se_distinguen():
    d = distinguibilidad(RECETA, RECETA, ConfigGemelo())
    assert d.delta == 0 and not d.distinguibles


def test_mismo_camino_optico_distinto_indice_solo_se_distingue_con_amplitud():
    mat2 = Desplazado(BK7, 0.02)
    f = float(BK7.n_grupo(LAM0)) / float(mat2.n_grupo(LAM0))
    receta2 = [Capa(mat2, 2100.0 * f), Capa(PVB_SUPUESTO, 760.0), Capa(mat2, 1600.0 * f)]
    sin_amp = distinguibilidad(RECETA, receta2, ConfigGemelo(sigma_lnA=None))
    assert sin_amp.delta < 1e-6 and not sin_amp.distinguibles
    con_amp = distinguibilidad(RECETA, receta2, ConfigGemelo(sigma_lnA=0.05))
    assert con_amp.distinguibles


def test_cambio_de_espesor_grande_se_distingue():
    receta2 = [Capa(BK7, 2100.0), Capa(PVB_SUPUESTO, 760.0 + 5.0), Capa(BK7, 1600.0)]
    assert distinguibilidad(
        RECETA, receta2, ConfigGemelo(sigma_a=0.3, sigma_lnA=None)
    ).distinguibles


def test_sensibilidad_sin_calibracion_no_identifica_n_y_d():
    cfg = ConfigGemelo(sigma_lnA=None)
    s = sensibilidad(RECETA, cfg)
    assert s.rango < s.J.shape[1]
    assert not s.identificables.any()
    assert np.all(np.isinf(s.delta_min))


def test_con_calibracion_sigma_d_coincide_con_la_formula():
    cfg = ConfigGemelo(sigma_a=0.3, sigma_lnA=None)
    u_rel = 7e-4
    ng = np.array([float(c.material.n_grupo(LAM0)) for c in RECETA])
    sp = ng * u_rel
    s = sensibilidad(RECETA, cfg, sigma_prior_dn=sp)
    d = np.array([c.d for c in RECETA])
    esperado = np.sqrt((cfg.sigma_a / ng) ** 2 + (d * sp / ng) ** 2)
    assert s.identificables.all()
    assert np.allclose(s.sigma[:3], esperado, rtol=1e-6)
    assert np.allclose(s.delta_min[:3], 3 * esperado, rtol=1e-6)


def test_grupos_de_material_y_amplitud_dan_rango_completo():
    cfg = ConfigGemelo(sigma_lnA=0.02)
    s = sensibilidad(RECETA, cfg, grupos=[0, 1, 0])
    assert s.J.shape[1] == 3 + 2
    assert s.rango == 5 and s.identificables.all()


def test_sensibilidad_valida_entradas():
    with pytest.raises(ErrorDeEntrada):
        sensibilidad(RECETA, ConfigGemelo(), grupos=[0, 1])
    with pytest.raises(ErrorDeEntrada):
        sensibilidad(RECETA, ConfigGemelo(), grupos=[0, 1, 0], sigma_prior_dn=[1e-3])


def test_imagen_doble_de_receta_y_cuna_optima():
    hud = ParametrosHUD.ejemplo()
    w_opt = cuna_optima_receta(RECETA, hud)
    assert 0.05e-3 < abs(w_opt) < 2e-3
    assert abs(imagen_doble_receta(RECETA, w_opt, hud)) < 1e-6
    sin_cuna = imagen_doble_receta(RECETA, 0.0, hud)
    assert abs(sin_cuna) > 0.5


def test_amplitudes_son_opt_in_y_no_inventa_contraste_en_interfaz_invisible():
    assert ConfigGemelo().sigma_lnA is None
    receta = [Capa(BK7, 100.0), Capa(BK7, 200.0)]
    for cfg in (ConfigGemelo(), ConfigGemelo(sigma_lnA=0.05)):
        with pytest.raises(ErrorDeEntrada, match="sin contraste"):
            vector_observables(receta, cfg)
    otra = [Capa(BK7, 150.0), Capa(BK7, 150.0)]
    with pytest.raises(ErrorDeEntrada, match="no son observables"):
        distinguibilidad(receta, otra, ConfigGemelo())


def test_distinguibilidad_usa_covarianza_y_no_solo_su_diagonal():
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    cfg = ConfigGemelo(covarianza=cov)
    base = [Capa(BK7, 100.0), Capa(PVB_SUPUESTO, 200.0)]
    ng = [float(c.material.n_grupo(cfg.lam0)) for c in base]
    for diferencia in (np.array([1.0, 1.0]), np.array([1.0, -1.0])):
        otra = [Capa(c.material, c.d + da / nj) for c, da, nj in zip(base, diferencia, ng)]
        resultado = distinguibilidad(base, otra, cfg)
        esperado = np.sqrt(diferencia @ np.linalg.solve(cov, diferencia))
        assert resultado.delta == pytest.approx(esperado)
        assert np.linalg.norm(resultado.contribuciones) == pytest.approx(esperado)


def test_diferencia_de_dos_mediciones_suma_covarianzas():
    otra = [Capa(BK7, 2101.0), *RECETA[1:]]
    uno = distinguibilidad(RECETA, otra, ConfigGemelo())
    dos = distinguibilidad(RECETA, otra, ConfigGemelo(), dos_mediciones=True)
    assert dos.delta == pytest.approx(uno.delta / np.sqrt(2))
    assert dos.error_clasificacion > uno.error_clasificacion
    assert not uno.limites_observabilidad_especificados


def test_gemelo_aplica_limites_caracterizados_de_resolucion_y_amplitud():
    delgada = [RECETA[0], Capa(PVB_SUPUESTO, 0.01), RECETA[2]]
    with pytest.raises(ErrorDeEntrada, match="picos no resueltos"):
        distinguibilidad(RECETA, delgada, ConfigGemelo(separacion_minima_a=10))
    with pytest.raises(ErrorDeEntrada, match="amplitud_minima"):
        distinguibilidad(RECETA, RECETA, ConfigGemelo(amplitud_minima=0.1))
    resultado = distinguibilidad(
        RECETA, RECETA, ConfigGemelo(separacion_minima_a=10, amplitud_minima=1e-4)
    )
    assert resultado.limites_observabilidad_especificados


def test_umbral_tres_es_distancia_snr_no_error_tres_sigma():
    cfg = ConfigGemelo(sigma_a=1.0)
    ng = float(BK7.n_grupo(cfg.lam0))
    base = [Capa(BK7, 100)]
    otra = [Capa(BK7, 100 + 3 / ng)]
    resultado = distinguibilidad(base, otra, cfg)
    assert resultado.delta == pytest.approx(3.0)
    assert resultado.error_clasificacion == pytest.approx(0.0668072, rel=1e-6)


@pytest.mark.parametrize("cov", [np.eye(2), [[0]], [[-1]], [[np.nan]], [[1, 0.2], [0, 1]]])
def test_covarianza_invalida_se_rechaza(cov):
    with pytest.raises(ErrorDeEntrada):
        covarianza_observables(1, ConfigGemelo(covarianza=cov))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_a": 0},
        {"sigma_lnA": np.nan},
        {"lam0": -1},
        {"sigma_c": np.inf},
        {"alpha": np.nan},
        {"sigma_a": None},
    ],
)
def test_configuracion_rechaza_ruido_y_geometria_invalidos(kwargs):
    with pytest.raises(ErrorDeEntrada):
        ConfigGemelo(**kwargs)


def test_fisher_conserva_correlaciones_de_instrumento_y_calibracion():
    receta = RECETA[:2]
    cov = np.array([[0.09, -0.03], [-0.03, 0.16]])
    cp = np.array([[4e-6, 1e-6], [1e-6, 9e-6]])
    resultado = sensibilidad(receta, ConfigGemelo(covarianza=cov), cov_prior_dn=cp)
    esperado = resultado.J.T @ np.linalg.solve(cov, resultado.J)
    esperado[2:, 2:] += np.linalg.inv(cp)
    assert np.allclose(resultado.fisher, esperado)
    ng = np.array([float(c.material.n_grupo(LAM0)) for c in receta])
    d = np.array([c.d for c in receta])
    cov_d = (cov + np.outer(d, d) * cp) / np.outer(ng, ng)
    assert np.allclose(resultado.sigma[:2], np.sqrt(np.diag(cov_d)), rtol=1e-6)


def test_prior_no_finito_y_doble_especificacion_no_crean_precision_falsa():
    with pytest.raises(ErrorDeEntrada):
        sensibilidad(RECETA, ConfigGemelo(), sigma_prior_dn=[np.nan] * 3)
    with pytest.raises(ErrorDeEntrada):
        sensibilidad(RECETA, ConfigGemelo(), sigma_prior_dn=[0.001] * 3, cov_prior_dn=np.eye(3))


def test_hud_usa_longitud_de_onda_especificada():
    hud_azul = ParametrosHUD(np.deg2rad(62), 600e3, 900e3, lam=0.45)
    hud_rojo = ParametrosHUD(np.deg2rad(62), 600e3, 900e3, lam=0.65)
    assert imagen_doble_receta(RECETA, 0.4e-3, hud_azul) != pytest.approx(
        imagen_doble_receta(RECETA, 0.4e-3, hud_rojo), abs=1e-3
    )
    with pytest.raises(ErrorDeEntrada):
        ParametrosHUD(np.deg2rad(62), np.nan, 900e3)


def test_hud_no_inventa_incertidumbre_y_propaga_cuando_se_proporciona():
    hud = ParametrosHUD.ejemplo()
    sin_cov = predecir_imagen_doble(RECETA, 0.35e-3, hud)
    assert sin_cov.sigma_parametros_arcmin is None
    assert not sin_cov.incertidumbre_modelo_incluida
    C = np.zeros((10, 10))
    C[6, 6] = (0.01e-3) ** 2
    resultado = predecir_imagen_doble(RECETA, 0.35e-3, hud, C)
    assert resultado.sigma_parametros_arcmin > 0
    assert resultado.sigma_parametros_arcmin == pytest.approx(abs(resultado.jacobiano[6]) * 0.01e-3)


def test_hud_propagacion_lineal_contra_monte_carlo_local_correlacionado():
    from motor.analisis.imagen_doble import separacion_laminado

    hud = ParametrosHUD.ejemplo()
    p0 = np.array(
        [c.d for c in RECETA]
        + [float(c.material.n(hud.lam)) for c in RECETA]
        + [0.35e-3, hud.theta_i, hud.Rp, hud.Rv]
    )
    sigmas = np.array([1, 0.5, 1, 0.001, 0.002, 0.001, 0.01e-3, 0.001, 300, 450])
    corr = np.eye(10)
    corr[0, 2] = corr[2, 0] = 0.6
    corr[3, 5] = corr[5, 3] = 0.8
    C = corr * np.outer(sigmas, sigmas)
    resultado = predecir_imagen_doble(RECETA, p0[6], hud, C)
    z = np.random.default_rng(12).standard_normal((256, 10)) @ np.linalg.cholesky(corr).T
    muestras = p0 + z * sigmas
    eta = [np.degrees(separacion_laminado(p[:3], p[3:6], *p[6:])) * 60 for p in muestras]
    assert np.std(eta, ddof=1) == pytest.approx(resultado.sigma_parametros_arcmin, rel=0.15)


def test_hud_rechaza_covarianza_indefinida_aunque_mezcle_unidades():
    C = np.diag([1.0] * 10)
    C[9, 9] = 1e12
    C[3, 3] = -1e-12
    with pytest.raises(ErrorDeEntrada):
        predecir_imagen_doble(RECETA, 0.35e-3, ParametrosHUD.ejemplo(), C)
