from dataclasses import replace

import numpy as np
import pytest

from motor.analisis.pipeline import configuracion_demo, picos_punto
from motor.geometria.soporte import (
    aplicar_mascara,
    cargar_especificacion,
    mapas_tolerancia,
    probabilidad_con_mapa,
)
from motor.geometria.superficies import ProcesoGaussiano2D
from motor.medicion.lectura import ErrorDeEntrada, Medicion, preparar_espectros_archivo
from motor.metrologia.diagnosticos import (
    CalculoCancelado,
    covarianza_normal,
    diagnostico_incidencia,
    notificar,
    presupuesto,
)


def _rejilla():
    X, Y = np.meshgrid(np.arange(7.0), np.arange(7.0))
    return X, Y, np.ones(X.shape, bool)


def test_hueco_explicito_excluye_contorno_y_centro_sin_rellenarlo():
    X, Y, soporte = _rejilla()
    hueco = [[2, 2], [4, 2], [4, 4], [2, 4]]
    m = aplicar_mascara(X, Y, soporte, {"excluir": [hueco]})
    assert not m[2:5, 2:5].any()
    assert m.sum() == 40
    assert m[0].all()


def test_mascara_npz_resuelve_ruta_relativa_y_rechaza_otra_rejilla(tmp_path):
    X, Y, soporte = _rejilla()
    soporte[3, 3] = False
    np.savez(tmp_path / "hueco.npz", mascara=soporte, X=X, Y=Y)
    esp = cargar_especificacion({"archivo": "hueco.npz"}, tmp_path)
    assert np.array_equal(aplicar_mascara(X, Y, np.ones_like(soporte), esp), soporte)
    with pytest.raises(ErrorDeEntrada, match="no coinciden"):
        aplicar_mascara(X + 1, Y, soporte, esp)


@pytest.mark.parametrize("m", [np.ones((7, 7), int), np.ones((8, 7), bool)])
def test_mascara_no_boolean_o_otra_forma_no_se_adivina(m):
    X, Y, soporte = _rejilla()
    with pytest.raises(ErrorDeEntrada, match="booleana"):
        aplicar_mascara(X, Y, soporte, {"mascara": m})


def test_tolerancia_por_zona_cambia_riesgo_solo_donde_corresponde():
    X, Y, _ = _rejilla()
    zonas = [
        {
            "nombre": "HUD",
            "poligono_xy_um": [[2, 2], [4, 2], [4, 4], [2, 4]],
            "tol_d": [1, 2, 3],
            "tol_w": 0.02,
            "tol_delta": [4, 5, 6, 7],
        }
    ]
    td, tw, tdelta = mapas_tolerancia(X, Y, [10, 20, 30], 0.1, [15] * 4, zonas)
    assert td[:, 3, 3] == pytest.approx([1, 2, 3])
    assert td[:, 0, 0] == pytest.approx([10, 20, 30])
    assert tw[3, 3] == 0.02 and tdelta[2, 3, 3] == 6
    p = probabilidad_con_mapa(np.full_like(X, 2.0), np.ones_like(X) * 0.1, 0.0, td[0])
    assert p[3, 3] > 0.99 and p[0, 0] < 0.001


def test_covarianza_normal_proyecta_normalizacion_y_rechaza_negativa():
    n = np.array([0.0, 0.0, -1.0])
    C = covarianza_normal(n, np.diag([0.001, 0.002, 0.003]))
    assert np.array_equal(C, np.diag([0.001, 0.002, 0.0]))
    with pytest.raises(ErrorDeEntrada, match="semidefinida"):
        covarianza_normal(n, -np.eye(3))
    k = np.linspace(3.0, 5.0, 1024)
    with pytest.raises(ErrorDeEntrada):
        Medicion(k, k, np.zeros(3), n, cov_normal=-np.eye(3)).validar()


def test_incidencia_tres_grados_cuantifica_sin_corregir_ni_fingir_sigma():
    angulo = np.deg2rad(3)
    n = np.array([[np.sin(angulo), 0.0, -np.cos(angulo)]])
    diag = diagnostico_incidencia(n, [[0.0, 0.0, 1.0]], [[2100.0]], [1.5])
    assert diag["angulo_max_deg"] == pytest.approx(3)
    assert diag["exceso_camino_geometrico_max_um_por_capa"][0] == pytest.approx(
        1.2793926404, rel=1e-8
    )
    assert diag["correccion_aplicada"] is False
    assert diag["sigma_angulo_sonda_deg"] is None
    assert diag["incertidumbre_correccion_oblicua_um"] is None


def test_incidencia_sin_indice_fase_no_sustituye_por_indice_grupo():
    diag = diagnostico_incidencia([[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]], [[2100.0]])
    assert diag["indices_fase_usados"] is None
    assert diag["exceso_camino_geometrico_max_um_por_capa"] is None


def test_presupuesto_no_declara_total_ni_cero_para_terminos_desconocidos():
    from motor.metrologia.calibracion import Calibracion

    cfg, _ = configuracion_demo()
    cal = Calibracion(np.array([1.5, 1.48, 1.5]), np.ones(3) * 0.001, np.ones(3) * 0.2, cfg.grupos)
    filas = {f["componente"]: f for f in presupuesto(cfg, cal, False, {})}
    assert set(filas) == {
        "referencia",
        "grupo_optico",
        "eje_espectral",
        "offset",
        "deriva",
        "montaje",
        "normal_sonda",
        "repetibilidad",
        "interpolacion",
        "modelo",
    }
    for componente in ("eje_espectral", "offset", "deriva", "modelo"):
        assert filas[componente]["sigma"] is None
        assert filas[componente]["propagada"] is False
    cfg.presupuesto_declarado = {
        "offset": {"sigma": 0.2, "unidad": "µm", "fuente": "certificado A"}
    }
    filas = {f["componente"]: f for f in presupuesto(cfg, cal, False, {})}
    assert filas["offset"]["declaracion_adicional"]["sigma"] == 0.2
    assert filas["offset"]["declaracion_adicional_integrada"] is False


def test_cancelacion_cooperativa_no_produce_resultado_nuevo():
    with pytest.raises(CalculoCancelado, match="cancelado"):
        notificar(cancelar=lambda: True)
    estados = []
    notificar(
        progreso=lambda frac, texto: estados.append((frac, texto)), fraccion=0.3, texto="Espesor"
    )
    assert estados == [(0.3, "Espesor")]


def test_gp_por_lotes_equivale_en_media_sigma_y_gradientes():
    X, Y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    xy = np.column_stack([X.ravel(), Y.ravel()])
    gp = ProcesoGaussiano2D().ajustar(xy, 2 + xy[:, 0] + 0.1 * np.sin(xy[:, 1]), np.ones(16) * 0.2)
    a = gp.predecir(xy, gradiente=True, lote=100)
    b = gp.predecir(xy, gradiente=True, lote=3)
    for aa, bb in zip(a, b):
        assert np.allclose(aa, bb, atol=1e-10, rtol=1e-8)


def test_lambda_y_eje_k_no_uniforme_exigen_opcion_y_mismo_tratamiento():
    lam = np.linspace(1.5, 1.6, 1024)
    z = {"lambda_um": lam, "I": np.array([np.sin(lam), np.cos(lam)])}
    with pytest.raises(ErrorDeEntrada, match="activa remuestrear"):
        preparar_espectros_archivo(z)
    out = preparar_espectros_archivo(z, remuestrear=True)
    assert np.ptp(np.diff(out["k"])) / np.diff(out["k"]).mean() < 1e-9
    assert out["I"].shape == (2, 1024)
    zk = {"k": np.linspace(3.0, 5.0, 1024) ** 1.01, "I": z["I"]}
    with pytest.raises(ErrorDeEntrada, match="no uniforme"):
        preparar_espectros_archivo(zk)
    assert np.ptp(np.diff(preparar_espectros_archivo(zk, True)["k"])) < 1e-12


def test_saturacion_declarada_detiene_y_dice_que_corregir():
    cfg, _ = configuracion_demo()
    cfg.instrumento = replace(cfg.instrumento, limites_saturacion=(0.0, 4095.0))
    with pytest.raises(ErrorDeEntrada, match="reduce exposición"):
        picos_punto(np.linspace(3.0, 5.0, 1024), np.full(1024, 4095.0), cfg)


def test_pose_real_cero_sin_declaracion_no_se_adivina_exacta():
    cfg, _ = configuracion_demo()
    with pytest.raises(ErrorDeEntrada, match="Covarianza cero"):
        replace(cfg, origen_datos="real_declarado")
    assert (
        replace(cfg, origen_datos="real_declarado", cov_transformacion=None).cov_transformacion
        is None
    )


def test_covarianza_normal_comun_se_agrega_despues_del_gp(monkeypatch):
    from test_integracion_geometria import _caso_geometrico

    base, _ = _caso_geometrico(monkeypatch, inclinar=True, cov=np.zeros((6, 6)))
    n = np.array([0.02, -0.01, -np.sqrt(1 - 0.02**2 - 0.01**2)])
    C = 1e-6 * (np.eye(3) - np.outer(n, n))
    monkeypatch.setattr("motor.analisis.pipeline.covarianza_normal", lambda normal, cov: C)
    con, _ = _caso_geometrico(monkeypatch, inclinar=True, cov=np.zeros((6, 6)))
    for k in range(4):
        profundidad = con.d_media[:k].sum(axis=0)
        esperado = (20000 + profundidad) ** 2 * C[2, 2]
        assert np.allclose(
            (con.delta_std[k] ** 2 - base.delta_std[k] ** 2)[con.mascara_soporte],
            esperado[con.mascara_soporte],
            atol=1e-7,
        )
    assert np.allclose(base.d_media, con.d_media)


def test_pipeline_hueco_llega_a_stl_y_ninguna_decision_lo_usa(monkeypatch, tmp_path):
    import test_integracion_geometria as casos
    from test_integracion_geometria import _caso_geometrico

    from motor.geometria.cad import leer_stl
    from motor.geometria.exportar_cad import exportar_stl

    original = casos.ConfigPipeline
    mascara = {"excluir": [[[-10000, -10000], [10000, -10000], [10000, 10000], [-10000, 10000]]]}
    monkeypatch.setattr(
        casos, "ConfigPipeline", lambda *a, **kw: original(*a, mascara_adquisicion=mascara, **kw)
    )
    res, cfg = _caso_geometrico(monkeypatch)
    interior = (abs(res.Xg) <= 10000) & (abs(res.Yg) <= 10000)
    assert not res.mascara_soporte[interior].any()
    assert not res.falla_d[:, interior].any() and not res.falla_w[interior].any()
    reporte = exportar_stl(tmp_path, res.interfaces, cfg.nombres_capas, mascara=res.mascara_soporte)
    assert all(reporte.cerradas.values())
    for ruta in reporte.archivos.values():
        mesh = leer_stl(ruta)
        assert not ((abs(mesh.v[:, 0]) < 10000) & (abs(mesh.v[:, 1]) < 10000)).any()
    from motor.analisis.pipeline import guardar
    from motor.metrologia.calibracion import Calibracion

    cal = Calibracion(np.array([1.5, 1.48, 1.5]), np.full(3, 0.001), np.full(3, 0.2), cfg.grupos)
    informe = guardar(res, cfg, cal, str(tmp_path))
    assert informe["soporte"]["mascara_explicita"] is True
    with np.load(tmp_path / "mapas.npz", allow_pickle=False) as z:
        for clave in ("d_media", "d_std", "delta_media", "P_fuera_d", "tol_d"):
            assert np.isnan(z[clave][..., ~res.mascara_soporte]).all()
        assert np.isnan(z["interfaces"][:, ~res.mascara_soporte]).all()
        assert np.isnan(z["gradiente_cuna"][~res.mascara_soporte]).all()
        assert len(z["ids_validos"]) == int(res.mascara_puntos.sum())


def test_presupuesto_referencia_conserva_sigma_y_cov_sin_sumar_dos_veces():
    from motor.medicion.picos import ResultadoPicos
    from motor.metrologia.calibracion import calibrar

    cfg, _ = configuracion_demo()
    d = np.array([2100.0, 760.0, 1600.0])
    ng = np.array([1.5, 1.48, 1.5])
    L = 60000 + np.r_[0.0, 2 * np.cumsum(d * ng)]
    r = ResultadoPicos(L, L.copy(), np.ones(4), np.ones(4) * 100, np.ones(4) * 0.3)
    cal = calibrar([r] * 5, d, [1.0, 2.0, 3.0], cfg.grupos)
    filas = presupuesto(cfg, cal, False, {})
    referencia = next(f for f in filas if f["componente"] == "referencia")
    assert referencia["sigma"] == [1.0, 2.0, 3.0]
    assert referencia["covarianza"] == np.diag([1.0, 4.0, 9.0]).tolist()
    assert cal.metadatos_referencia["incluida_en_cov_ng"] is True
    assert "no sumar" in referencia["estado"]


def test_hueco_menor_que_rejilla_nunca_es_puenteado_por_stl():
    X, Y, soporte = _rejilla()
    hueco = [[2.3, 2.3], [2.7, 2.3], [2.7, 2.7], [2.3, 2.7]]
    m = aplicar_mascara(X, Y, soporte, {"excluir": [hueco]})
    assert not m[2:4, 2:4].any()
    assert m[0].all()


def test_config_real_sin_sigma_pos_no_usa_default_sintetico(tmp_path):
    import json

    from motor.analisis.pipeline import ejecutar_desde_config

    c = {
        "mediciones": "m.npz",
        "calibracion": {"archivo": "c.npz"},
        "cad": "c.stl",
        "instrumento": {"lam0": 1.55, "offset": 10000},
        "capas": {
            "d_nom": [2100, 760, 1600],
            "tol_d": [10, 25, 10],
            "ng_prior": [1.5, 1.48, 1.5],
            "grupos": ["g", "p", "g"],
        },
        "cuna": {"w_nom": 0.35, "tol": 0.1},
        "rango_frontal": [19000, 21000],
        "registro": {"modo": "fijo", "R": np.eye(3).tolist(), "t": [0, 0, 0]},
        "origen_datos": "real_declarado",
    }
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(c))
    with pytest.raises(ErrorDeEntrada, match="falta sigma_pos"):
        ejecutar_desde_config(ruta, verbose=False)


def test_gp_predice_100x100_con_bloques_acotados(monkeypatch):
    X, Y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    xy = np.column_stack([X.ravel(), Y.ravel()])
    gp = ProcesoGaussiano2D().ajustar(xy, 1 + xy[:, 0], np.full(9, 0.1))
    original = gp._k
    tamanos = []

    def limitado(A, B, ell, s):
        tamanos.append(len(A))
        return original(A, B, ell, s)

    monkeypatch.setattr(gp, "_k", limitado)
    XX, YY = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    mu, sigma = gp.predecir(np.column_stack([XX.ravel(), YY.ravel()]))
    assert max(tamanos) <= 1024 and len(tamanos) > 1
    assert np.allclose(mu, 1 + XX.ravel(), atol=1e-6)
    assert np.isfinite(sigma).all()


@pytest.mark.parametrize("normal", [[[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]])
def test_sonda_no_entrante_no_se_confunde_con_incidencia_normal(normal):
    with pytest.raises(ErrorDeEntrada, match="hacia el vidrio"):
        diagnostico_incidencia(normal, [[0.0, 0.0, 1.0]], [[2100.0]], [1.5])
