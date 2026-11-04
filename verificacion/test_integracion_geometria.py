import numpy as np
import pytest

from motor.analisis.pipeline import ConfigPipeline, ejecutar, evaluar_malla_en_xy
from motor.geometria.cad import Malla, Transformacion, _rodrigues, malla_desde_grilla
from motor.medicion.lectura import Instrumento, Medicion
from motor.medicion.picos import ResultadoPicos
from motor.metrologia.calibracion import Calibracion


def _caso_geometrico(
    monkeypatch,
    *,
    cov=None,
    transformacion=None,
    inclinar=False,
    desplazamiento_z=0.0,
    soporte_triangular=False
):
    ng = np.array([1.5, 1.48, 1.5])
    cfg = ConfigPipeline(
        Instrumento(1.55, 10000.0),
        np.array([2100.0, 760.0, 1600.0]),
        np.array([10.0, 25.0, 10.0]),
        ng,
        ["vidrio", "pvb", "vidrio"],
        grilla=(11, 11),
        transformacion=transformacion,
        cov_transformacion=cov,
        sigma_pos=1.0,
    )
    cal = Calibracion(ng, np.full(3, 0.001), np.full(3, 0.2), cfg.grupos)
    xs = np.linspace(-40000.0, 40000.0, 7)
    x, y = np.meshgrid(xs, xs)
    q = np.column_stack(
        [
            x.ravel(),
            y.ravel(),
            8.0 * np.exp(-((x.ravel() / 25000.0) ** 2 + (y.ravel() / 20000.0) ** 2)),
        ]
    )
    if soporte_triangular:
        q = q[q[:, 0] + q[:, 1] <= 1e-6]
    q[:, 2] += desplazamiento_z
    xy = q[:, :2] / 1000.0
    d = np.column_stack(
        [
            2100.0 + 2.0 * np.sin(xy[:, 0] / 8.0),
            760.0 + 0.35 * xy[:, 0] + np.cos(xy.sum(axis=1) / 13.0),
            1600.0 + np.cos(xy[:, 1] / 11.0),
        ]
    )
    n = (
        np.array([0.02, -0.01, -np.sqrt(1.0 - 0.02**2 - 0.01**2)])
        if inclinar
        else np.array([0.0, 0.0, -1.0])
    )
    T = transformacion or Transformacion(np.eye(3), np.zeros(3))
    k = np.linspace(3.0, 5.0, 1024)
    resultados, meds = [], []
    for i, (p, ds) in enumerate(zip(q, d)):
        L = 60000.0 + np.r_[0.0, 2.0 * np.cumsum(ng * ds)]
        resultados.append(
            ResultadoPicos(L, L.copy(), np.ones(4), np.full(4, 100.0), np.full(4, 0.3))
        )
        posicion = (p - n * 20000.0 - T.t) @ T.R
        normal = n @ T.R
        meds.append(Medicion(k, np.full(1024, float(i)), posicion, normal, i))
    monkeypatch.setattr(
        "motor.analisis.pipeline.picos_punto", lambda k, I, cfg: resultados[int(I[0])]
    )
    X, Y = np.meshgrid(np.linspace(-60000.0, 60000.0, 9), np.linspace(-60000.0, 60000.0, 9))
    cad = malla_desde_grilla(X, Y, np.zeros_like(X))
    return ejecutar(meds, cfg, cal, cad), cfg


def test_evaluar_cad_respeta_huecos_de_las_caras():
    X, Y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    completa = malla_desde_grilla(X, Y, X + Y)
    caras = completa.f[~np.any(completa.f == 12, axis=1)]
    usados, inv = np.unique(caras, return_inverse=True)
    m = Malla(completa.v[usados], inv.reshape(-1, 3))
    Z, N = evaluar_malla_en_xy(m, np.array([0.5, 2.0]), np.array([0.5, 2.0]))
    assert Z[0] == pytest.approx(1.0)
    assert np.isnan(Z[1]) and np.isnan(N[1]).all()


def test_interfaces_exportadas_conservan_espesor_reconstruido(monkeypatch):
    res, _ = _caso_geometrico(monkeypatch, inclinar=True)
    for j in range(3):
        distancia = np.linalg.norm(res.interfaces[j + 1] - res.interfaces[j], axis=-1)
        assert np.allclose(
            distancia[res.mascara_soporte],
            res.d_media[j][res.mascara_soporte],
            rtol=1e-10,
            atol=1e-7,
        ), np.max(np.abs(distancia - res.d_media[j]))


def test_covarianza_marco_cad_incremento_izquierdo_no_se_promedia(monkeypatch):
    T = Transformacion(_rodrigues(np.array([0.1, -0.05, 0.08])), np.array([100.0, -200.0, 300.0]))
    direccion = np.array([1e-4, -2e-4, 0.0, 1.0, 2.0, 3.0])
    C = np.outer(direccion, direccion)
    C[5, 5] += 16.0
    sin, _ = _caso_geometrico(monkeypatch, transformacion=T, cov=np.zeros((6, 6)))
    con, _ = _caso_geometrico(monkeypatch, transformacion=T, cov=C)
    diferencia = con.delta_std**2 - sin.delta_std**2
    esperado = (con.Yg * direccion[0] - con.Xg * direccion[1] + direccion[5]) ** 2 + 16.0
    assert np.allclose(diferencia[:, con.mascara_soporte], esperado[con.mascara_soporte], atol=1e-7)
    assert np.allclose(con.delta_media, sin.delta_media)


def test_marco_fijo_conserva_error_global_frente_al_cad(monkeypatch):
    base, _ = _caso_geometrico(monkeypatch, cov=np.zeros((6, 6)))
    desplazado, _ = _caso_geometrico(monkeypatch, cov=np.zeros((6, 6)), desplazamiento_z=40.0)
    assert np.allclose(desplazado.d_media, base.d_media, atol=1e-6)
    assert np.allclose(desplazado.delta_media - base.delta_media, 40.0, atol=0.05)


def test_soporte_triangular_no_se_convierte_en_pieza_rectangular(monkeypatch):
    from motor.geometria.exportar_cad import es_cerrada, malla_superficie, solido_capa

    res, _ = _caso_geometrico(monkeypatch, soporte_triangular=True)
    mask = res.mascara_soporte
    assert not mask.all()
    assert np.all(res.Xg[mask] + res.Yg[mask] <= 1e-5)
    assert not res.falla_d[:, ~mask].any() and not res.falla_w[~mask].any()
    superficie = malla_superficie(res.interfaces[0], mask)
    assert superficie.areas().sum() < 0.6 * (80000.0**2)
    assert es_cerrada(solido_capa(res.interfaces[0], res.interfaces[1], mask))
