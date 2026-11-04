import numpy as np
import pytest

from motor.geometria.cad import (
    Malla,
    _rodrigues,
    escribir_stl,
    interfaces_nominales,
    leer_stl,
    malla_desde_grilla,
    registrar_icp,
)
from motor.medicion.lectura import ErrorDeEntrada

UM = 1e3


def _cad(nx=41, ny=41, plano=False):
    xs = np.linspace(-60 * UM, 60 * UM, nx)
    ys = np.linspace(-60 * UM, 60 * UM, ny)
    X, Y = np.meshgrid(xs, ys)
    if plano:
        Z = np.zeros_like(X)
    else:
        Z = (
            X**2 / (2 * 1500 * UM)
            + Y**2 / (2 * 2500 * UM)
            + 400.0 * np.exp(-((X - 20 * UM) ** 2 + (Y + 15 * UM) ** 2) / (25 * UM) ** 2)
        )
    return malla_desde_grilla(X, Y, Z), (X, Y, Z)


def test_proyeccion_plano_exacta_a_la_micra():
    m, _ = _cad(plano=True)
    P = np.array([[3.3 * UM, -7.1 * UM, 123.456], [-59 * UM, 59 * UM, -50.0], [0.0, 0.0, 1e-3]])
    Q, d, _ = m.proyectar(P)
    assert np.allclose(np.abs(d), np.abs(P[:, 2]), atol=1e-6)
    assert np.allclose(Q[:, 2], 0.0, atol=1e-9)
    assert d[0] > 0 and d[1] < 0


def test_proyeccion_fuera_del_borde_va_a_la_arista():
    m, _ = _cad(plano=True)
    Q, d, _ = m.proyectar(np.array([[70 * UM, 0.0, 0.0]]))
    assert abs(Q[0, 0] - 60 * UM) < 1e-6 and abs(d[0]) == pytest.approx(10 * UM, rel=1e-9)


def test_proyeccion_esfera_error_de_cuerda():
    m, (X, Y, Z) = _cad(nx=121, ny=121)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, X.size, 50)
    p0 = np.column_stack([X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx]])
    _, d, _ = m.proyectar(p0)
    assert np.max(np.abs(d)) < 1e-6


def test_stl_ida_y_vuelta_binario_y_ascii(tmp_path):
    m, _ = _cad(nx=9, ny=9)
    for modo in (False, True):
        ruta = str(tmp_path / f"m_{modo}.stl")
        escribir_stl(ruta, m, ascii_=modo)
        r = leer_stl(ruta)
        assert len(r.f) == len(m.f) and len(r.v) == len(m.v)
        assert abs(r.areas().sum() - m.areas().sum()) / m.areas().sum() < 1e-5


def test_stl_corrupto_rechazado(tmp_path):
    ruta = tmp_path / "x.stl"
    ruta.write_bytes(b"\0" * 80 + (100).to_bytes(4, "little") + b"\0" * 10)
    with pytest.raises(ErrorDeEntrada):
        leer_stl(str(ruta))


def test_volumen_de_caja_cerrada():
    v = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0], [0, 0, 4], [2, 0, 4], [2, 3, 4], [0, 3, 4]],
        float,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    assert Malla(v, f).volumen() == pytest.approx(24.0)


def test_interfaces_nominales_distancia():
    m, _ = _cad(nx=61, ny=61)
    cap = interfaces_nominales(m, [2100.0, 760.0, 1600.0])
    assert len(cap) == 4
    Dcum = np.cumsum([0, 2100.0, 760.0, 1600.0])
    for k in (1, 2, 3):
        P = cap[k].v[::37]
        _, d, _ = m.proyectar(P)
        assert np.allclose(np.abs(d), Dcum[k], rtol=2e-3)
        assert np.all(d < 0)


def test_icp_recupera_transformacion_conocida():
    m, _ = _cad(nx=81, ny=81)
    rng = np.random.default_rng(1)
    P_cad = m.muestrear(600, rng)
    w_true = np.array([2e-3, -1.5e-3, 1e-3])
    t_true = np.array([30.0, -20.0, 80.0])
    Rt = _rodrigues(w_true)
    P = (P_cad - t_true) @ Rt + rng.normal(0, 1.0, P_cad.shape)
    r = registrar_icp(P, m)
    assert r.convergio and r.rms < 3.0
    e = r.T.aplicar(P) - P_cad
    _, _, ti = m.proyectar(P_cad)
    n = m.normales_cara()[ti]
    en = np.einsum("ij,ij->i", e, n)
    et = np.linalg.norm(e - en[:, None] * n, axis=1)
    assert np.sqrt(np.mean(en**2)) < 2.0
    assert np.sqrt(np.mean(et**2)) < 100.0


def test_icp_superficie_plana_se_marca_mal_condicionada():
    m, _ = _cad(plano=True)
    rng = np.random.default_rng(2)
    P = m.muestrear(300, rng) + np.array([0.0, 0.0, 50.0])
    r = registrar_icp(P, m)
    assert any(b.startswith("mal_condicionado") for b in r.banderas), r.cond
    assert abs(r.T.t[2] + 50.0) < 1e-3


def test_icp_resiste_atipicos():
    m, _ = _cad(nx=81, ny=81)
    rng = np.random.default_rng(3)
    P = m.muestrear(500, rng)
    P = P + np.array([0.0, 0.0, 40.0])
    P[:25] += rng.normal(0, 3000.0, (25, 3))
    r = registrar_icp(P, m, recorte=0.9)
    assert r.rms < 2.0
    assert abs(r.T.t[2] + 40.0) < 20.0


def test_icp_rechaza_pocos_puntos():
    m, _ = _cad(nx=9, ny=9)
    with pytest.raises(ErrorDeEntrada):
        registrar_icp(np.zeros((3, 3)), m)


def test_icp_con_prior_de_montaje_evita_la_deriva_tangencial():
    m, _ = _cad(nx=81, ny=81)
    rng = np.random.default_rng(5)
    P_cad = m.muestrear(500, rng)
    w_true = np.array([0.5e-3, -0.4e-3, 0.6e-3])
    t_true = np.array([40.0, -25.0, 120.0])
    Rt = _rodrigues(w_true)
    P = (P_cad - t_true) @ Rt + rng.normal(0, 1.0, P_cad.shape)
    r = registrar_icp(P, m, sigma_prior_traslacion=150.0, sigma_prior_rotacion=1e-3)
    assert r.convergio, r.banderas
    e = r.T.aplicar(P) - P_cad
    _, _, ti = m.proyectar(P_cad)
    n = m.normales_cara()[ti]
    en = np.einsum("ij,ij->i", e, n)
    et = np.linalg.norm(e - en[:, None] * n, axis=1)
    assert np.sqrt(np.mean(en**2)) < 2.0
    assert np.sqrt(np.mean(et**2)) < 100.0


def test_proyeccion_no_confunde_centro_lejano_con_triangulo_lejano():
    v = [[-1000.0, -1000.0, 0.0], [4000.0, -1000.0, 0.0], [-1000.0, 4000.0, 0.0]]
    f = [[0, 1, 2]]
    for j in range(20):
        x = 0.2 + j * 0.01
        k = len(v)
        v.extend([[x, 0.0, 10.0], [x + 0.01, 0.0, 10.0], [x, 0.01, 10.0]])
        f.append([k, k + 1, k + 2])
    m = Malla(v, f)
    q, d, _ = m.proyectar([[0.0, 0.0, 1.0]], k=1)
    assert np.allclose(q, [[0.0, 0.0, 0.0]])
    assert d[0] == pytest.approx(1.0)


def test_malla_rechaza_degenerados_no_finitos_e_indices_fraccionarios():
    v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    for vertices, caras in [
        (v, [[0, 0, 1]]),
        ([[np.nan, 0, 0], *v[1:]], [[0, 1, 2]]),
        (v, [[0, 1, 1.5]]),
        ([], []),
    ]:
        with pytest.raises(ErrorDeEntrada):
            Malla(vertices, caras)


def test_cad_nominal_no_se_modifica_por_alias_de_entrada():
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m = Malla(v, [[0, 1, 2]])
    v[0] = 900
    assert np.allclose(m.v[0], 0)
    with pytest.raises(ValueError):
        m.v[0] = 900


def test_nominal_espacial_fijo_por_vertice():
    m, _ = _cad(nx=4, ny=4, plano=True)
    nominal = np.vstack([np.full(len(m.v), 2100.0), 760.0 + 0.35 * m.v[:, 0] / UM])
    capas = interfaces_nominales(m, nominal)
    assert np.allclose(capas[2].v[:, 2], -nominal.sum(axis=0))
    assert np.allclose(m.v[:, 2], 0)


def test_stl_binario_con_encabezado_solid_y_archivo_truncado(tmp_path):
    m, _ = _cad(nx=3, ny=3, plano=True)
    p = tmp_path / "solid_binario.stl"
    escribir_stl(str(p), m, nombre="solid facet encabezado binario valido")
    assert len(leer_stl(str(p)).f) == len(m.f)
    for datos in (b"", b"abc", b"solid vacio\nendsolid vacio\n", b"solid x\nvertex 1 2\n"):
        p.write_bytes(datos)
        with pytest.raises(ErrorDeEntrada):
            leer_stl(str(p))


def test_icp_no_inventa_covarianza_en_grados_libres():
    m, _ = _cad(nx=5, ny=5, plano=True)
    p = m.muestrear(40) + [0.0, 0.0, 50.0]
    r = registrar_icp(p, m)
    assert r.covarianza is None
    rp = registrar_icp(p, m, sigma_prior_traslacion=150.0, sigma_prior_rotacion=0.001)
    assert rp.covarianza is not None
    assert np.all(np.linalg.eigvalsh(rp.covarianza) > 0)


@pytest.mark.parametrize(
    "opciones",
    [{"recorte": 0}, {"max_iter": 0}, {"sigma_ruido": np.nan}, {"sigma_prior_traslacion": -1.0}],
)
def test_registro_rechaza_opciones_invalidas(opciones):
    m, _ = _cad(nx=3, ny=3, plano=True)
    with pytest.raises(ErrorDeEntrada):
        registrar_icp(m.muestrear(20), m, **opciones)


def test_stl_binario_rechaza_triangulos_que_colapsan_por_precision(tmp_path):
    m = Malla([[1e12, 0.0, 0.0], [1e12 + 1.0, 0.0, 0.0], [1e12, 1.0, 0.0]], [[0, 1, 2]])
    ruta = tmp_path / "colapsado.stl"
    with pytest.raises(ErrorDeEntrada):
        escribir_stl(str(ruta), m)
    assert not ruta.exists()
