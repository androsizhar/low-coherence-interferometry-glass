import itertools
import json
from pathlib import Path

import laminado_sintetico as L
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motor.geometria.cad import Malla, Transformacion, escribir_stl
from motor.geometria.importar_cad import (
    OpcionesImportacion,
    PiezaCAD,
    Solido,
    Superficie,
    analizar_pieza,
    importar,
    leer_stl_capas,
)
from motor.geometria.soporte import aplicar_mascara, dentro_region, mapas_tolerancia
from motor.medicion.lectura import ErrorDeEntrada


def _pieza(cuna=0.5, orden=(2, 1, 0), extra=()):
    capas = L.mallas_capas(cuna)
    nombres = ["vidrio_ext", "intercapa", "vidrio_int"]
    solidos = [Solido(nombres[i], capas[i], abs(capas[i].volumen())) for i in orden]
    zona = Superficie("HUD", L.zona_rectangular(20e3, 80e3, -50e3, 10e3, 3, cuna_mrad=cuna))
    return PiezaCAD(solidos + list(extra), [zona])


def _verdad_local(modelo, x, y, z):
    R0, t0 = L.pose()
    M = modelo.marco_R @ R0
    tt = modelo.marco_R @ t0 + modelo.marco_t
    return (np.stack([x, y, z], -1) - tt) @ M


@pytest.fixture(scope="module")
def modelo():
    return analizar_pieza(_pieza())


def test_capas_por_geometria_sin_nombres_ni_orden(modelo):
    assert modelo.nombres_capas == ["vidrio_ext", "intercapa", "vidrio_int"]
    m = modelo.mascara
    P = _verdad_local(modelo, modelo.X[m], modelo.Y[m], modelo.z0[m])
    verdad = L.espesores(P[:, 0], P[:, 1])
    assert np.abs(modelo.d_nom[:, m] - verdad).max() < 0.01
    assert np.abs(P[:, 2] - L.z_frontal(P[:, 0], P[:, 1])).max() < 0.01


def test_cuna_zonas_y_diagnostico(modelo):
    assert modelo.sugerencias["idx_cuna"] == 1
    c = modelo.diagnostico["capas"]["intercapa"]
    assert c["cuna_p50_mrad"] == pytest.approx(0.5, abs=1e-3)
    assert modelo.diagnostico["radio_curvatura_min_mm"] == pytest.approx(1500, rel=0.02)
    assert modelo.diagnostico["supuesto_capas_localmente_planas"]["valido"]
    (z,) = modelo.zonas
    assert z["nombre"] == "HUD" and z["interfaz"] == 3 and len(z["lazos_xy_um"]) == 1
    assert modelo.diagnostico["error_cuerda_stl_frontal_um"] <= 1.0 + 1e-9


def test_lado_concavo_invierte_el_apilado():
    m = analizar_pieza(_pieza(), OpcionesImportacion(lado_sonda="concavo"))
    assert m.nombres_capas == ["vidrio_int", "intercapa", "vidrio_ext"]
    assert np.nanmean(m.d_nom[0]) == pytest.approx(1600, abs=0.01)


def test_accesorio_pegado_se_ignora():
    S = L.superficies()
    arriba = S[0][18:24, 25:35]
    n = L.normal_frontal(arriba[..., 0], arriba[..., 1])
    cinta = L._solido(arriba + 800 * n, arriba)
    R, t = L.pose()
    cinta = Malla(cinta.v @ R.T + t, cinta.f)
    m = analizar_pieza(_pieza(extra=[Solido("cinta", cinta, abs(cinta.volumen()))]))
    assert len(m.nombres_capas) == 3
    assert any("cinta" in a for a in m.avisos)


def test_capas_sin_contacto_fallan():
    capas = L.mallas_capas()
    lejos = Malla(capas[2].v + [0, 0, 5e5], capas[2].f)
    pieza = PiezaCAD([Solido("a", capas[0]), Solido("b", lejos)])
    with pytest.raises(ErrorDeEntrada, match="contacto"):
        analizar_pieza(pieza)


def test_stl_abierto_rechazado(tmp_path):
    zona = L.zona_rectangular(0, 1e4, 0, 1e4, 0)
    r = tmp_path / "abierta.stl"
    escribir_stl(str(r), zona)
    with pytest.raises(ErrorDeEntrada, match="cerrado"):
        leer_stl_capas([str(r), str(r)], "um")


def test_region_par_impar_con_hueco():
    X, Y = np.meshgrid(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21))
    xy = np.stack([X, Y], -1)
    anillo = [[[-8, -8], [8, -8], [8, 8], [-8, 8]], [[-4, -4], [4, -4], [4, 4], [-4, 4]]]
    r = dentro_region(xy, anillo)
    assert r[10, 10] == False and r[10, 3] == True and r[0, 0] == False  # noqa: E712
    m = aplicar_mascara(X, Y, np.ones(X.shape, bool), {"excluir_regiones": [anillo]})
    assert m[10, 10] and not m[10, 3]
    _td, tw, _ = mapas_tolerancia(
        X, Y, [10.0], 0.1, zonas=[{"poligonos_xy_um": anillo, "tol_w": 0.05}]
    )
    assert tw[10, 3] == 0.05 and tw[10, 10] == 0.1


def test_composicion_de_marcos_y_covarianza():
    rng = np.random.default_rng(0)
    M = Transformacion(
        Rotation.from_rotvec(np.array([0.3, -0.2, 0.5])).as_matrix(), np.array([1e5, -2e5, 3e4])
    )
    T = Transformacion(
        Rotation.from_rotvec(np.array([0.01, 0.02, -0.01])).as_matrix(), np.array([10.0, 5.0, -3.0])
    )
    P = rng.normal(size=(5, 3)) * 1e4
    assert np.allclose(M.componer(T).aplicar(P), M.aplicar(T.aplicar(P)))
    w, dt = np.array([1e-4, -2e-4, 5e-5]), np.array([3.0, -1.0, 2.0])
    S = T.aplicar(P)
    dS = np.cross(w, S) + dt
    A = M.transportar_covarianza(np.eye(6))
    assert np.allclose(A, A.T)
    Rm = M.R
    w2 = Rm @ w
    dt2 = Rm @ dt + np.cross(M.t, w2)
    assert np.allclose(np.cross(w2, M.aplicar(S)) + dt2, dS @ Rm.T)


def _cadena(tmp_path, cuna, n_lado=9, extra_cfg=None):
    capas = L.escribir(str(tmp_path), cuna)
    _, ruta = importar(capas[0], str(tmp_path / "modelo"), unidades="um", zonas_stl=capas[1])
    from motor.analisis.pipeline import ejecutar_desde_config
    from motor.analisis.sintetico_cad import comparar_con_verdad, generar_desde_cad

    cfg = generar_desde_cad(ruta, str(tmp_path / "pieza"), n_lado=n_lado, semilla=3)
    if extra_cfg:
        ruta = Path(cfg)
        ruta.write_text(json.dumps(json.loads(ruta.read_text()) | extra_cfg))
    _, informe = ejecutar_desde_config(cfg, verbose=False)
    return informe, comparar_con_verdad(str(tmp_path / "pieza"))


def test_cadena_completa_con_cuna_grande(tmp_path):
    # 2 mrad sobre 300 mm: el PVB varía 600 µm, fuera de una ventana fija de picos
    informe, error = _cadena(tmp_path, 2.0)
    assert informe["puntos_con_defecto"] == {}
    assert all(e["rms_um"] < 0.5 for e in error.values())
    assert informe["cuna_pvb_mrad"]["media"] == pytest.approx(2.0, abs=0.02)
    assert informe["cuna_pvb_mrad"]["nominal_origen"] == "mapa_desde_cad"
    assert informe["nominal"]["tipo"] == "mapa_por_nodo_desde_cad"
    assert informe["cad_importado"]["zonas"] == ["HUD"]


def test_zonas_cad_en_mascara_y_tolerancias(tmp_path):
    extra = {
        "mascara_adquisicion": {"excluir_zonas_cad": ["HUD"]},
        "zonas_tolerancia": [{"zona_cad": "HUD", "tol_w": 0.03}],
    }
    informe, _ = _cadena(tmp_path, 0.5, extra_cfg=extra)
    with np.load(tmp_path / "pieza" / "salida" / "mapas.npz") as m:
        soporte, tw = m["mascara_soporte"], m["tol_w"]
    assert informe["soporte"]["mascara_explicita"]
    assert soporte.sum() > 0 and (~soporte).sum() > 0
    assert np.all(np.isnan(tw[~soporte]))


def test_zona_inexistente_falla(tmp_path):
    with pytest.raises(ErrorDeEntrada, match="no está en el CAD"):
        _cadena(tmp_path, 0.5, extra_cfg={"zonas_tolerancia": [{"zona_cad": "nada", "tol_w": 1}]})


def test_step_con_gmsh(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    try:
        gmsh.initialize(interruptible=False)
    except Exception:
        pytest.skip("gmsh no inicializa en este entorno")
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        R = 1500.0
        radios = [R, R - 2.1, R - 2.1 - 0.76, R - 2.1 - 0.76 - 1.6]
        caja = gmsh.model.occ.addBox(-150, -100, R - 200, 300, 200, 400)
        for a, b in itertools.pairwise(radios):
            ext = gmsh.model.occ.addCylinder(0, -100, 0, 0, 200, 0, a)
            inn = gmsh.model.occ.addCylinder(0, -100, 0, 0, 200, 0, b)
            tubo, _ = gmsh.model.occ.cut([(3, ext)], [(3, inn)])
            gmsh.model.occ.intersect(tubo, [(3, caja)], removeTool=False)
        gmsh.model.occ.remove([(3, caja)], recursive=True)
        gmsh.model.occ.synchronize()
        ruta = str(tmp_path / "laminado.step")
        gmsh.write(ruta)
    finally:
        gmsh.finalize()
    modelo, _ = importar(ruta, str(tmp_path / "modelo"), tam_malla_mm=8.0)
    medias = [c["espesor_media_um"] for c in modelo.diagnostico["capas"].values()]
    assert np.allclose(medias, [2100, 760, 1600], atol=1.0)
    assert modelo.diagnostico["radio_curvatura_min_mm"] == pytest.approx(1500, rel=0.05)
