import numpy as np
import pytest


@pytest.mark.parametrize("pose", ["sintetica", "real", "icp"])
def test_pdf_headless_a4_veredicto_etiquetas(tmp_path, datos_visual, informe_visual, pose):
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    from motor.salida.reporte_pdf import generar_pdf

    esperado = "pose sintética exacta"
    if pose == "real":
        informe_visual["estado_validacion"]["origen_datos"] = "real_declarado"
        C = np.eye(6) * 0.000123456789012345 + np.ones((6, 6)) * 0.000012345678901234
        informe_visual["registro"]["covarianza"] = C.tolist()
        esperado = "pose declarada (real) con covarianza X"
    elif pose == "icp":
        informe_visual["registro"]["modo"] = "icp_forma"
        esperado = "mejor ajuste: forma relativa, no absoluta"
    informe_visual["presupuesto_incertidumbre"] = [
        {
            "componente": "referencia",
            "estado": "declarada",
            "sigma_um": 1,
            "descripcion": "Muestra A conocida",
        },
        {
            "componente": "modelo",
            "estado": "pendiente",
            "sigma_um": None,
            "descripcion": "No disponible; falta contraste real",
        },
    ]
    ruta = generar_pdf(informe_visual, datos_visual, tmp_path / "informe.pdf")
    assert ruta.stat().st_size > 20000
    pdf = pypdf.PdfReader(ruta)
    assert len(pdf.pages) == 5
    for pagina in pdf.pages:
        assert abs(float(pagina.mediabox.width) - 841.89) < 0.1
        assert abs(float(pagina.mediabox.height) - 595.28) < 0.1
        texto = pagina.extract_text()
        assert "no concluyente" in texto
        assert esperado in texto
        assert "predicción exploratoria" in texto
    texto = " ".join(p.extract_text() for p in pdf.pages)
    assert "no disponible" in texto
    assert "Ningún valor sintético es precisión esperada de un montaje real" in texto
    assert "µm" in texto and "mrad" in texto and "minutos de arco" in texto
