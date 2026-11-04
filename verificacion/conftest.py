import numpy as np
import pytest


@pytest.fixture
def datos_visual():
    X, Y = np.meshgrid(np.linspace(-3000, 3000, 7), np.linspace(-2000, 2000, 5))
    d = np.stack([2100 + X * 0.0005, 760 + X * 0.00035, 1600 + Y * 0.0002])
    Z0 = 1e-6 * (X * X + Y * Y) + X * 0.01
    S = [np.stack([X, Y, Z0], axis=-1)]
    for j in range(3):
        superficie = S[-1].copy()
        superficie[..., 2] += d[j]
        S.append(superficie)
    nominal = np.stack(S).copy()
    nominal[..., 2] -= 2
    return {
        "X": X,
        "Y": Y,
        "d_media": d,
        "d_std": np.ones_like(d),
        "delta_media": np.ones((4, 5, 7)) * 2,
        "delta_std": np.ones((4, 5, 7)),
        "w": X * 0 + 0.35,
        "sigma_w": X * 0 + 0.01,
        "P_fuera_d": d * 0,
        "mascara_soporte": np.ones_like(X, bool),
        "interfaces": np.stack(S),
        "interfaces_nominales": nominal,
    }


@pytest.fixture
def informe_visual():
    return {
        "registro": {"modo": "fijo", "covarianza": np.zeros((6, 6)).tolist()},
        "estado_validacion": {
            "origen_datos": "simulado",
            "validacion_experimental_completa": False,
        },
        "calibracion": {"ok": True, "banderas": []},
        "banderas": ["solo_datos_simulados"],
        "espesor": {
            "vidrio_ext": {"media_um": 2100},
            "pvb": {"media_um": 760},
            "vidrio_int": {"media_um": 1600},
        },
        "cuna_pvb_mrad": {"media": 0.35, "sigma_media": 0.01},
        "gemelo": {"imagen_doble_arcmin": 4.2},
    }
