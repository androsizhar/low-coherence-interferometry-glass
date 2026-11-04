from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from motor.medicion.lectura import ErrorDeEntrada, validar_espectro


@dataclass
class ResultadoPicos:
    L: np.ndarray
    L_fft: np.ndarray
    amp: np.ndarray
    snr: np.ndarray
    sigma_L: np.ndarray
    banderas: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.banderas) == 0

    def caminos_capas(self) -> np.ndarray:
        return np.diff(self.L) / 2.0

    def cov_caminos(self) -> np.ndarray:
        s2 = self.sigma_L**2
        C = np.diag((s2[1:] + s2[:-1]) / 4.0)
        if s2.size > 2:
            C += np.diag(-s2[1:-1] / 4.0, 1) + np.diag(-s2[1:-1] / 4.0, -1)
        return C

    def sigma_caminos(self) -> np.ndarray:
        return 0.5 * np.hypot(self.sigma_L[1:], self.sigma_L[:-1])


def _eje_L(k: np.ndarray):
    N = k.size
    dk = (k[-1] - k[0]) / (N - 1)
    return 2 * np.pi * np.fft.fftfreq(N, d=dk), dk


def piso_de_ruido(amp: np.ndarray, L_eje: np.ndarray, L_corte: float) -> float:
    zona = amp[(L_eje > L_corte)]
    if zona.size < 32:
        zona = amp[L_eje > 0]
    return float(np.median(zona))


def _refinar_periodograma(k: np.ndarray, s: np.ndarray, L0: float, dL: float) -> float:
    kc = k - k.mean()

    def negP(L):
        return -np.abs(np.dot(s, np.exp(-1j * kc * L))) ** 2

    res = minimize_scalar(
        negP, bounds=(L0 - dL, L0 + dL), method="bounded", options={"xatol": 1e-5}
    )
    return float(res.x)


def detectar_picos(
    k: np.ndarray,
    I: np.ndarray,
    n_picos: int,
    ventanas: Optional[Sequence[Tuple[float, float]]] = None,
    L_min: float = 0.0,
    sep_min: float = 300.0,
    snr_min: float = 8.0,
    apodizacion: float = 0.2,
    metodo: str = "periodograma",
) -> ResultadoPicos:
    if metodo not in ("periodograma", "fase"):
        raise ErrorDeEntrada("metodo debe ser 'periodograma' o 'fase'")
    k, I = validar_espectro(k, I)
    if (
        isinstance(n_picos, (bool, np.bool_))
        or not isinstance(n_picos, (int, np.integer))
        or n_picos < 1
    ):
        raise ErrorDeEntrada("n_picos debe ser un entero >= 1")
    if not (
        np.isfinite(L_min)
        and L_min >= 0
        and np.isfinite(sep_min)
        and sep_min > 0
        and np.isfinite(snr_min)
        and snr_min > 0
        and np.isfinite(apodizacion)
        and 0 <= apodizacion <= 1
    ):
        raise ErrorDeEntrada(
            "L_min >= 0, sep_min/snr_min > 0 y apodizacion en [0,1], todos finitos"
        )
    banderas: List[str] = []
    if ventanas is not None:
        ventanas = np.asarray(ventanas, float)
        if ventanas.shape != (n_picos, 2) or not np.all(np.isfinite(ventanas)):
            raise ErrorDeEntrada("ventanas debe tener un par finito por pico")
        if np.any(ventanas[:, 0] >= ventanas[:, 1]):
            raise ErrorDeEntrada("cada ventana debe cumplir lo < hi")
        if np.any(ventanas[1:, 0] <= ventanas[:-1, 1]):
            banderas.append("ventanas_solapadas_o_desordenadas")
    N = k.size
    L_eje, dk = _eje_L(k)
    ventana = tukey(N, alpha=apodizacion) if apodizacion > 0 else np.ones(N)
    sw_senal = (I - np.mean(I)) * ventana
    S = np.fft.fft(sw_senal)
    amp = np.abs(S)
    pos = L_eje > 0
    dL = 2 * np.pi / (N * dk)

    if ventanas is not None:
        idx = []
        for lo, hi in ventanas:
            if lo <= 0 or hi > L_eje[pos].max():
                banderas.append(f"ventana_fuera_de_rango:{lo:.0f}-{hi:.0f}")
            m = np.where((L_eje >= lo) & (L_eje <= hi))[0]
            if m.size == 0:
                banderas.append(f"ventana_fuera_de_rango:{lo:.0f}-{hi:.0f}")
                idx.append(int(np.argmin(np.abs(L_eje - 0.5 * (lo + hi)))))
            else:
                elegido = int(m[np.argmax(amp[m])])
                if elegido in (m[0], m[-1]):
                    banderas.append(f"pico_en_borde_de_ventana:{len(idx)}")
                idx.append(elegido)
        L_ref_max = max(h for _, h in ventanas)
    else:
        zona = np.where(pos & (L_eje > L_min))[0]
        if zona.size < 3:
            raise ErrorDeEntrada("L_min deja menos de 3 bins en el rango observable")
        piso0 = np.median(amp[zona])
        pk, _ = find_peaks(amp[zona], height=snr_min * piso0, distance=max(1, int(sep_min / dL)))
        pk = zona[pk]
        top = sorted(pk, key=lambda i: amp[i], reverse=True)[:n_picos]
        idx = sorted(top)
        if len(idx) < n_picos:
            banderas.append(f"faltan_picos:{len(idx)}/{n_picos}")
            while len(idx) < n_picos:
                idx.append(idx[-1] if idx else 1)
        L_ref_max = float(L_eje[max(idx)])

    idx = np.array(idx)
    piso = piso_de_ruido(amp, L_eje, 1.5 * L_ref_max)
    if not np.isfinite(piso) or piso <= 0:
        piso = max(float(np.max(amp)) * np.finfo(float).eps, np.finfo(float).tiny)
    snr = amp[idx] / piso
    for i, s in enumerate(snr):
        if s < snr_min:
            banderas.append(f"snr_bajo:{i}:{s:.1f}")

    L_fft = L_eje[idx].astype(float)
    if np.any(np.diff(L_fft) <= 0):
        banderas.append("orden_invalido")

    L_fit = np.zeros(n_picos)
    sig = np.zeros(n_picos)
    k_bar = k.mean()
    for i in range(n_picos):
        dmin = np.min(np.abs(np.delete(L_fft, i) - L_fft[i])) if n_picos > 1 else 1000.0
        w = float(np.clip(0.35 * dmin, dL, 150.0))
        filtro = np.exp(-(((L_eje - L_fft[i]) / w) ** 2))
        z = np.fft.ifft(S * filtro)
        fase = np.unwrap(np.angle(z))
        W = np.abs(z) ** 2
        if W.sum() <= np.finfo(float).tiny:
            L_fit[i], sig[i] = L_fft[i], np.inf
            banderas.append(f"pico_sin_senal:{i}")
            continue
        X = k - k_bar
        sw = W.sum()
        xb = (W * X).sum() / sw
        yb = (W * fase).sum() / sw
        sxx = (W * (X - xb) ** 2).sum()
        L_fase = (W * (X - xb) * (fase - yb)).sum() / sxx
        if metodo == "fase":
            L_fit[i] = L_fase
        else:
            L_fit[i] = _refinar_periodograma(k, sw_senal, L_fft[i], dL)
        sigma_n = piso / (np.sqrt(np.log(2.0)) * np.linalg.norm(ventana))
        sig[i] = np.sqrt(2.0 * sigma_n**2 / (4.0 * sxx))
        if abs(L_fit[i] - L_fft[i]) > 3 * dL:
            banderas.append(f"refinamiento_inconsistente:{i}")

    if np.any(np.diff(L_fit) <= 0):
        banderas.append("orden_refinado_invalido")
    if n_picos > 1 and np.any(np.diff(L_fit) < max(sep_min, 2 * dL)):
        banderas.append("picos_no_resueltos")
    if not np.all(np.isfinite(L_fit)) or not np.all(np.isfinite(sig)):
        banderas.append("estimacion_no_finita")
    return ResultadoPicos(
        L=L_fit, L_fft=L_fft, amp=amp[idx], snr=snr, sigma_L=sig, banderas=banderas
    )
