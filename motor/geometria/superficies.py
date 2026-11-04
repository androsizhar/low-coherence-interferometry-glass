from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from motor.geometria.cad import Malla
from motor.medicion.lectura import ErrorDeEntrada


def puntos_interfaces(q: np.ndarray, n_in: np.ndarray, d: np.ndarray) -> np.ndarray:
    q, n_in, d = np.asarray(q, float), np.asarray(n_in, float), np.asarray(d, float)
    if (
        q.shape != n_in.shape
        or q.ndim != 2
        or q.shape[1] != 3
        or d.ndim != 2
        or d.shape[0] != len(q)
        or not d.shape[1]
    ):
        raise ErrorDeEntrada("puntos_interfaces: q y n_in (N,3), d (N,nL)")
    if not (
        np.all(np.isfinite(q))
        and np.all(np.isfinite(n_in))
        and np.all(np.isfinite(d))
        and np.all(d > 0)
    ):
        raise ErrorDeEntrada("puntos_interfaces: datos finitos y espesores positivos")
    if np.any(np.abs(np.linalg.norm(n_in, axis=1) - 1) > 1e-6):
        raise ErrorDeEntrada("las normales deben ser unitarias")
    D = np.concatenate([np.zeros((len(q), 1)), np.cumsum(d, axis=1)], axis=1)
    return q[None, :, :] + D.T[:, :, None] * n_in[None, :, :]


def desviacion_contra_cad(P: np.ndarray, malla_k: Malla) -> Tuple[np.ndarray, np.ndarray]:
    Q, dist, _ = malla_k.proyectar(P)
    return dist, Q


class ProcesoGaussiano2D:

    def __init__(self, ajustar_nugget: bool = True):
        self.ajustar_nugget = ajustar_nugget
        self.ell = self.s = self.nug = None

    @staticmethod
    def _k(A, B, ell, s):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return s**2 * np.exp(-0.5 * d2 / ell**2)

    def _H(self, X):
        return np.column_stack(
            [np.ones(len(X)), (X[:, 0] - self.c[0]) / self.esc, (X[:, 1] - self.c[1]) / self.esc]
        )

    def _nll(self, th, X, y, sig2, H):
        ell, s = np.exp(th[0]), np.exp(th[1])
        nug = np.exp(th[2]) if self.ajustar_nugget else 0.0
        K = self._k(X, X, ell, s) + np.diag(sig2 + nug**2) + 1e-10 * s**2 * np.eye(len(X))
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        Ki_y = cho_solve(cf, y)
        Ki_H = cho_solve(cf, H)
        A = H.T @ Ki_H
        try:
            beta = np.linalg.solve(A, H.T @ Ki_y)
            logdetA = np.linalg.slogdet(A)[1]
        except np.linalg.LinAlgError:
            return 1e12
        logdetK = 2.0 * np.log(np.diag(cf[0])).sum()
        r = y - H @ beta
        quad = float(r @ cho_solve(cf, r))
        return 0.5 * (quad + logdetK + logdetA)

    def ajustar(self, X: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> "ProcesoGaussiano2D":
        X, y, sigma = np.asarray(X, float), np.asarray(y, float), np.asarray(sigma, float)
        if X.ndim != 2 or X.shape[1] != 2 or y.shape != (len(X),) or sigma.shape != y.shape:
            raise ErrorDeEntrada("GP: X (N,2), y (N,), sigma (N,)")
        if len(X) < 8:
            raise ErrorDeEntrada("GP: se necesitan >= 8 puntos")
        if not (
            np.all(np.isfinite(X))
            and np.all(np.isfinite(y))
            and np.all(np.isfinite(sigma))
            and np.all(sigma > 0)
        ):
            raise ErrorDeEntrada("GP: datos no finitos o sigma <= 0")
        self.c = X.mean(0)
        self.esc = float(np.ptp(X, axis=0).max()) or 1.0
        H = self._H(X)
        if np.linalg.matrix_rank(H) < 3:
            raise ErrorDeEntrada("GP: el mapa 2-D requiere puntos no colineales")
        sig2 = sigma**2
        dom = float(np.ptp(X, axis=0).max())
        from scipy.spatial import cKDTree

        dnn = cKDTree(X).query(X, k=2)[0][:, 1]
        l_lo, l_hi = max(0.5 * np.median(dnn), 1e-9), 2.0 * dom
        s0 = max(float(np.std(y)), 1e-6)
        mejor = None
        for l0 in (0.15 * dom, 0.4 * dom, 1.0 * dom):
            x0 = [
                np.log(np.clip(l0, l_lo, l_hi)),
                np.log(s0),
                np.log(0.3 * float(np.median(sigma)) + 1e-9),
            ]
            bounds = [
                (np.log(l_lo), np.log(l_hi)),
                (np.log(1e-6 * s0 + 1e-12), np.log(1e3 * s0 + 1e-6)),
                (np.log(1e-9), np.log(10 * float(np.max(sigma)) + 1e-9)),
            ]
            if not self.ajustar_nugget:
                bounds[2] = (x0[2], x0[2])
            res = minimize(self._nll, x0, args=(X, y, sig2, H), method="L-BFGS-B", bounds=bounds)
            if mejor is None or res.fun < mejor.fun:
                mejor = res
        self.ell, self.s = float(np.exp(mejor.x[0])), float(np.exp(mejor.x[1]))
        self.nug = float(np.exp(mejor.x[2])) if self.ajustar_nugget else 0.0
        self.X, self.y, self.sig2 = X, y, sig2
        K = (
            self._k(X, X, self.ell, self.s)
            + np.diag(sig2 + self.nug**2)
            + 1e-10 * self.s**2 * np.eye(len(X))
        )
        self._cf = cho_factor(K, lower=True)
        self._H_ = H
        self._KiH = cho_solve(self._cf, H)
        self._A = H.T @ self._KiH
        self._beta = np.linalg.solve(self._A, H.T @ cho_solve(self._cf, y))
        self._alpha = cho_solve(self._cf, y - H @ self._beta)
        self._Ainv = np.linalg.inv(self._A)
        return self

    def predecir(self, Xs: np.ndarray, gradiente: bool = False, lote: int = 1024):
        if self.ell is None:
            raise ErrorDeEntrada("GP: ajustar antes de predecir")
        Xs = np.asarray(Xs, float)
        if Xs.ndim != 2 or Xs.shape[1] != 2 or not np.all(np.isfinite(Xs)):
            raise ErrorDeEntrada("GP: puntos de prediccion finitos (M,2)")
        if not isinstance(lote, (int, np.integer)) or lote < 1:
            raise ErrorDeEntrada("GP: lote debe ser entero positivo")
        if len(Xs) > lote:
            resultados = [
                self.predecir(Xs[i : i + lote], gradiente=gradiente, lote=lote)
                for i in range(0, len(Xs), lote)
            ]
            return tuple(
                np.concatenate([r[j] for r in resultados], axis=0)
                for j in range(4 if gradiente else 2)
            )
        Ks = self._k(Xs, self.X, self.ell, self.s)
        Hs = self._H(Xs)
        media = Hs @ self._beta + Ks @ self._alpha
        KiKs = cho_solve(self._cf, Ks.T)
        var = self.s**2 - np.einsum("mn,nm->m", Ks, KiKs)
        R = Hs - (KiKs.T @ self._H_)
        var = var + np.einsum("mi,ij,mj->m", R, self._Ainv, R)
        std = np.sqrt(np.maximum(var, 0.0))
        if not gradiente:
            return media, std
        dH = [
            np.column_stack([np.zeros(len(Xs)), np.full(len(Xs), 1 / self.esc), np.zeros(len(Xs))]),
            np.column_stack([np.zeros(len(Xs)), np.zeros(len(Xs)), np.full(len(Xs), 1 / self.esc)]),
        ]
        dK = [-(Xs[:, a][:, None] - self.X[:, a][None, :]) / self.ell**2 * Ks for a in range(2)]
        g = np.column_stack([dH[a] @ self._beta + dK[a] @ self._alpha for a in range(2)])
        KiDK = [cho_solve(self._cf, dK[a].T) for a in range(2)]
        Rg = [dH[a] - KiDK[a].T @ self._H_ for a in range(2)]
        cov = np.zeros((len(Xs), 2, 2))
        for a in range(2):
            for b in range(2):
                prior = (self.s**2 / self.ell**2) if a == b else 0.0
                cov[:, a, b] = (
                    prior
                    - np.einsum("mn,nm->m", dK[a], KiDK[b])
                    + np.einsum("mi,ij,mj->m", Rg[a], self._Ainv, Rg[b])
                )
        cov = (cov + cov.transpose(0, 2, 1)) / 2
        valores, vectores = np.linalg.eigh(cov)
        cov = np.einsum("mik,mk,mjk->mij", vectores, np.maximum(valores, 0), vectores)
        return media, std, g, cov
