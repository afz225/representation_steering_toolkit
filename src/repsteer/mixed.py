"""Linear mixed models with crossed random effects, fitted by maximum likelihood.

Implements the profiled-deviance formulation of Bates, Mächler, Bolker & Walker (2015,
"Fitting linear mixed-effects models using lme4", eqs. 13-34): for relative covariance
factor Λ(θ) the penalized least-squares system is solved with q×q and p×p Cholesky
factors, so one deviance evaluation costs O(n q^2) instead of the O(n^3) of a dense
single-group fit. Each random-effect term is ``(columns | grouping factor)`` with an
unstructured (correlated) covariance: per level of the factor, b ~ N(0, σ² T Tᵀ) with T
lower-triangular. θ (the free entries of every T) is optimized by L-BFGS-B with the
diagonal of T bounded at zero, so variance components on the boundary converge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg, optimize


@dataclass(frozen=True)
class RandomTerm:
    """``(columns | factor)``: ``codes`` are level indices 0..n_levels-1 per row and
    ``columns`` the n×p random-effect design within each level (e.g. [1, x])."""

    name: str
    codes: np.ndarray
    columns: np.ndarray

    @property
    def n_levels(self) -> int:
        return int(self.codes.max()) + 1

    @property
    def p(self) -> int:
        return int(self.columns.shape[1])


@dataclass(frozen=True)
class MixedFit:
    beta: np.ndarray
    cov_beta: np.ndarray
    sigma2: float
    llf: float
    covariances: dict[str, np.ndarray]  # per term: p×p random-effect covariance σ² T Tᵀ
    fitted_fixed: np.ndarray
    converged: bool

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_beta))


def _z_matrix(terms: Sequence[RandomTerm]) -> np.ndarray:
    n = len(terms[0].codes)
    blocks = []
    for term in terms:
        block = np.zeros((n, term.n_levels * term.p))
        rows = np.arange(n)
        for column in range(term.p):
            block[rows, term.codes * term.p + column] = term.columns[:, column]
        blocks.append(block)
    return np.hstack(blocks)


def _theta_layout(terms: Sequence[RandomTerm]):
    layout, bounds, start = [], [], []
    for term in terms:
        rows, cols = np.tril_indices(term.p)
        layout.append((rows, cols))
        for row, col in zip(rows, cols):
            bounds.append((0.0, None) if row == col else (None, None))
            start.append(1.0 if row == col else 0.0)
    return layout, bounds, np.asarray(start)


def _lambda_t(theta: np.ndarray, terms: Sequence[RandomTerm], layout) -> tuple[np.ndarray, list[np.ndarray]]:
    """Λᵀ is block diagonal: per term, I_levels ⊗ T (stored as Λ; ZΛ uses it directly)."""

    factors, offset = [], 0
    for term, (rows, cols) in zip(terms, layout):
        t = np.zeros((term.p, term.p))
        t[rows, cols] = theta[offset:offset + len(rows)]
        offset += len(rows)
        factors.append(t)
    blocks = [np.kron(np.eye(term.n_levels), t) for term, t in zip(terms, factors)]
    return linalg.block_diag(*blocks), factors


def _solve(theta, y, x, z, terms, layout):
    lam, factors = _lambda_t(theta, terms, layout)
    zl = z @ lam
    q = zl.shape[1]
    chol_l = linalg.cholesky(zl.T @ zl + np.eye(q), lower=True)
    cu = linalg.solve_triangular(chol_l, zl.T @ y, lower=True)
    rzx = linalg.solve_triangular(chol_l, zl.T @ x, lower=True)
    rxtrx = x.T @ x - rzx.T @ rzx
    beta = linalg.solve(rxtrx, x.T @ y - rzx.T @ cu, assume_a="pos")
    u = linalg.solve_triangular(chol_l.T, cu - rzx @ beta, lower=False)
    residual = y - x @ beta - zl @ u
    r2 = float(residual @ residual + u @ u)
    n = len(y)
    deviance = 2.0 * float(np.sum(np.log(np.diag(chol_l)))) + n * (1.0 + np.log(2.0 * np.pi * r2 / n))
    return deviance, beta, rxtrx, r2, factors


def fit_mixed(y: np.ndarray, x: np.ndarray, terms: Sequence[RandomTerm]) -> MixedFit:
    """ML fit of y = Xβ + Σ_terms Z_t b_t + ε with crossed, per-term unstructured covariances."""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    z = _z_matrix(terms)
    layout, bounds, start = _theta_layout(terms)

    def objective(theta):
        try:
            return _solve(theta, y, x, z, terms, layout)[0]
        except (linalg.LinAlgError, ValueError):
            return np.inf

    best = None
    for scale in (1.0, 0.1):
        result = optimize.minimize(objective, start * scale, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    deviance, beta, rxtrx, r2, factors = _solve(best.x, y, x, z, terms, layout)
    sigma2 = r2 / len(y)
    return MixedFit(
        beta=beta, cov_beta=sigma2 * linalg.inv(rxtrx), sigma2=sigma2, llf=-0.5 * deviance,
        covariances={term.name: sigma2 * t @ t.T for term, t in zip(terms, factors)},
        fitted_fixed=x @ beta, converged=bool(best.success) and np.isfinite(deviance))


def marginal_r2(fit: MixedFit, terms: Sequence[RandomTerm]) -> float:
    """Nakagawa & Schielzeth marginal R² with random slopes (Johnson 2014):
    random variance = mean over rows of z_iᵀ Σ_t z_i for every term."""

    var_fixed = float(np.var(fit.fitted_fixed))
    var_random = sum(float(np.mean(np.einsum("ij,jk,ik->i", term.columns, fit.covariances[term.name],
                                             term.columns))) for term in terms)
    return var_fixed / (var_fixed + var_random + fit.sigma2)
