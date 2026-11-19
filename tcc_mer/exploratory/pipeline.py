"""Calibração e fusão ajustadas no gold set, avaliadas com validação cruzada aninhada.

Em cada dobra, a calibração afim (por LF) e os pesos de fusão são ajustados só no
treino e aplicados no teste, para não calibrar e avaliar nas mesmas faixas. Um
"canal" tem uma ou mais LFs; com uma só, a fusão é trivial (peso 1) e o pipeline se
reduz à calibração afim.

* :func:`nested_kfold_ccc`: o CCC fora da amostra e os pesos médios.
* :func:`fit_channel` e :func:`apply_channel`: ajustam no gold inteiro e aplicam a
  qualquer faixa.
"""

import numpy as np

from ..metrics import ccc
from .calibration import AffineCalibration
from .weighted_fusion import fit_fusion_weights

__all__ = ["nested_kfold_ccc", "fit_channel", "apply_channel"]


def _valid_rows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ~np.isnan(y) & ~np.isnan(X).any(axis=1)


def fit_channel(
    X_gold: np.ndarray, y_gold: np.ndarray
) -> tuple[list[AffineCalibration], np.ndarray]:
    """Ajusta a calibração afim de cada LF e os pesos de fusão no gold.

    Args:
        X_gold: saídas brutas das LFs nas faixas do gold, shape ``(n, n_lfs)``.
        y_gold: gold humano do canal e da dimensão, shape ``(n,)``.

    Returns:
        ``(calibrações, pesos)``, ajustados só nos pares válidos.
    """
    X_gold = np.asarray(X_gold, dtype=float)
    y_gold = np.asarray(y_gold, dtype=float)
    if X_gold.ndim != 2:
        raise ValueError("X_gold deve ser 2-D (n, n_lfs)")
    valid = _valid_rows(X_gold, y_gold)
    X, y = X_gold[valid], y_gold[valid]
    n_lfs = X.shape[1]
    calibrations = [AffineCalibration.fit(X[:, j], y) for j in range(n_lfs)]
    calibrated = np.column_stack([calibrations[j].apply(X[:, j]) for j in range(n_lfs)])
    weights = fit_fusion_weights(calibrated, y) if n_lfs > 1 else np.array([1.0])
    return calibrations, weights


def apply_channel(
    calibrations: list[AffineCalibration], weights: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Aplica calibração e fusão às saídas brutas das LFs (``NaN`` propaga)."""
    X = np.asarray(X, dtype=float)
    calibrated = np.column_stack([calibrations[j].apply(X[:, j]) for j in range(X.shape[1])])
    return calibrated @ np.asarray(weights, dtype=float)


def nested_kfold_ccc(
    X: np.ndarray, y: np.ndarray, k: int = 5, seed: int | None = 0
) -> tuple[float, np.ndarray, int]:
    """CCC fora da amostra de um canal, com calibração e fusão aninhadas no k-fold.

    Args:
        X: saídas brutas das LFs, shape ``(n, n_lfs)``.
        y: gold do canal e da dimensão, shape ``(n,)``.
        k: número de dobras.
        seed: semente do embaralhamento.

    Returns:
        ``(ccc_oos, pesos_médios, n_usado)``. Faixas com ``NaN`` na LF ou no gold são
        descartadas antes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X deve ser 2-D (n, n_lfs)")
    valid = _valid_rows(X, y)
    X, y = X[valid], y[valid]
    n = len(y)
    k = min(k, n)
    if k < 2:
        return float("nan"), np.full(X.shape[1], np.nan), n

    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    predictions = np.full(n, np.nan)
    fold_weights = []
    for i, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        calibrations, weights = fit_channel(X[train], y[train])
        predictions[test] = apply_channel(calibrations, weights, X[test])
        fold_weights.append(weights)
    return float(ccc(predictions, y)), np.mean(np.vstack(fold_weights), axis=0), n
