"""Pipeline de calibração + fusão avaliado com validação cruzada aninhada (Fase D).

Junta as peças da Fase A/D num fluxo honesto: para cada dobra do k-fold, a
calibração afim (por LF) **e** os pesos de fusão são ajustados **só no treino** e
aplicados no teste. Isso evita a armadilha de calibrar e avaliar no mesmo fold —
o número reportável é o **CCC fora-da-amostra**.

Um "canal" pode ter 1 LF (áudio: só LF1) ou várias (letra: LF2 + LF3). Com 1 LF,
a fusão é trivial (peso 1) e o pipeline vira só calibração afim.

- ``nested_kfold_ccc``: o número reportável (CCC OOS) + pesos médios.
- ``fit_channel`` / ``apply_channel``: ajusta no gold inteiro e produz a estimativa
  calibrada+fundida do canal para QUALQUER faixa (usado para calcular a dissociação).
"""
from __future__ import annotations

import numpy as np

from .calibration import AffineCalibration
from .fusion import fit_fusion_weights
from .metrics import ccc

__all__ = ["nested_kfold_ccc", "fit_channel", "apply_channel"]


def _valid_rows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ~np.isnan(y) & ~np.isnan(X).any(axis=1)


def fit_channel(X_gold: np.ndarray, y_gold: np.ndarray):
    """Ajusta calibração afim por LF + pesos de fusão no gold inteiro.

    Args:
        X_gold: LFs cruas nas faixas do gold, shape ``(n, n_lfs)``.
        y_gold: gold humano do canal/dimensão, shape ``(n,)``.

    Returns:
        ``(cals, weights)`` — lista de :class:`AffineCalibration` (uma por LF) e o
        vetor de pesos de fusão (soma 1). Ajustados só onde há pares válidos.
    """
    X_gold = np.asarray(X_gold, float)
    y_gold = np.asarray(y_gold, float)
    if X_gold.ndim != 2:
        raise ValueError("X_gold deve ser 2-D (n, n_lfs)")
    mask = _valid_rows(X_gold, y_gold)
    Xv, yv = X_gold[mask], y_gold[mask]
    n_lfs = Xv.shape[1]
    cals = [AffineCalibration.fit(Xv[:, j], yv) for j in range(n_lfs)]
    Xcal = np.column_stack([cals[j].apply(Xv[:, j]) for j in range(n_lfs)])
    weights = fit_fusion_weights(Xcal, yv) if n_lfs > 1 else np.array([1.0])
    return cals, weights


def apply_channel(cals, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Aplica calibração+fusão a LFs cruas -> estimativa do canal (``NaN`` propaga)."""
    X = np.asarray(X, float)
    Xcal = np.column_stack([cals[j].apply(X[:, j]) for j in range(X.shape[1])])
    return Xcal @ np.asarray(weights, float)


def nested_kfold_ccc(
    X: np.ndarray, y: np.ndarray, k: int = 5, seed: int | None = 0
) -> tuple[float, np.ndarray, int]:
    """CCC fora-da-amostra do canal (calibração+fusão aninhadas no k-fold).

    Em cada dobra: ajusta afim por LF e pesos de fusão no treino, prediz o teste;
    concatena as predições de teste e computa um único CCC contra o gold.

    Args:
        X: LFs cruas, shape ``(n, n_lfs)``.
        y: gold do canal/dimensão, shape ``(n,)``.
        k: número de dobras.
        seed: semente do embaralhamento.

    Returns:
        ``(ccc_oos, pesos_medios, n_usado)``. Faixas com qualquer ``NaN`` (LF ou
        gold) são descartadas antes.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2:
        raise ValueError("X deve ser 2-D (n, n_lfs)")
    mask = _valid_rows(X, y)
    X, y = X[mask], y[mask]
    n = len(y)
    k = min(k, n)
    if k < 2:
        return (float("nan"), np.full(X.shape[1], np.nan), n)

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    oof = np.full(n, np.nan)
    weights = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        cals, w = fit_channel(X[tr], y[tr])
        oof[te] = apply_channel(cals, w, X[te])
        weights.append(w)
    return float(ccc(oof, y)), np.mean(np.vstack(weights), axis=0), n
