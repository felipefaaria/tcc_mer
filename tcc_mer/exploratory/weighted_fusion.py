"""Fusão ponderada cujos pesos maximizam o CCC contra o gold set.

Cada LF entra com um peso não negativo e a soma dos pesos é 1 (simplex), o que torna
o peso interpretável como a confiança numa LF para uma dimensão. Como o gold é
pequeno, o número reportável é o CCC fora da amostra (:func:`kfold_out_of_sample_ccc`).
"""

import numpy as np
from scipy.optimize import minimize

from ..metrics import ccc

__all__ = ["fit_fusion_weights", "fuse", "kfold_out_of_sample_ccc"]


def fuse(calibrated: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combinação ponderada das saídas calibradas, shape ``(n_faixas,)``."""
    return np.asarray(calibrated, dtype=float) @ np.asarray(weights, dtype=float)


def fit_fusion_weights(calibrated: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Pesos no simplex que maximizam o CCC da fusão contra o gold.

    Args:
        calibrated: saídas já calibradas das LFs, shape ``(n_faixas, n_lfs)``.
        gold: rótulos humanos, shape ``(n_faixas,)``.

    Returns:
        Pesos ``(n_lfs,)``, não negativos e somando 1.
    """
    calibrated = np.asarray(calibrated, dtype=float)
    gold = np.asarray(gold, dtype=float).ravel()
    n_lfs = calibrated.shape[1]

    result = minimize(
        lambda w: -ccc(calibrated @ w, gold),
        np.full(n_lfs, 1.0 / n_lfs),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_lfs,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    return weights / total if total > 0 else np.full(n_lfs, 1.0 / n_lfs)


def kfold_out_of_sample_ccc(
    calibrated: np.ndarray,
    gold: np.ndarray,
    k: int = 5,
    seed: int | None = None,
) -> tuple[float, np.ndarray]:
    """CCC fora da amostra dos pesos de fusão, por validação cruzada k-fold.

    Em cada dobra, aprende os pesos no treino e prediz o teste. As predições de
    teste são concatenadas e um único CCC é calculado contra o gold.

    Returns:
        ``(ccc_oos, pesos_médios)``, a média dos pesos aprendidos em cada dobra.

    Raises:
        ValueError: com menos de 2 dobras possíveis.
    """
    calibrated = np.asarray(calibrated, dtype=float)
    gold = np.asarray(gold, dtype=float).ravel()
    n = calibrated.shape[0]
    k = min(k, n)
    if k < 2:
        raise ValueError("k-fold requer ao menos 2 faixas / 2 folds")

    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    predictions = np.full(n, np.nan)
    fold_weights = []
    for i, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        weights = fit_fusion_weights(calibrated[train], gold[train])
        predictions[test] = calibrated[test] @ weights
        fold_weights.append(weights)
    return float(ccc(predictions, gold)), np.mean(np.vstack(fold_weights), axis=0)
