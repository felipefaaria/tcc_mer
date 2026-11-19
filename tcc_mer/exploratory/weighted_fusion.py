"""Fusão tardia calibrada — pesos por dimensão que maximizam o CCC no gold.

Fusão *tardia*: cada LF produz seu (V, A) independentemente e só depois
combinamos numa média ponderada por dimensão (oposto de *early fusion*, que
concatenaria features cruas). Preferimos tardia porque preserva a saída modal de
cada LF — que é o que precisamos para calcular a dissociação.

Os pesos são aprendidos **maximizando o CCC contra o gold** (equivalente a
minimizar ``-CCC`` como perda), restritos ao **simplex** (não-negativos, soma 1)
para interpretabilidade — cada peso vira "quanto confiamos naquela LF para
aquela dimensão". Empiricamente, o peso de áudio domina em Arousal e o de letra
(léxico+LLM) domina em Valência — esse quadro de pesos é um resultado de PP1.

Rigor estatístico: o gold set é pequeno. Ajuste os pesos com validação cruzada
k-fold e **reporte CCC fora-da-amostra** — ajustar e avaliar no mesmo punhado de
faixas infla tudo.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .metrics import ccc

__all__ = ["fit_fusion_weights", "fuse", "kfold_out_of_sample_ccc"]


def fuse(calibrated: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combina as saídas calibradas das LFs numa predição fundida.

    Args:
        calibrated: saídas já calibradas, shape ``(n_faixas, n_lfs)``.
        weights: pesos por LF, shape ``(n_lfs,)``.

    Returns:
        Vetor fundido, shape ``(n_faixas,)``.
    """
    calibrated = np.asarray(calibrated, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return calibrated @ weights


def fit_fusion_weights(calibrated: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Aprende pesos no simplex que maximizam o CCC da fusão contra o gold.

    Args:
        calibrated: saídas **já calibradas** das LFs, shape ``(n_faixas, n_lfs)``.
        gold: rótulos humanos, shape ``(n_faixas,)``.

    Returns:
        Vetor de pesos ``(n_lfs,)``, não-negativos e somando 1.

    Notas:
        Ajuste **e** avaliação no mesmo conjunto inflam o CCC. Use
        :func:`kfold_out_of_sample_ccc` para o número reportável.
    """
    calibrated = np.asarray(calibrated, dtype=float)
    gold = np.asarray(gold, dtype=float).ravel()
    n_lfs = calibrated.shape[1]

    def neg_ccc(w: np.ndarray) -> float:
        return -ccc(calibrated @ w, gold)

    w0 = np.full(n_lfs, 1.0 / n_lfs)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n_lfs
    res = minimize(neg_ccc, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    # Reprojeta no simplex caso o otimizador devolva soma ligeiramente != 1.
    return w / total if total > 0 else w0


def kfold_out_of_sample_ccc(
    calibrated: np.ndarray,
    gold: np.ndarray,
    k: int = 5,
    seed: int | None = None,
) -> tuple[float, np.ndarray]:
    """CCC fora-da-amostra dos pesos de fusão via validação cruzada k-fold.

    Em cada fold, aprende os pesos no treino e prediz o teste; concatena as
    predições de teste de todos os folds e computa um único CCC contra o gold —
    o número honesto para reportar.

    Args:
        calibrated: saídas já calibradas das LFs, shape ``(n_faixas, n_lfs)``.
        gold: rótulos humanos, shape ``(n_faixas,)``.
        k: número de folds.
        seed: semente do embaralhamento (reprodutibilidade).

    Returns:
        ``(ccc_oos, pesos_medios)`` — CCC fora-da-amostra e a média dos vetores de
        peso aprendidos por fold (visão descritiva do quadro de confiança por LF).
    """
    calibrated = np.asarray(calibrated, dtype=float)
    gold = np.asarray(gold, dtype=float).ravel()
    n = calibrated.shape[0]
    k = min(k, n)
    if k < 2:
        raise ValueError("k-fold requer ao menos 2 faixas / 2 folds")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)

    oof_pred = np.full(n, np.nan)
    weights_per_fold = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        w = fit_fusion_weights(calibrated[train_idx], gold[train_idx])
        oof_pred[test_idx] = calibrated[test_idx] @ w
        weights_per_fold.append(w)

    ccc_oos = ccc(oof_pred, gold)
    mean_weights = np.mean(np.vstack(weights_per_fold), axis=0)
    return float(ccc_oos), mean_weights
