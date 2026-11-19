"""Métricas de avaliação: CCC de Lin (primária), MAE, RMSE, quadrante e bootstrap.

O CCC é reportado separadamente para valência e arousal. Diferente da correlação
de Pearson, ele penaliza também o deslocamento e a diferença de escala entre as
séries, que é o que se quer medir em rótulos sem calibração.

Todas as métricas operam em [-1, 1] (ver :mod:`tcc_mer.scale`) e descartam pares
com ``NaN``.
"""

from collections.abc import Callable

import numpy as np

from .quadrant import QUADRANTS, quadrants

__all__ = [
    "ccc",
    "mae",
    "rmse",
    "quadrant_accuracy",
    "quadrant_f1_macro",
    "bootstrap_ci",
]


def _paired_valid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alinha ``x`` e ``y`` removendo as posições com ``NaN`` em qualquer um."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shapes incompatíveis: {x.shape} vs {y.shape}")
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Coeficiente de correlação de concordância de Lin entre ``x`` e ``y``.

    ``CCC = 2·cov / (var_x + var_y + (mean_x - mean_y)²)``, com variância e
    covariância populacionais (divisor N). O termo ``(mean_x - mean_y)²`` é o que
    captura o viés de deslocamento.

    Retorna ``nan`` se restarem menos de 2 pares ou se o denominador for zero.
    """
    x, y = _paired_valid(x, y)
    if x.size < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    denominator = x.var() + y.var() + (mx - my) ** 2
    if denominator == 0:
        return float("nan")
    covariance = ((x - mx) * (y - my)).mean()
    return float(2.0 * covariance / denominator)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """Erro absoluto médio (ignora pares com ``NaN``)."""
    x, y = _paired_valid(x, y)
    return float(np.mean(np.abs(x - y))) if x.size else float("nan")


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Raiz do erro quadrático médio (ignora pares com ``NaN``)."""
    x, y = _paired_valid(x, y)
    return float(np.sqrt(np.mean((x - y) ** 2))) if x.size else float("nan")


def _paired_quadrants(v_true, a_true, v_pred, a_pred) -> tuple[np.ndarray, np.ndarray]:
    """Quadrantes verdadeiro e predito, só das faixas em que ambos existem."""
    q_true = quadrants(v_true, a_true).ravel()
    q_pred = quadrants(v_pred, a_pred).ravel()
    valid = (q_true != "") & (q_pred != "")
    return q_true[valid], q_pred[valid]


def quadrant_accuracy(v_true, a_true, v_pred, a_pred) -> float:
    """Fração de faixas em que o quadrante predito coincide com o verdadeiro."""
    q_true, q_pred = _paired_quadrants(v_true, a_true, v_pred, a_pred)
    return float(np.mean(q_true == q_pred)) if q_true.size else float("nan")


def quadrant_f1_macro(v_true, a_true, v_pred, a_pred) -> float:
    """F1-macro sobre os quatro quadrantes (média simples).

    Quadrantes sem suporte e sem predição não entram na média.
    """
    q_true, q_pred = _paired_quadrants(v_true, a_true, v_pred, a_pred)
    scores = []
    for q in QUADRANTS:
        tp = int(np.sum((q_pred == q) & (q_true == q)))
        fp = int(np.sum((q_pred == q) & (q_true != q)))
        fn = int(np.sum((q_pred != q) & (q_true == q)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """Intervalo de confiança por bootstrap percentil para qualquer estatística.

    Reamostra **linhas** de ``values`` com reposição, o que preserva o pareamento.
    Para estatísticas pareadas como o CCC, passe uma matriz ``(n, 2)``::

        bootstrap_ci(np.column_stack([x, y]), lambda m: ccc(m[:, 0], m[:, 1]))

    Args:
        values: array ``(n,)`` ou ``(n, k)``.
        stat_fn: recebe uma reamostra e devolve um ``float``.
        n_boot: número de reamostragens.
        alpha: nível do intervalo (0,05 dá IC de 95%).
        seed: semente, ou um ``Generator`` já criado para compartilhar o fluxo de sorteios
            entre várias chamadas.

    Returns:
        ``(low, high)``. Reamostras que devolvem ``nan`` são descartadas.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.array([stat_fn(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )
