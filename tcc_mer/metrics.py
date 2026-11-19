"""Métricas de avaliação — CCC (métrica primária), MAE/RMSE, quadrante, bootstrap.

A métrica primária é o **CCC de Lin** (Lin's Concordance Correlation Coefficient),
reportado **separadamente** para Valência e Arousal. Diferente da Pearson do TCC1,
o CCC penaliza tanto o desalinhamento de forma quanto o de localização/escala
(viés de deslocamento e diferença de variância) — que é exatamente o que
queremos medir em PP1.

Todas as métricas operam em escala interna [-1, 1] (ver :mod:`src.scale`) e
tratam pares com ``NaN`` como ausentes (removidos par-a-par).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "ccc",
    "mae",
    "rmse",
    "quadrant_from_va",
    "quadrant_accuracy",
    "quadrant_f1_macro",
    "bootstrap_ci",
]


def _paired_valid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alinha ``x`` e ``y`` removendo posições com ``NaN`` em qualquer um dos dois."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shapes incompatíveis: {x.shape} vs {y.shape}")
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Concordance Correlation Coefficient de Lin entre ``x`` e ``y``.

    ``CCC = 2*cov / (var_x + var_y + (mean_x - mean_y)**2)``, usando variância/
    covariância populacionais (divisor N). O termo ``(mean_x - mean_y)**2`` no
    denominador é o que captura o viés de deslocamento — perfeito para PP1.

    Pares com ``NaN`` são descartados. Retorna ``nan`` se restarem <2 pares ou se
    ambas as séries forem constantes (denominador zero).
    """
    x, y = _paired_valid(x, y)
    if x.size < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()  # populacional (ddof=0)
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float(2.0 * cov / denom)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """Erro absoluto médio (ignora pares com ``NaN``)."""
    x, y = _paired_valid(x, y)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x - y)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Raiz do erro quadrático médio (ignora pares com ``NaN``)."""
    x, y = _paired_valid(x, y)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def quadrant_from_va(v: float, a: float) -> str:
    """Mapeia (V, A) em [-1, 1]^2 para o quadrante do circumplexo de Russell.

    Convenção de borda: ``>= 0`` conta como positivo (eixo pertence ao lado +).

    - Q1: V+, A+  (eufórico/animado)
    - Q2: V-, A+  (tenso/raivoso)
    - Q3: V-, A-  (triste/deprimido)
    - Q4: V+, A-  (sereno/relaxado)

    Retorna ``""`` se algum dos valores for ``NaN``.
    """
    if np.isnan(v) or np.isnan(a):
        return ""
    if v >= 0 and a >= 0:
        return "Q1"
    if v < 0 and a >= 0:
        return "Q2"
    if v < 0 and a < 0:
        return "Q3"
    return "Q4"


def _quadrants(v: np.ndarray, a: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    return np.array([quadrant_from_va(vi, ai) for vi, ai in zip(v, a)], dtype=object)


def quadrant_accuracy(
    v_true: np.ndarray, a_true: np.ndarray, v_pred: np.ndarray, a_pred: np.ndarray
) -> float:
    """Acurácia de quadrante (visão categórica grosseira, comparável ao TCC1).

    Faixas em que verdadeiro ou predito caia em quadrante vazio (``NaN``) são
    ignoradas.
    """
    qt = _quadrants(v_true, a_true)
    qp = _quadrants(v_pred, a_pred)
    mask = (qt != "") & (qp != "")
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(qt[mask] == qp[mask]))


def quadrant_f1_macro(
    v_true: np.ndarray, a_true: np.ndarray, v_pred: np.ndarray, a_pred: np.ndarray
) -> float:
    """F1-macro sobre as 4 classes de quadrante (média não ponderada).

    Classes sem suporte nem predição são omitidas da média. Faixas com quadrante
    vazio (``NaN``) são ignoradas.
    """
    qt = _quadrants(v_true, a_true)
    qp = _quadrants(v_pred, a_pred)
    mask = (qt != "") & (qp != "")
    qt, qp = qt[mask], qp[mask]
    if qt.size == 0:
        return float("nan")
    f1s = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        tp = int(np.sum((qp == q) & (qt == q)))
        fp = int(np.sum((qp == q) & (qt != q)))
        fn = int(np.sum((qp != q) & (qt == q)))
        if tp + fp + fn == 0:
            continue  # classe ausente em ambos: não entra na média
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    if not f1s:
        return float("nan")
    return float(np.mean(f1s))


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Intervalo de confiança por bootstrap percentil para qualquer estatística.

    Reamostra **linhas** de ``values`` com reposição (mantém pareamento). Para
    estatísticas pareadas como o CCC, passe ``values`` 2-D com shape ``(n, 2)``
    (colunas = as duas séries) e um ``stat_fn`` que desempacota as colunas, ex.::

        bootstrap_ci(np.column_stack([x, y]),
                     lambda m: ccc(m[:, 0], m[:, 1]))

    Args:
        values: array 1-D ``(n,)`` ou 2-D ``(n, k)``; a reamostragem é por linha.
        stat_fn: função que recebe uma reamostra (mesmo n. de colunas) e devolve
            um ``float``.
        n_boot: número de reamostragens.
        alpha: nível do IC (0.05 => IC de 95%).
        seed: semente para reprodutibilidade.

    Returns:
        ``(low, high)`` — percentis ``alpha/2`` e ``1 - alpha/2`` da distribuição
        bootstrap. Reamostras que devolvem ``nan`` são descartadas.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(values[idx])
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)
