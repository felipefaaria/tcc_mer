"""Concordância entre anotadores: alfa de Krippendorff intervalar.

O alfa corrige o acaso e aceita dados contínuos, vários anotadores e valores
faltantes (o desenho do gold set tem sobreposição parcial). Calcule uma vez por
dimensão: valência e arousal nunca entram na mesma matriz.

Implementação pela matriz de coincidências, com diferença intervalar
``δ²(c, k) = (c - k)²``::

    Do = (1/n) · Σ_u [ 1/(m_u - 1) · Σ_{i≠j} (x_ui - x_uj)² ]
    De = (1/(n·(n - 1))) · Σ_{i,j ∈ pool} (P_i - P_j)²
    α  = 1 - Do/De

em que ``n = Σ_u m_u`` sobre as unidades com pelo menos dois valores e ``pool`` é
a lista de todos os valores emparelháveis.
"""

import numpy as np

__all__ = ["krippendorff_alpha", "alpha_leave_one_out"]


def _pair_sum_sq(values: np.ndarray) -> float:
    """``Σ_{i≠j} (x_i - x_j)² = 2·(m·Σx² - (Σx)²)``."""
    m = values.size
    return 2.0 * (m * (values * values).sum() - values.sum() ** 2)


def krippendorff_alpha(ratings: np.ndarray) -> float:
    """Alfa de Krippendorff intervalar.

    Args:
        ratings: matriz ``(n_faixas, n_anotadores)`` com ``NaN`` para ausência.

    Returns:
        Alfa em (-∞, 1]: 1 é concordância perfeita e 0, o nível do acaso. Retorna
        ``nan`` sem unidades emparelháveis ou quando o desacordo esperado é zero.
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ratings deve ser 2-D (n_faixas, n_anotadores)")

    observed_numerator = 0.0
    n = 0
    pool: list[np.ndarray] = []
    for row in ratings:
        values = row[~np.isnan(row)]
        if values.size >= 2:
            observed_numerator += _pair_sum_sq(values) / (values.size - 1)
            n += values.size
            pool.append(values)

    if n < 2:
        return float("nan")

    observed = observed_numerator / n
    expected = _pair_sum_sq(np.concatenate(pool)) / (n * (n - 1))
    if expected == 0:
        return float("nan")
    return float(1.0 - observed / expected)


def alpha_leave_one_out(ratings: np.ndarray, annotator_ids: list[str]) -> dict[str, float]:
    """Alfa recalculado sem cada anotador (uma coluna por vez).

    Um salto grande ao remover alguém sinaliza um anotador a investigar. Serve de
    diagnóstico, nunca de critério de exclusão por discordância.

    Args:
        ratings: matriz ``(n_faixas, n_anotadores)`` com ``NaN`` para ausência.
        annotator_ids: rótulos das colunas.

    Returns:
        ``{id: alfa_sem_esse_anotador}``.
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ratings deve ser 2-D (n_faixas, n_anotadores)")
    if len(annotator_ids) != ratings.shape[1]:
        raise ValueError(f"annotator_ids ({len(annotator_ids)}) != n colunas ({ratings.shape[1]})")
    return {
        annotator: krippendorff_alpha(np.delete(ratings, j, axis=1))
        for j, annotator in enumerate(annotator_ids)
    }
