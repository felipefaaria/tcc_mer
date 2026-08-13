"""Concordância entre anotadores — alfa de Krippendorff (nível intervalar).

Quando vários humanos anotam a mesma faixa, precisamos saber se concordam —
senão o gold não é ouro. O alfa de Krippendorff corrige o acaso, funciona com
dados contínuos, múltiplos anotadores e valores faltantes (ideal para o desenho
de sobreposição parcial da coleta).

Calcular **separadamente** para Valência e Arousal (chamadas distintas) — nunca
combinar as dimensões. Espere alfa menor em Valência que em Arousal: isso é
*resultado*, não fracasso.

Implementação pela matriz de coincidências, com a função de diferença intervalar
δ²(c,k) = (c-k)²:

    Do = (1/n) · Σ_u [ 1/(m_u-1) · Σ_{i≠j} (x_ui - x_uj)² ]
    De = (1/(n·(n-1))) · Σ_{i,j∈pool} (P_i - P_j)²
    α  = 1 - Do/De

onde n = Σ_u m_u sobre unidades com ≥2 valores válidos, e ``pool`` é a lista de
todos os valores emparelháveis.
"""
from __future__ import annotations

import numpy as np

__all__ = ["krippendorff_alpha", "alpha_leave_one_out"]


def _pair_sum_sq(vals: np.ndarray) -> float:
    """Σ_{i≠j} (x_i - x_j)² = 2·(m·Σx² - (Σx)²) para um vetor ``vals``."""
    m = vals.size
    s = vals.sum()
    ss = (vals * vals).sum()
    return 2.0 * (m * ss - s * s)


def krippendorff_alpha(ratings: np.ndarray) -> float:
    """Alfa de Krippendorff intervalar para uma matriz de avaliações.

    Args:
        ratings: matriz ``(n_faixas, n_anotadores)`` com ``NaN`` para ausência.
            Cada linha é uma faixa (unidade); cada coluna, um anotador.

    Returns:
        Alfa em (-∞, 1]. ``1.0`` = concordância perfeita; ``0`` = nível do acaso.
        Retorna ``nan`` se não houver unidades emparelháveis (≥2 valores) ou se o
        desacordo esperado for zero (todos os valores idênticos).

    Notas:
        Chame uma vez para Valência e outra para Arousal — nunca misture as
        dimensões numa mesma matriz.
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ratings deve ser 2-D (n_faixas, n_anotadores)")

    do_num = 0.0
    n = 0
    pool: list[np.ndarray] = []
    for row in ratings:
        vals = row[~np.isnan(row)]
        m = vals.size
        if m >= 2:
            do_num += _pair_sum_sq(vals) / (m - 1)
            n += m
            pool.append(vals)

    if n < 2:
        return float("nan")

    do = do_num / n
    pooled = np.concatenate(pool)
    de_num = _pair_sum_sq(pooled)  # Σ_{i,j} (P_i - P_j)²
    de = de_num / (n * (n - 1))
    if de == 0:
        return float("nan")
    return float(1.0 - do / de)


def alpha_leave_one_out(
    ratings: np.ndarray, annotator_ids: list[str]
) -> dict[str, float]:
    """Recalcula o alfa excluindo cada anotador (uma coluna) por vez.

    Um salto grande no alfa ao remover alguém sinaliza um outlier a investigar —
    é uma das camadas do QC de anotadores. Nunca excluir por discordância pura;
    este é apenas um sinal de diagnóstico.

    Args:
        ratings: matriz ``(n_faixas, n_anotadores)`` com ``NaN`` para ausência.
        annotator_ids: rótulos das colunas, ``len == n_anotadores``.

    Returns:
        ``{pid: alfa_sem_esse_pid}`` para cada anotador.
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ratings deve ser 2-D (n_faixas, n_anotadores)")
    if len(annotator_ids) != ratings.shape[1]:
        raise ValueError(
            f"annotator_ids ({len(annotator_ids)}) != n colunas ({ratings.shape[1]})"
        )
    out: dict[str, float] = {}
    for j, pid in enumerate(annotator_ids):
        reduced = np.delete(ratings, j, axis=1)
        out[pid] = krippendorff_alpha(reduced)
    return out
