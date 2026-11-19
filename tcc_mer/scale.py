"""Conversão de escala — única fonte de verdade do projeto.

Toda a literatura mistura escalas (DEAM/Music2Emo reportam 1-9; SAM costuma ser
1-9; o TCC1 usou 0-10). A convenção interna é padronizar **tudo** para [-1, 1]
com 0 no centro afetivo, porque diferenças assinadas (D_V, D_A) só fazem sentido
numa escala centrada.

Regra do projeto: todas as LFs e a agregação de anotações devem chamar
``to_unit`` na borda de entrada/saída de cada ferramenta — nunca reimplementar a
conversão inline.
"""
from __future__ import annotations

import numpy as np

__all__ = ["to_unit"]


def to_unit(x, lo: float, hi: float):
    """Converte ``x`` de ``[lo, hi]`` para ``[-1, 1]`` (linear, 0 no centro).

    Aceita escalar ou ``np.ndarray`` (mantém o shape e propaga ``NaN``).

    Args:
        x: valor(es) na escala original.
        lo: limite inferior da escala original (mapeia para -1).
        hi: limite superior da escala original (mapeia para +1).

    Returns:
        Valor(es) em ``[-1, 1]``. Escalar de entrada devolve ``float``.

    Raises:
        ValueError: se ``hi == lo`` (escala degenerada).

    Exemplos:
        >>> to_unit(1, 1, 9)   # borda inferior SAM 1-9
        -1.0
        >>> to_unit(9, 1, 9)   # borda superior
        1.0
        >>> to_unit(5, 1, 9)   # meio -> centro afetivo
        0.0
    """
    if hi == lo:
        raise ValueError(f"escala degenerada: hi == lo == {lo}")
    arr = np.asarray(x, dtype=float)
    out = 2.0 * (arr - lo) / (hi - lo) - 1.0
    # Preserva semântica escalar: entrada 0-d devolve float puro.
    if out.ndim == 0:
        return float(out)
    return out
