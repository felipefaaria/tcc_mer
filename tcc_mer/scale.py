"""Conversão de escala para o intervalo interno [-1, 1].

DEAM, Music2Emo e o SAM usam 1-9. Diferenças assinadas como ``D_V`` e ``D_A`` só
fazem sentido numa escala centrada em zero, então toda saída de modelo e toda
anotação humana passa por :func:`to_unit` na borda, nunca inline.
"""

import numpy as np

__all__ = ["to_unit"]


def to_unit(x, lo: float, hi: float):
    """Converte ``x`` de ``[lo, hi]`` para ``[-1, 1]`` (linear, ponto médio em 0).

    Aceita escalar ou ``np.ndarray`` (mantém o shape e propaga ``NaN``).

    Raises:
        ValueError: se ``hi == lo``.

    Exemplos:
        >>> to_unit(1, 1, 9)
        -1.0
        >>> to_unit(5, 1, 9)
        0.0
        >>> to_unit(9, 1, 9)
        1.0
    """
    if hi == lo:
        raise ValueError(f"escala degenerada: hi == lo == {lo}")
    out = 2.0 * (np.asarray(x, dtype=float) - lo) / (hi - lo) - 1.0
    return float(out) if out.ndim == 0 else out
