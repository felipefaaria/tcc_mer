"""Vetor de dissociação áudio-letra — o objeto de estudo do TCC2.

A discrepância entre o que o áudio "sente" e o que a letra "diz" **não é erro de
medição**: é o fenômeno central. Uma sofrência dançante tem áudio legitimamente
alegre e letra legitimamente triste; colapsar isso num rótulo único destrói a
informação que interessa.

O vetor de dissociação é a diferença assinada, por dimensão:

    D_V = V_letra - V_audio
    D_A = A_letra - A_audio

Regras invioláveis:
- Os canais áudio e letra devem chegar **separados e calibrados individualmente**
  — nunca fundir as LFs antes de chamar esta função, ou a dissociação some.
- Faixas instrumentais (``tem_letra=não``) não têm canal de letra: o valor de
  letra entra como ``NaN`` e D sai como ``NaN`` (explícito), nunca zero.
- Só faz sentido em escala centrada [-1, 1] (ver :mod:`src.scale`).
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_dissociation"]


def compute_dissociation(
    v_audio: np.ndarray,
    v_letra: np.ndarray,
    a_audio: np.ndarray,
    a_letra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula (D_V, D_A) a partir dos canais modais **já separados**.

    Args:
        v_audio: valência estimada só do áudio, shape ``(n,)``.
        v_letra: valência estimada só da letra, shape ``(n,)`` (``NaN`` p/ instrumental).
        a_audio: arousal estimado só do áudio, shape ``(n,)``.
        a_letra: arousal estimado só da letra, shape ``(n,)`` (``NaN`` p/ instrumental).

    Returns:
        ``(D_V, D_A)`` — diferenças assinadas letra-menos-áudio, mesmo shape da
        entrada. ``NaN`` propaga naturalmente para faixas sem letra.

    Raises:
        ValueError: se os quatro vetores não tiverem shapes idênticos.
    """
    v_audio = np.asarray(v_audio, dtype=float)
    v_letra = np.asarray(v_letra, dtype=float)
    a_audio = np.asarray(a_audio, dtype=float)
    a_letra = np.asarray(a_letra, dtype=float)

    shapes = {v_audio.shape, v_letra.shape, a_audio.shape, a_letra.shape}
    if len(shapes) != 1:
        raise ValueError(f"shapes divergentes entre os canais: {shapes}")

    d_v = v_letra - v_audio
    d_a = a_letra - a_audio
    return d_v, d_a
