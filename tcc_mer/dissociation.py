"""Dissociação áudio-letra, o objeto de estudo do trabalho.

A distância entre o que o áudio expressa e o que a letra diz não é erro de
medição. Uma sofrência dançante tem áudio alegre e letra triste, e colapsar os dois
num rótulo único apaga justamente isso. A dissociação é a diferença assinada, por
dimensão::

    D_V = V_letra - V_áudio
    D_A = A_letra - A_áudio

Os canais precisam chegar separados, nunca fundidos. Faixas instrumentais não têm
canal de letra: o valor entra como ``NaN`` e ``D`` sai ``NaN``, nunca zero.
"""

import numpy as np

__all__ = ["compute_dissociation", "magnitude"]


def compute_dissociation(
    v_audio: np.ndarray,
    v_lyrics: np.ndarray,
    a_audio: np.ndarray,
    a_lyrics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula ``(D_V, D_A)`` a partir dos canais já separados, em [-1, 1].

    ``NaN`` propaga para as faixas sem letra.

    Raises:
        ValueError: se os quatro vetores não tiverem o mesmo shape.
    """
    v_audio, v_lyrics, a_audio, a_lyrics = (
        np.asarray(x, dtype=float) for x in (v_audio, v_lyrics, a_audio, a_lyrics)
    )
    shapes = {v_audio.shape, v_lyrics.shape, a_audio.shape, a_lyrics.shape}
    if len(shapes) != 1:
        raise ValueError(f"shapes divergentes entre os canais: {shapes}")
    return v_lyrics - v_audio, a_lyrics - a_audio


def magnitude(d_v: np.ndarray, d_a: np.ndarray) -> np.ndarray:
    """Norma euclidiana ``‖D‖ = sqrt(D_V² + D_A²)``, em ``[0, 2·sqrt(2)]``."""
    return np.hypot(np.asarray(d_v, dtype=float), np.asarray(d_a, dtype=float))
