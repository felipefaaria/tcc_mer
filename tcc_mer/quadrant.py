"""Quadrantes do plano valência-arousal de Russell e o rótulo de emoção de cada um.

======  =========  =======  ================
Código  Valência   Arousal  Emoção
======  =========  =======  ================
Q1      >= 0       >= 0     alegria
Q2      < 0        >= 0     tensão
Q3      < 0        < 0      tristeza
Q4      >= 0       < 0      calma
======  =========  =======  ================

O zero pertence ao lado positivo. Um par com ``NaN`` não tem quadrante (``""``).
"""

import numpy as np

__all__ = ["QUADRANTS", "EMOTION_BY_QUADRANT", "quadrants", "quadrant_of", "emotion_of", "margin"]

QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
EMOTION_BY_QUADRANT = {"Q1": "alegria", "Q2": "tensão", "Q3": "tristeza", "Q4": "calma"}


def quadrants(valence, arousal) -> np.ndarray:
    """Quadrante de cada par ``(valência, arousal)`` em [-1, 1], como array de ``str``.

    Pares com ``NaN`` recebem ``""``.
    """
    v = np.asarray(valence, dtype=float)
    a = np.asarray(arousal, dtype=float)
    return np.select(
        [(v >= 0) & (a >= 0), (v < 0) & (a >= 0), (v < 0) & (a < 0), (v >= 0) & (a < 0)],
        QUADRANTS,
        default="",
    )


def quadrant_of(valence: float, arousal: float) -> str:
    """Quadrante de um único par ``(valência, arousal)``; ``""`` se houver ``NaN``."""
    return str(quadrants(valence, arousal))


def emotion_of(quadrant: str) -> str:
    """Nome da emoção de um quadrante; ``""`` para quadrante vazio."""
    return EMOTION_BY_QUADRANT.get(quadrant, "")


def margin(valence, arousal) -> np.ndarray:
    """Distância do ponto ao eixo mais próximo, ``min(|V|, |A|)``.

    Valores pequenos indicam um ponto perto da fronteira entre quadrantes, onde o
    rótulo muda com pouca variação da estimativa.
    """
    return np.minimum(
        np.abs(np.asarray(valence, dtype=float)), np.abs(np.asarray(arousal, dtype=float))
    )
