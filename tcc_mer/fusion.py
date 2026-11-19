"""Fusão tardia dos rótulos de áudio (LF1) e de letra (LF2), sem parâmetros ajustados.

As duas fusões operam sobre os valores brutos, em [-1, 1]:

* **simples**: média dos dois canais em cada dimensão (é o rótulo fundido do teste
  pré-especificado com ouvintes);
* **por dimensão**: valência da letra e arousal do áudio (análise exploratória,
  motivada pela assimetria entre os canais).

A fusão só existe quando os dois canais existem. Numa faixa instrumental, ou em que
algum canal falhou, as duas saídas são ``NaN``.

Uma fusão ajustada no gold set (calibração afim e pesos por CCC) foi explorada e
abandonada. O código está em :mod:`tcc_mer.exploratory`.
"""

import numpy as np

__all__ = ["simple_fusion", "dimension_fusion"]


def _as_channels(v_audio, a_audio, v_lyrics, a_lyrics) -> tuple[np.ndarray, ...]:
    channels = tuple(np.asarray(x, dtype=float) for x in (v_audio, a_audio, v_lyrics, a_lyrics))
    if len({c.shape for c in channels}) != 1:
        raise ValueError(f"shapes divergentes entre os canais: {[c.shape for c in channels]}")
    return channels


def _undefined_where_a_channel_is_missing(channels, *outputs) -> tuple[np.ndarray, ...]:
    missing = np.any([np.isnan(c) for c in channels], axis=0)
    return tuple(np.where(missing, np.nan, out) for out in outputs)


def simple_fusion(v_audio, a_audio, v_lyrics, a_lyrics) -> tuple[np.ndarray, np.ndarray]:
    """Média simples dos canais: ``y = (y_áudio + y_letra) / 2`` em cada dimensão.

    Returns:
        ``(valência, arousal)`` fundidos.
    """
    channels = _as_channels(v_audio, a_audio, v_lyrics, a_lyrics)
    v_audio, a_audio, v_lyrics, a_lyrics = channels
    return _undefined_where_a_channel_is_missing(
        channels, (v_audio + v_lyrics) / 2, (a_audio + a_lyrics) / 2
    )


def dimension_fusion(v_audio, a_audio, v_lyrics, a_lyrics) -> tuple[np.ndarray, np.ndarray]:
    """Regra por dimensão: valência vem da letra e arousal vem do áudio.

    Returns:
        ``(valência, arousal)`` fundidos.
    """
    channels = _as_channels(v_audio, a_audio, v_lyrics, a_lyrics)
    return _undefined_where_a_channel_is_missing(channels, channels[2], channels[1])
