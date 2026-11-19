"""Testes de tcc_mer/dissociation.py — D_V, D_A a partir de canais separados."""

import numpy as np
import pytest

from tcc_mer.dissociation import compute_dissociation


def test_formula_basica():
    """D = letra - audio, por dimensão."""
    v_audio = np.array([0.5, -0.2])
    v_lyrics = np.array([-0.5, -0.2])
    a_audio = np.array([0.3, 0.1])
    a_lyrics = np.array([0.1, 0.4])
    d_v, d_a = compute_dissociation(v_audio, v_lyrics, a_audio, a_lyrics)
    np.testing.assert_allclose(d_v, [-1.0, 0.0])
    np.testing.assert_allclose(d_a, [-0.2, 0.3])


def test_sofrencia_dancante():
    """Sofrência: áudio alegre (V+), letra triste (V-) => D_V fortemente negativo."""
    v_audio = np.array([0.7])
    v_lyrics = np.array([-0.6])  # letra triste
    a_audio = np.array([0.6])
    a_lyrics = np.array([0.4])
    d_v, d_a = compute_dissociation(v_audio, v_lyrics, a_audio, a_lyrics)
    assert d_v[0] < -1.0  # letra bem mais triste que o áudio
    assert d_a[0] < 0
    assert abs(d_a[0]) < abs(d_v[0])


def test_instrumental_nan_propaga():
    """faixa instrumental: canal de letra é NaN => D é NaN (nunca zero)."""
    v_audio = np.array([0.5, 0.2])
    v_lyrics = np.array([-0.3, np.nan])  # 2ª faixa é instrumental
    a_audio = np.array([0.4, 0.1])
    a_lyrics = np.array([0.2, np.nan])
    d_v, d_a = compute_dissociation(v_audio, v_lyrics, a_audio, a_lyrics)
    assert not np.isnan(d_v[0])
    assert np.isnan(d_v[1])
    assert np.isnan(d_a[1])
    assert d_v[1] != 0.0 or np.isnan(d_v[1])  # explicitamente não é zero


def test_shapes_divergentes_levanta():
    with pytest.raises(ValueError):
        compute_dissociation(
            np.array([0.1, 0.2]),
            np.array([0.1]),
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
        )


def test_concordancia_total_da_zero():
    """Áudio e letra concordam => dissociação nula (MPB introspectiva)."""
    v = np.array([-0.5, -0.4])
    a = np.array([-0.3, -0.2])
    d_v, d_a = compute_dissociation(v, v.copy(), a, a.copy())
    np.testing.assert_allclose(d_v, [0.0, 0.0])
    np.testing.assert_allclose(d_a, [0.0, 0.0])
