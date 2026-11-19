"""Testes de tcc_mer/fusion.py: fusão simples e fusão por dimensão."""

import numpy as np
import pytest

from tcc_mer.fusion import dimension_fusion, simple_fusion

V_AUDIO = np.array([0.8, -0.2, 0.4])
A_AUDIO = np.array([0.6, -0.4, 0.2])
V_LYRICS = np.array([-0.6, -0.8, 0.4])
A_LYRICS = np.array([0.0, 0.2, -0.2])


def test_fusao_simples_e_a_media_por_dimensao():
    v, a = simple_fusion(V_AUDIO, A_AUDIO, V_LYRICS, A_LYRICS)
    assert v == pytest.approx([0.1, -0.5, 0.4])
    assert a == pytest.approx([0.3, -0.1, 0.0])


def test_fusao_por_dimensao_usa_valencia_da_letra_e_arousal_do_audio():
    v, a = dimension_fusion(V_AUDIO, A_AUDIO, V_LYRICS, A_LYRICS)
    assert v == pytest.approx(V_LYRICS)
    assert a == pytest.approx(A_AUDIO)


@pytest.mark.parametrize("fusion", [simple_fusion, dimension_fusion])
def test_faixa_sem_letra_fica_indefinida_nas_duas_dimensoes(fusion):
    v_lyrics = np.array([-0.6, np.nan])
    a_lyrics = np.array([0.0, np.nan])
    v, a = fusion(V_AUDIO[:2], A_AUDIO[:2], v_lyrics, a_lyrics)
    assert not np.isnan(v[0]) and not np.isnan(a[0])
    assert np.isnan(v[1]) and np.isnan(a[1])


@pytest.mark.parametrize("fusion", [simple_fusion, dimension_fusion])
def test_audio_ausente_tambem_deixa_a_fusao_indefinida(fusion):
    v, a = fusion(np.array([np.nan]), np.array([0.2]), np.array([0.1]), np.array([0.3]))
    assert np.isnan(v[0]) and np.isnan(a[0])


@pytest.mark.parametrize("fusion", [simple_fusion, dimension_fusion])
def test_shapes_divergentes_levantam_erro(fusion):
    with pytest.raises(ValueError, match="shapes divergentes"):
        fusion(V_AUDIO, A_AUDIO[:2], V_LYRICS, A_LYRICS)
