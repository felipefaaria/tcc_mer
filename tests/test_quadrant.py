"""Testes de tcc_mer/quadrant.py: quadrantes, emoção e margem."""

import numpy as np
import pytest

from tcc_mer.quadrant import EMOTION_BY_QUADRANT, emotion_of, margin, quadrant_of, quadrants


@pytest.mark.parametrize(
    ("valence", "arousal", "expected"),
    [(0.5, 0.5, "Q1"), (-0.5, 0.5, "Q2"), (-0.5, -0.5, "Q3"), (0.5, -0.5, "Q4")],
)
def test_quadrante_de_cada_canto(valence, arousal, expected):
    assert quadrant_of(valence, arousal) == expected


def test_zero_conta_como_positivo():
    assert quadrant_of(0.0, 0.0) == "Q1"
    assert quadrant_of(0.0, -0.1) == "Q4"
    assert quadrant_of(-0.1, 0.0) == "Q2"


def test_nan_nao_tem_quadrante():
    assert quadrant_of(np.nan, 0.5) == ""
    assert quadrant_of(0.5, np.nan) == ""


def test_quadrants_vetorizado_mantem_ordem_e_nan():
    result = quadrants([0.5, np.nan, -0.5], [0.5, 0.5, -0.5])
    assert list(result) == ["Q1", "", "Q3"]


def test_cada_quadrante_tem_uma_emocao():
    assert set(EMOTION_BY_QUADRANT) == {"Q1", "Q2", "Q3", "Q4"}
    assert emotion_of("Q1") == "alegria"
    assert emotion_of("Q2") == "tensão"
    assert emotion_of("Q3") == "tristeza"
    assert emotion_of("Q4") == "calma"


def test_emocao_de_quadrante_vazio_e_vazia():
    assert emotion_of("") == ""


def test_margem_e_a_distancia_ao_eixo_mais_proximo():
    result = margin([0.5, -0.02, np.nan], [-0.1, 0.9, 0.3])
    assert result[0] == pytest.approx(0.1)
    assert result[1] == pytest.approx(0.02)
    assert np.isnan(result[2])
