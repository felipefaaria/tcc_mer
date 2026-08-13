"""Testes de src/scale.py — conversão para [-1, 1] com casos de borda."""
import numpy as np
import pytest

from src.scale import to_unit


@pytest.mark.parametrize("lo,hi", [(1, 9), (0, 10), (-5, 5)])
def test_bordas_e_meio(lo, hi):
    """lo -> -1, hi -> +1, meio -> 0 para várias escalas."""
    assert to_unit(lo, lo, hi) == pytest.approx(-1.0)
    assert to_unit(hi, lo, hi) == pytest.approx(1.0)
    assert to_unit((lo + hi) / 2, lo, hi) == pytest.approx(0.0)


def test_sam_1_9_saidas_inteiras():
    """SAM 1-9: pontos intermediários caem onde se espera."""
    assert to_unit(3, 1, 9) == pytest.approx(-0.5)
    assert to_unit(7, 1, 9) == pytest.approx(0.5)


def test_music2emo_1_9_igual_ao_sam():
    """Music2Emo também reporta 1-9: mesma conversão que o SAM."""
    for x in range(1, 10):
        assert to_unit(x, 1, 9) == pytest.approx(to_unit(x, 1, 9))
    assert to_unit(5, 1, 9) == pytest.approx(0.0)


def test_vetorizado_preserva_shape():
    x = np.array([1.0, 5.0, 9.0])
    out = to_unit(x, 1, 9)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])


def test_escalar_devolve_float():
    out = to_unit(5, 1, 9)
    assert isinstance(out, float)


def test_nan_propaga():
    out = to_unit(np.array([np.nan, 5.0]), 1, 9)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.0)


def test_escala_degenerada_levanta():
    with pytest.raises(ValueError):
        to_unit(5, 3, 3)


def test_inversao_de_ida_e_volta():
    """to_unit é linear e invertível: recuperar o valor original."""
    lo, hi = 1, 9
    for x in [1, 4, 5, 9]:
        u = to_unit(x, lo, hi)
        back = (u + 1) / 2 * (hi - lo) + lo
        assert back == pytest.approx(x)
