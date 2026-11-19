"""Testes de tcc_mer/exploratory/calibration.py — correção afim ajustada no gold."""

import numpy as np
import pytest

from tcc_mer.exploratory.calibration import AffineCalibration


def test_recupera_relacao_linear_exata():
    """Se gold = 2*lf - 0.5, o fit deve recuperar a=2, b=-0.5."""
    lf = np.linspace(-1, 1, 20)
    gold = 2.0 * lf - 0.5
    cal = AffineCalibration.fit(lf, gold)
    assert cal.a == pytest.approx(2.0)
    assert cal.b == pytest.approx(-0.5)
    np.testing.assert_allclose(cal.apply(lf), gold, atol=1e-9)


def test_corrige_vies_de_deslocamento():
    """LF com viés constante (arousal-mais-baixo): b captura o deslocamento."""
    rng = np.random.default_rng(0)
    gold = rng.uniform(-1, 1, size=50)
    lf = gold - 0.4  # LF subestima em 0.4
    cal = AffineCalibration.fit(lf, gold)
    assert cal.a == pytest.approx(1.0, abs=1e-6)
    assert cal.b == pytest.approx(0.4, abs=1e-6)


def test_ignora_nan_no_ajuste():
    lf = np.array([-1.0, 0.0, np.nan, 1.0, 0.5])
    gold = np.array([-0.5, 0.5, 0.2, 1.5, np.nan])
    cal = AffineCalibration.fit(lf, gold)
    # equivalente a ajustar só nos pares válidos [-1,0,1] -> [-0.5,0.5,1.5]
    cal_ref = AffineCalibration.fit(np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5, 1.5]))
    assert cal.a == pytest.approx(cal_ref.a)
    assert cal.b == pytest.approx(cal_ref.b)


def test_apply_propaga_nan():
    cal = AffineCalibration(a=1.5, b=0.2)
    out = cal.apply(np.array([0.0, np.nan]))
    assert out[0] == pytest.approx(0.2)
    assert np.isnan(out[1])


def test_poucos_pares_levanta():
    with pytest.raises(ValueError):
        AffineCalibration.fit(np.array([0.5]), np.array([0.5]))


def test_shapes_incompativeis_levanta():
    with pytest.raises(ValueError):
        AffineCalibration.fit(np.array([0.1, 0.2]), np.array([0.1]))
