"""Testes de tcc_mer/exploratory/pipeline.py — calibração+fusão com k-fold aninhado."""

import numpy as np
import pytest

from tcc_mer.exploratory.pipeline import apply_channel, fit_channel, nested_kfold_ccc
from tcc_mer.metrics import ccc


def test_single_lf_calibracao_recupera_gold():
    """1 LF afim-relacionada ao gold -> CCC OOS alto; corrige escala/viés."""
    rng = np.random.default_rng(0)
    gold = rng.uniform(-1, 1, 60)
    lf = (gold + 1) / 2 * 8 + 1 + rng.normal(0, 0.1, 60)  # escala 1-9 enviesada
    X = lf[:, None]
    c, w, n = nested_kfold_ccc(X, gold, k=5, seed=0)
    assert n == 60
    assert w.shape == (1,)
    assert c > 0.9


def test_fusao_supera_lf_ruidosa():
    """2 LFs (uma boa, uma ruído): peso vai para a boa; CCC OOS alto."""
    rng = np.random.default_rng(1)
    gold = rng.uniform(-1, 1, 80)
    boa = gold + rng.normal(0, 0.1, 80)
    ruim = rng.uniform(-1, 1, 80)
    X = np.column_stack([boa, ruim])
    c, w, n = nested_kfold_ccc(X, gold, k=5, seed=0)
    assert c > 0.8
    assert w[0] > w[1]


def test_fit_apply_channel_recupera_afim():
    gold = np.linspace(-1, 1, 30)
    lf = 3.0 * gold - 0.4
    cals, w = fit_channel(lf[:, None], gold)
    est = apply_channel(cals, w, lf[:, None])
    np.testing.assert_allclose(est, gold, atol=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_descarta_nan():
    gold = np.array([-1.0, 0.0, 1.0, np.nan, 0.5, -0.5, 0.2, -0.2])
    lf = np.array([1.0, 5.0, 9.0, 5.0, np.nan, 3.0, 6.0, 4.0])
    c, w, n = nested_kfold_ccc(lf[:, None], gold, k=3, seed=0)
    assert n == 6  # duas linhas com NaN removidas


def test_oos_nao_maior_que_in_sample():
    """CCC OOS não deve superar o ajuste no todo (evita autoengano)."""
    rng = np.random.default_rng(2)
    gold = rng.uniform(-1, 1, 50)
    lf = gold + rng.normal(0, 0.3, 50)
    X = lf[:, None]
    c_oos, _, _ = nested_kfold_ccc(X, gold, k=5, seed=0)
    cals, w = fit_channel(X, gold)
    c_in = ccc(apply_channel(cals, w, X), gold)
    assert c_oos <= c_in + 1e-6


def test_poucos_dados_retorna_nan():
    c, w, n = nested_kfold_ccc(np.array([[0.5]]), np.array([0.3]), k=5)
    assert np.isnan(c)
