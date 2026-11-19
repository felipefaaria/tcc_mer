"""Testes de tcc_mer/exploratory/weighted_fusion.py: pesos no simplex e k-fold fora da amostra."""

import numpy as np
import pytest

from tcc_mer.exploratory.weighted_fusion import fit_fusion_weights, fuse, kfold_out_of_sample_ccc
from tcc_mer.metrics import ccc


def test_pesos_no_simplex():
    """Pesos não-negativos e somando 1."""
    rng = np.random.default_rng(0)
    gold = rng.uniform(-1, 1, size=40)
    lf_bom = gold + rng.normal(0, 0.05, size=40)
    lf_ruim = rng.uniform(-1, 1, size=40)
    calibrated = np.column_stack([lf_bom, lf_ruim])
    w = fit_fusion_weights(calibrated, gold)
    assert w.shape == (2,)
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0)


def test_peso_maior_para_lf_melhor():
    """A LF que concorda com o gold deve receber mais peso que a ruidosa."""
    rng = np.random.default_rng(1)
    gold = rng.uniform(-1, 1, size=60)
    lf_bom = gold + rng.normal(0, 0.03, size=60)
    lf_ruim = rng.uniform(-1, 1, size=60)
    calibrated = np.column_stack([lf_bom, lf_ruim])
    w = fit_fusion_weights(calibrated, gold)
    assert w[0] > w[1]
    assert w[0] > 0.8


def test_fusao_nao_piora_ccc_no_treino():
    """CCC da fusão >= CCC da melhor LF isolada (no conjunto de ajuste)."""
    rng = np.random.default_rng(2)
    gold = rng.uniform(-1, 1, size=80)
    lf1 = gold + rng.normal(0, 0.2, size=80)
    lf2 = gold + rng.normal(0, 0.3, size=80)
    calibrated = np.column_stack([lf1, lf2])
    w = fit_fusion_weights(calibrated, gold)
    fused = fuse(calibrated, w)
    melhor_isolada = max(ccc(lf1, gold), ccc(lf2, gold))
    assert ccc(fused, gold) >= melhor_isolada - 1e-6


def test_fuse_shape():
    calibrated = np.ones((5, 3))
    w = np.array([0.2, 0.3, 0.5])
    out = fuse(calibrated, w)
    assert out.shape == (5,)
    np.testing.assert_allclose(out, 1.0)


def test_kfold_fora_da_amostra_menor_que_no_treino():
    """CCC fora-da-amostra deve ser <= CCC ajustado no todo (evita inflar)."""
    rng = np.random.default_rng(3)
    gold = rng.uniform(-1, 1, size=60)
    lf1 = gold + rng.normal(0, 0.25, size=60)
    lf2 = gold + rng.normal(0, 0.25, size=60)
    calibrated = np.column_stack([lf1, lf2])

    w_full = fit_fusion_weights(calibrated, gold)
    ccc_treino = ccc(fuse(calibrated, w_full), gold)
    ccc_oos, mean_w = kfold_out_of_sample_ccc(calibrated, gold, k=5, seed=0)

    assert -1.0 <= ccc_oos <= 1.0
    assert ccc_oos <= ccc_treino + 1e-6
    assert mean_w.shape == (2,)
    assert mean_w.sum() == pytest.approx(1.0, abs=1e-6)


def test_kfold_reprodutivel():
    rng = np.random.default_rng(4)
    gold = rng.uniform(-1, 1, size=40)
    calibrated = np.column_stack([gold + 0.1, gold - 0.1])
    a = kfold_out_of_sample_ccc(calibrated, gold, k=4, seed=42)
    b = kfold_out_of_sample_ccc(calibrated, gold, k=4, seed=42)
    assert a[0] == pytest.approx(b[0])


def test_kfold_exige_min_2():
    calibrated = np.array([[0.1, 0.2]])
    gold = np.array([0.15])
    with pytest.raises(ValueError):
        kfold_out_of_sample_ccc(calibrated, gold, k=5)
