"""Testes de src/metrics.py — CCC, MAE/RMSE, quadrantes, bootstrap.

Inclui o teste que motiva a escolha do CCC: sob viés de deslocamento, a Pearson
continua alta enquanto o CCC cai (o CCC pune o viés que a Pearson ignora).
"""
import numpy as np
import pytest

from src.metrics import (
    bootstrap_ci,
    ccc,
    mae,
    quadrant_accuracy,
    quadrant_f1_macro,
    quadrant_from_va,
    rmse,
)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def test_ccc_igual_1_quando_identico():
    x = np.array([-1.0, -0.3, 0.0, 0.5, 1.0])
    assert ccc(x, x) == pytest.approx(1.0)


def test_ccc_menor_que_pearson_sob_vies():
    """CCC ≠ Pearson: com deslocamento de média, Pearson=1 mas CCC<1.

    Este é o caso que justifica usar CCC em vez de Pearson (PP1). y = x + 0.3:
    forma perfeitamente correlacionada, mas sistematicamente deslocada.
    """
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = x + 0.3  # viés de deslocamento puro
    r = pearson(x, y)
    c = ccc(x, y)
    assert r == pytest.approx(1.0)  # Pearson cega ao viés
    assert c < r  # CCC pune o deslocamento
    assert c < 0.98


def test_ccc_valor_conhecido():
    """Valor fechado: y = x + b => CCC = 2σ²/(2σ² + b²)."""
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    b = 0.3
    y = x + b
    var = x.var()  # populacional
    esperado = 2 * var / (2 * var + b ** 2)
    assert ccc(x, y) == pytest.approx(esperado)


def test_ccc_simetrico():
    x = np.array([-1.0, 0.2, 0.7, -0.4])
    y = np.array([-0.8, 0.1, 0.5, 0.0])
    assert ccc(x, y) == pytest.approx(ccc(y, x))


def test_ccc_ignora_nan_pareado():
    x = np.array([-1.0, np.nan, 0.5, 1.0])
    y = np.array([-1.0, 0.3, 0.5, 1.0])
    # deve equivaler a remover o par com NaN
    assert ccc(x, y) == pytest.approx(ccc(np.array([-1.0, 0.5, 1.0]),
                                          np.array([-1.0, 0.5, 1.0])))


def test_ccc_poucos_pares_retorna_nan():
    assert np.isnan(ccc(np.array([0.5]), np.array([0.5])))


def test_mae_rmse_basico():
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([1.0, -1.0, 1.0])
    assert mae(x, y) == pytest.approx(1.0)
    assert rmse(x, y) == pytest.approx(1.0)


def test_rmse_penaliza_outlier_mais_que_mae():
    x = np.zeros(4)
    y = np.array([0.0, 0.0, 0.0, 2.0])
    assert rmse(x, y) > mae(x, y)


def test_quadrantes():
    assert quadrant_from_va(0.5, 0.5) == "Q1"
    assert quadrant_from_va(-0.5, 0.5) == "Q2"
    assert quadrant_from_va(-0.5, -0.5) == "Q3"
    assert quadrant_from_va(0.5, -0.5) == "Q4"


def test_quadrante_bordas_zero_positivo():
    """Convenção: 0 conta como positivo."""
    assert quadrant_from_va(0.0, 0.0) == "Q1"
    assert quadrant_from_va(0.0, -0.1) == "Q4"


def test_quadrante_nan_vazio():
    assert quadrant_from_va(np.nan, 0.5) == ""


def test_quadrant_accuracy_perfeita():
    v = np.array([0.5, -0.5, -0.5, 0.5])
    a = np.array([0.5, 0.5, -0.5, -0.5])
    assert quadrant_accuracy(v, a, v, a) == pytest.approx(1.0)


def test_quadrant_accuracy_metade():
    vt = np.array([0.5, 0.5])
    at = np.array([0.5, 0.5])   # ambos Q1
    vp = np.array([0.5, -0.5])
    ap = np.array([0.5, 0.5])   # Q1, Q2
    assert quadrant_accuracy(vt, at, vp, ap) == pytest.approx(0.5)


def test_quadrant_f1_macro_perfeita():
    v = np.array([0.5, -0.5, -0.5, 0.5])
    a = np.array([0.5, 0.5, -0.5, -0.5])
    assert quadrant_f1_macro(v, a, v, a) == pytest.approx(1.0)


def test_bootstrap_ci_contem_media():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.2, 0.1, size=200)
    lo, hi = bootstrap_ci(vals, np.mean, n_boot=1000, seed=1)
    assert lo < 0.2 < hi
    assert lo < hi


def test_bootstrap_ci_ccc_pareado():
    """bootstrap_ci com stat pareada via matriz 2 colunas."""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.5, size=100)
    y = x + rng.normal(0, 0.1, size=100)
    m = np.column_stack([x, y])
    lo, hi = bootstrap_ci(m, lambda mm: ccc(mm[:, 0], mm[:, 1]),
                          n_boot=800, seed=3)
    assert 0.0 < lo <= hi <= 1.0


def test_bootstrap_ci_reprodutivel_com_seed():
    vals = np.random.default_rng(4).normal(size=50)
    a = bootstrap_ci(vals, np.mean, n_boot=500, seed=7)
    b = bootstrap_ci(vals, np.mean, n_boot=500, seed=7)
    assert a == b
