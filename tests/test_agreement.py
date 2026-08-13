"""Testes de src/agreement.py — alfa de Krippendorff intervalar + leave-one-out."""
import numpy as np
import pytest

from src.agreement import alpha_leave_one_out, krippendorff_alpha


def test_concordancia_perfeita_alpha_1():
    """Todas as colunas idênticas => alfa = 1."""
    ratings = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ])
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_valor_conhecido_hand_computed():
    """Caso fechado calculado à mão.

    u1=[1,1,1] (perfeito), u2=[2,2,3] (um discordante).
      Do = (0 + 4/2)/6 = 2/6 = 1/3
      pool=[1,1,1,2,2,3]: Σ_{i,j}(Pi-Pj)² = 2·(6·20 - 100) = 40
      De = 40/(6·5) = 4/3
      α  = 1 - (1/3)/(4/3) = 1 - 1/4 = 0.75
    """
    ratings = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 3.0],
    ])
    assert krippendorff_alpha(ratings) == pytest.approx(0.75, abs=1e-9)


def test_uma_unidade_so_alpha_0():
    """Uma única unidade não distingue sinal de ruído => alfa = 0."""
    ratings = np.array([[1.0, 2.0]])
    assert krippendorff_alpha(ratings) == pytest.approx(0.0)


def test_missing_values_sobreposicao_parcial():
    """NaN (ausência) é tolerado; unidades com <2 valores são ignoradas."""
    ratings = np.array([
        [1.0, 1.0, np.nan],
        [2.0, np.nan, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, np.nan, np.nan],  # ignorada (só 1 valor)
    ])
    a = krippendorff_alpha(ratings)
    assert a == pytest.approx(1.0)  # onde há pares, concordam perfeitamente


def test_discordancia_reduz_alpha():
    perfeito = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    ruidoso = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    assert krippendorff_alpha(ruidoso) < krippendorff_alpha(perfeito)


def test_sem_unidades_pareaveis_nan():
    ratings = np.array([[1.0, np.nan], [np.nan, 2.0]])
    assert np.isnan(krippendorff_alpha(ratings))


def test_valencia_e_arousal_sao_chamadas_separadas():
    """A API força uma matriz por dimensão (nunca combinar V e A)."""
    v = np.array([[0.5, 0.4], [-0.2, -0.3]])
    a = np.array([[0.9, 0.9], [0.8, 0.85]])
    av = krippendorff_alpha(v)
    aa = krippendorff_alpha(a)
    assert av != aa  # dimensões independentes


def test_leave_one_out_detecta_outlier():
    """Remover um anotador destoante deve elevar bastante o alfa."""
    # 2 anotadores concordam, o 3º é ruído invertido.
    ratings = np.array([
        [1.0, 1.1, 3.0],
        [2.0, 2.0, 1.0],
        [3.0, 2.9, 1.0],
        [4.0, 4.1, 2.0],
    ])
    ids = ["A", "B", "C"]
    loo = alpha_leave_one_out(ratings, ids)
    base = krippendorff_alpha(ratings)
    # remover C (o outlier) sobe o alfa acima do baseline e acima de remover A/B
    assert loo["C"] > base
    assert loo["C"] > loo["A"]
    assert loo["C"] > loo["B"]


def test_leave_one_out_valida_tamanho_ids():
    ratings = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        alpha_leave_one_out(ratings, ["so_um_id"])
