"""Testes de tcc_mer/validation.py com respostas sintéticas."""

import numpy as np
import pandas as pd
import pytest

from tcc_mer.validation import (
    analyzable,
    anchor_accuracy,
    duel,
    human_quadrant,
    load_item_map,
    load_responses,
    ordinal_alpha,
    pairwise_quadrant_agreement,
    repetition_agreement,
    track_labels,
)


@pytest.fixture
def item_map(tmp_path):
    rows = [
        # codigo, tipo, trackId, bloco, periodo, familia, macro, q_audio, q_fusao,
        # V1, A1, V3, A3, Vf, Af, consenso
        ("V01", "alvo", "t1", "A", "2000+", "A", "samba", 1, 2, 0.5, 0.5, -0.9, 0.5, -0.2, 0.5, ""),
        (
            "V02",
            "alvo",
            "t2",
            "A",
            "2000+",
            "A",
            "samba",
            3,
            4,
            -0.5,
            -0.5,
            0.9,
            -0.5,
            0.2,
            -0.5,
            "",
        ),
        ("V03", "alvo", "t3", "B", "2000+", "A", "samba", 1, 4, 0.5, 0.5, 0.9, -0.5, 0.2, -0.1, ""),
        ("A1", "ancora", "g1", "", "", "", "", "", "", 0, 0, 0, 0, 0, 0, "Q1"),
    ]
    path = tmp_path / "mapa.csv"
    pd.DataFrame(
        rows,
        columns=[
            "codigo",
            "tipo",
            "trackId",
            "bloco",
            "periodo",
            "familia",
            "macro",
            "q_audio",
            "q_fusao",
            "V1",
            "A1",
            "V3",
            "A3",
            "Vf",
            "Af",
            "quadrante_consenso",
        ],
    ).to_csv(path, index=False, encoding="utf-8-sig")
    return load_item_map(path)


def _responses(rows):
    """rows: (pid, item, tipo, valencia, arousal) -> DataFrame no formato de ``load_responses``."""
    df = pd.DataFrame(rows, columns=["pid", "item", "tipo", "valencia", "arousal"])
    df["itemId"] = df["item"]
    df.loc[df["tipo"] == "repeticao", "itemId"] += "#r"
    df["rater"] = df["pid"].str.lower()
    df["V"] = df["valencia"].astype(float) - 3
    df["A"] = df["arousal"].astype(float) - 3
    return df


def test_item_map_traduz_quadrantes_dos_modelos(item_map):
    assert item_map.loc["V01", "q_lf1"] == "Q1"
    assert item_map.loc["V01", "q_simple"] == "Q2"
    assert item_map.loc["V01", "q_dimension"] == "Q2"  # valência da letra, arousal do áudio
    assert item_map.loc["A1", "q_lf1"] == ""


@pytest.mark.parametrize(
    ("valence", "arousal", "expected"),
    [(1, 1, "Q1"), (-1, 2, "Q2"), (-2, -1, "Q3"), (1, -2, "Q4"), (0, 1, ""), (1, 0, "")],
)
def test_quadrante_humano_exige_mediana_nao_neutra(valence, arousal, expected):
    assert human_quadrant(valence, arousal) == expected


def test_load_responses_exclui_o_pesquisador_e_mantem_a_ultima_resposta(tmp_path):
    path = tmp_path / "respostas.csv"
    pd.DataFrame(
        {
            "pid": ["Ana", "Ana", "eu"],
            "itemId": ["V01", "V01", "V01"],
            "valencia": [1, 5, 3],
            "arousal": [3, 3, 3],
        }
    ).to_csv(path, index=False)
    responses = load_responses(path)
    assert list(responses["rater"]) == ["ana"]
    assert responses.iloc[0]["V"] == 2  # a última resposta (5) menos o centro (3)


def test_rotulo_humano_e_a_mediana_por_dimensao(item_map):
    responses = _responses(
        [
            (p, "V01", "alvo", v, a) for p, v, a in [("a", 1, 5), ("b", 2, 5), ("c", 5, 4)]
        ]  # V: -2,-1,+2 -> mediana -1; A: +2,+2,+1 -> +2
    )
    labels = track_labels(responses, item_map)
    assert labels.loc["V01", "n"] == 3
    assert labels.loc["V01", "V"] == -1
    assert labels.loc["V01", "A"] == 2
    assert labels.loc["V01", "q_human"] == "Q2"


def test_faixa_com_mediana_neutra_fica_fora_da_analise(item_map):
    responses = _responses(
        [(p, "V01", "alvo", v, 5) for p, v in [("a", 3), ("b", 3), ("c", 1)]]
        + [(p, "V02", "alvo", 1, 1) for p in "abc"]
    )
    kept = analyzable(track_labels(responses, item_map))
    assert list(kept.index) == ["V02"]


def test_faixa_com_poucos_avaliadores_fica_fora(item_map):
    responses = _responses([(p, "V01", "alvo", 1, 5) for p in "ab"])
    assert analyzable(track_labels(responses, item_map)).empty


def test_duelo_conta_quem_concorda_com_o_humano(item_map):
    # V01: humano Q2 = fusão (áudio Q1). V02: humano Q3 = áudio (fusão Q4). V03: humano Q4 = fusão.
    answers = {"V01": (1, 5), "V02": (1, 2), "V03": (5, 1)}
    responses = _responses(
        [(p, item, "alvo", v, a) for item, (v, a) in answers.items() for p in "abc"]
    )
    result = duel(analyzable(track_labels(responses, item_map)))
    assert (result.n_tracks, result.first, result.second, result.neither) == (3, 2, 1, 0)
    assert result.proportion == pytest.approx(2 / 3)
    assert 0 < result.ci_low < result.proportion < result.ci_high < 1


def test_duelo_sem_faixa_decidida_devolve_nan(item_map):
    responses = _responses([(p, "V01", "alvo", 5, 1) for p in "abc"])  # humano Q4, nenhum acerta
    result = duel(analyzable(track_labels(responses, item_map)))
    assert result.neither == 1
    assert np.isnan(result.p_value)


def test_ancoras_medem_acerto_de_sinal_e_de_quadrante(item_map):
    responses = _responses(
        [("a", "A1", "ancora", 5, 5), ("b", "A1", "ancora", 1, 5), ("c", "A1", "ancora", 3, 5)]
    )
    anchors = anchor_accuracy(responses, item_map)
    assert list(anchors["valence_ok"]) == [True, False, False]  # neutro não é positivo
    assert list(anchors["arousal_ok"]) == [True, True, True]
    assert list(anchors["exact_ok"]) == [True, False, False]


def test_repeticoes_pareiam_resposta_original_e_repetida():
    responses = _responses([("a", "V01", "alvo", 4, 2), ("a", "V01", "repeticao", 3, 2)])
    pairs = repetition_agreement(responses)
    assert pairs.iloc[0].to_dict() == {
        "rater": "a",
        "V_first": 1.0,
        "V_repeat": 0.0,
        "A_first": -1.0,
        "A_repeat": -1.0,
    }


def test_concordancia_de_quadrante_entre_pares():
    responses = _responses(
        [("a", "V01", "alvo", 5, 5), ("b", "V01", "alvo", 5, 5), ("c", "V01", "alvo", 1, 1)]
    )
    assert pairwise_quadrant_agreement(responses) == pytest.approx(1 / 3)


def test_alfa_ordinal_e_1_com_concordancia_perfeita():
    rows = [(p, f"V0{i}", "alvo", v, v) for p in "abc" for i, v in [(1, 1), (2, 3), (3, 5), (4, 2)]]
    alpha, shape = ordinal_alpha(_responses(rows), "V")
    assert alpha == pytest.approx(1.0)
    assert shape == (3, 4)
