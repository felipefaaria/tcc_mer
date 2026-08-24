"""Testes de src/aggregate_annotations.py — consolidação do gold."""
import numpy as np
import pandas as pd
import pytest

from src.aggregate_annotations import (
    agreement_table,
    load_annotations_xlsx,
    to_gold_wide,
)


def _df(rows):
    cols = ["pid", "modalidade", "trackId", "valencia", "arousal", "naoConheco"]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def longo():
    # T1: melodia (A e B), letra (A); T2: melodia (A)
    return _df([
        ["A", "melodia", "T1", 1, 5, 0],
        ["B", "melodia", "T1", 9, 5, 0],
        ["A", "letra",   "T1", 5, 1, 0],
        ["A", "melodia", "T2", 5, 9, 0],
    ])


def test_gold_media_e_conversao(longo):
    g = to_gold_wide(longo)
    # T1 melodia valencia média (1+9)/2=5 -> to_unit(5,1,9)=0
    assert g.loc["T1", "V_audio"] == pytest.approx(0.0)
    assert g.loc["T1", "N_audio"] == 2
    # T1 letra: 1 anotador, valencia 5 -> 0 ; arousal 1 -> -1
    assert g.loc["T1", "V_letra"] == pytest.approx(0.0)
    assert g.loc["T1", "A_letra"] == pytest.approx(-1.0)
    assert g.loc["T1", "N_letra"] == 1


def test_celula_ausente_vira_nan(longo):
    g = to_gold_wide(longo)
    # T2 não tem letra
    assert np.isnan(g.loc["T2", "V_letra"])
    assert g.loc["T2", "N_letra"] == 0
    # T2 melodia arousal 9 -> +1
    assert g.loc["T2", "A_audio"] == pytest.approx(1.0)


def test_teste_reteste_promediado():
    # mesmo (pid,track,modalidade) duas vezes -> média antes de agregar
    df = _df([
        ["A", "letra", "T1", 1, 5, 0],
        ["A", "letra", "T1", 9, 5, 0],  # reteste
    ])
    g = to_gold_wide(df)
    # média das duas respostas do mesmo anotador = 5 -> to_unit = 0; N=1 anotador
    assert g.loc["T1", "V_letra"] == pytest.approx(0.0)
    assert g.loc["T1", "N_letra"] == 1


def test_drop_naoconheco():
    df = _df([
        ["A", "letra", "T1", 9, 5, 0],
        ["B", "letra", "T1", 1, 5, 1],  # não conheço
    ])
    com = to_gold_wide(df, drop_naoconheco=False)
    sem = to_gold_wide(df, drop_naoconheco=True)
    assert com.loc["T1", "N_letra"] == 2
    assert sem.loc["T1", "N_letra"] == 1
    # sem a resposta B (val=1), a média sobe
    assert sem.loc["T1", "V_letra"] > com.loc["T1", "V_letra"]


def test_agreement_table_estrutura(longo):
    agr = agreement_table(longo)
    assert set(agr["modalidade"]) == {"melodia", "letra", "completa"}
    assert len(agr) == 6  # 3 modalidades x 2 dimensões
    assert {"alpha", "n_faixas", "n_anotadores"}.issubset(agr.columns)


def test_load_valida_colunas(tmp_path):
    # falta 'arousal'
    ruim = pd.DataFrame({"pid": ["A"], "modalidade": ["letra"],
                         "trackId": ["T1"], "valencia": [5]})
    p = tmp_path / "ruim.xlsx"
    ruim.to_excel(p, sheet_name="respostas", index=False)
    with pytest.raises(ValueError):
        load_annotations_xlsx(p)


def test_load_valida_modalidade(tmp_path):
    ruim = _df([["A", "instrumental_solo", "T1", 5, 5, 0]])
    p = tmp_path / "modruim.xlsx"
    ruim.to_excel(p, sheet_name="respostas", index=False)
    with pytest.raises(ValueError):
        load_annotations_xlsx(p)


def test_load_ok(tmp_path, longo):
    p = tmp_path / "ok.xlsx"
    longo.to_excel(p, sheet_name="respostas", index=False)
    df = load_annotations_xlsx(p)
    assert len(df) == 4
    assert "naoConheco" in df.columns
