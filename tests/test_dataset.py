"""Testes de tcc_mer/dataset.py com manifestos e saídas de LF sintéticos."""

import numpy as np
import pandas as pd
import pytest

from tcc_mer.dataset import (
    DATASET_COLUMNS,
    GROUP_EXPANSION,
    GROUP_GOLD,
    GROUP_HISTORICAL,
    build_annotated_corpus,
    load_metadata,
    read_dataset,
)


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "id": ["1990-01", "1990-02", "INS01", "RAP01"],
            "grupo": [GROUP_HISTORICAL, GROUP_HISTORICAL, GROUP_GOLD, GROUP_EXPANSION],
            "ano": pd.array([1990, 1990, 1980, 2004], dtype="Int64"),
            "ano_aproximado": [False, False, True, True],
            "titulo": ["A", "B", "C", "D"],
            "interprete": ["w", "x", "y", "z"],
            "genero": ["samba", None, "choro", "rap"],
            "instrumental": [False, False, True, False],
        }
    )


@pytest.fixture
def lf1():
    return pd.DataFrame(
        {
            "id": ["1990-01", "1990-02", "INS01", "RAP01"],
            "V_raw": [0.8, -0.4, 0.3, 0.2],
            "A_raw": [0.6, -0.2, -0.1, 0.5],
        }
    )


@pytest.fixture
def lf2():
    """A faixa instrumental tem nota do LLM (texto de preenchimento) e deve ser descartada."""
    return pd.DataFrame(
        {
            "id": ["1990-01", "1990-02", "INS01", "RAP01"],
            "V_raw": [-0.6, -0.8, 0.0, 0.4],
            "A_raw": [0.0, 0.2, -1.0, -0.3],
        }
    )


def test_colunas_e_ordem_do_esquema(metadata, lf1, lf2):
    table = build_annotated_corpus(metadata, lf1, lf2)
    assert list(table.columns) == DATASET_COLUMNS
    assert list(table["id"]) == list(metadata["id"])


def test_valores_por_canal_sao_os_brutos_das_lfs(metadata, lf1, lf2):
    row = build_annotated_corpus(metadata, lf1, lf2).iloc[0]
    assert (row["lf1_audio_valencia"], row["lf1_audio_arousal"]) == (0.8, 0.6)
    assert (row["lf2_llm_valencia"], row["lf2_llm_arousal"]) == (-0.6, 0.0)


def test_fusoes_e_rotulos_de_emocao(metadata, lf1, lf2):
    row = build_annotated_corpus(metadata, lf1, lf2).iloc[0]
    assert row["fusao_simples_valencia"] == pytest.approx(0.1)
    assert row["fusao_simples_arousal"] == pytest.approx(0.3)
    assert row["quadrante_fusao_simples"] == "Q1"
    assert row["emocao_fusao_simples"] == "alegria"
    assert row["fusao_dimensao_valencia"] == pytest.approx(-0.6)
    assert row["fusao_dimensao_arousal"] == pytest.approx(0.6)
    assert row["quadrante_fusao_dimensao"] == "Q2"
    assert row["emocao_fusao_dimensao"] == "tensão"


def test_dissociacao_e_diferenca_letra_menos_audio(metadata, lf1, lf2):
    row = build_annotated_corpus(metadata, lf1, lf2).iloc[0]
    assert row["dissociacao_valencia"] == pytest.approx(-1.4)
    assert row["dissociacao_arousal"] == pytest.approx(-0.6)
    assert row["dissociacao_magnitude"] == pytest.approx(np.hypot(1.4, 0.6), abs=1e-4)
    assert bool(row["dissociacao_quadrante"]) is True
    assert (row["quadrante_lf1"], row["quadrante_lf2"]) == ("Q1", "Q2")


def test_margem_e_a_distancia_ao_eixo_mais_proximo(metadata, lf1, lf2):
    row = build_annotated_corpus(metadata, lf1, lf2).iloc[0]
    assert row["margem_fusao_simples"] == pytest.approx(0.1)
    assert row["margem_fusao_dimensao"] == pytest.approx(0.6)


def test_instrumental_fica_nan_em_tudo_que_depende_da_letra(metadata, lf1, lf2):
    row = build_annotated_corpus(metadata, lf1, lf2).set_index("id").loc["INS01"]
    for column in (
        "lf2_llm_valencia",
        "fusao_simples_valencia",
        "fusao_dimensao_arousal",
        "dissociacao_valencia",
        "dissociacao_magnitude",
        "margem_fusao_simples",
    ):
        assert np.isnan(row[column]), column
    assert pd.isna(row["dissociacao_quadrante"])
    assert row["quadrante_lf2"] == ""
    assert row["quadrante_fusao_simples"] == ""
    assert row["emocao_fusao_simples"] == ""
    assert row["quadrante_lf1"] == "Q4"  # o áudio continua rotulado


def test_faixas_do_experimento_sao_marcadas(metadata, lf1, lf2):
    table = build_annotated_corpus(metadata, lf1, lf2, experiment_ids=["1990-02"])
    assert list(table["no_experimento"]) == [False, True, False, False]


def test_faixa_sem_linha_na_lf_levanta_erro(metadata, lf1, lf2):
    with pytest.raises(ValueError, match="sem linha em LF2"):
        build_annotated_corpus(metadata, lf1, lf2.iloc[:-1])


def test_ids_duplicados_nos_metadados_levantam_erro(metadata, lf1, lf2):
    duplicated = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="ids duplicados"):
        build_annotated_corpus(duplicated, lf1, lf2)


def test_leitura_devolve_quadrantes_vazios_como_string_vazia(tmp_path, metadata, lf1, lf2):
    path = tmp_path / "corpus.csv"
    build_annotated_corpus(metadata, lf1, lf2).to_csv(path, index=False)
    table = read_dataset(path).set_index("id")
    assert table.loc["INS01", "quadrante_fusao_simples"] == ""
    assert table.loc["1990-01", "quadrante_fusao_simples"] == "Q1"
    assert pd.isna(table.loc["INS01", "dissociacao_quadrante"])


def test_load_metadata_unifica_os_tres_manifestos(tmp_path):
    historical = tmp_path / "historical.csv"
    gold = tmp_path / "gold.csv"
    expansion = tmp_path / "expansion.csv"
    pd.DataFrame(
        {
            "id": ["1901-01", "1901-02"],
            "ano": [1901, 1901],
            "faixa": ["A", "B"],
            "interprete": ["x", "y"],
            "genero": ["lundu", None],
            "instrumental": ["não", "sim"],
        }
    ).to_csv(historical, index=False)
    pd.DataFrame(
        {
            "id": ["BOS01", "INS01"],
            "titulo": ["C", "D"],
            "artista": ["z", "w"],
            "genero": ["bossa nova", "choro"],
            "genero_macro": ["bossa", "choro"],
            "ano_aprox": ["1962", "1951"],
            "tem_letra": ["sim", "não"],
        }
    ).to_csv(gold, index=False)
    pd.DataFrame(
        {
            "id": ["RAP01"],
            "faixa": ["E"],
            "interprete": ["v"],
            "genero_macro_ou_ritmo": ["rap"],
            "ano_aprox": ["1999"],
            "instrumental": [False],
        }
    ).to_csv(expansion, index=False)

    table = load_metadata(historical, gold, expansion).set_index("id")

    assert list(table["grupo"]) == [GROUP_HISTORICAL] * 2 + [GROUP_GOLD] * 2 + [GROUP_EXPANSION]
    assert list(table["instrumental"]) == [False, True, False, True, False]
    assert list(table["ano_aproximado"]) == [False, False, True, True, True]
    assert table.loc["BOS01", "ano"] == 1962
    assert table.loc["BOS01", "interprete"] == "z"
    assert table.loc["RAP01", "genero"] == "rap"
    assert table.loc["BOS01", "genero"] == "bossa"
