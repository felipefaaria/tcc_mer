"""Invariantes da tabela publicada em ``dataset/corpus_anotado.csv``."""

import json

import numpy as np
import pytest

from tcc_mer.dataset import DATASET_COLUMNS, DATASET_PATH, read_dataset
from tcc_mer.paths import DATASET_DIR
from tcc_mer.quadrant import EMOTION_BY_QUADRANT, quadrants

pytestmark = pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset ainda não gerado")


@pytest.fixture(scope="module")
def table():
    return read_dataset()


def test_esquema_e_ids(table):
    assert list(table.columns) == DATASET_COLUMNS
    assert table["id"].is_unique


def test_tamanho_e_grupos_do_corpus(table):
    assert len(table) == 606
    assert table["grupo"].value_counts().to_dict() == {"historica": 516, "gold": 53, "expansao": 37}
    assert table["no_experimento"].sum() == 50


def test_instrumentais_ficam_sem_letra_e_sem_fusao(table):
    instrumentals = table[table["instrumental"]]
    assert len(instrumentals) == 18
    dependent = [
        c for c in DATASET_COLUMNS if c.startswith(("lf2_", "fusao_", "dissociacao_", "margem_"))
    ]
    assert instrumentals[dependent].isna().all().all()
    assert instrumentals["quadrante_lf1"].ne("").all()
    assert table[~table["instrumental"]][dependent].notna().all().all()
    assert table["fusao_simples_valencia"].notna().sum() == 588


def test_valores_em_escala_unitaria(table):
    for column in table.columns[table.columns.str.contains("valencia|arousal")]:
        limit = 2 if column.startswith("dissociacao") else 1
        assert table[column].abs().max() <= limit, column


def test_quadrantes_correspondem_aos_valores(table):
    """Só para pontos longe do eixo, onde o arredondamento a 4 casas não muda o sinal."""
    for name, v_col, a_col in [
        ("quadrante_lf1", "lf1_audio_valencia", "lf1_audio_arousal"),
        ("quadrante_lf2", "lf2_llm_valencia", "lf2_llm_arousal"),
        ("quadrante_fusao_simples", "fusao_simples_valencia", "fusao_simples_arousal"),
        ("quadrante_fusao_dimensao", "fusao_dimensao_valencia", "fusao_dimensao_arousal"),
    ]:
        away = (table[[v_col, a_col]].abs() > 1e-3).all(axis=1) | table[v_col].isna()
        expected = quadrants(table.loc[away, v_col], table.loc[away, a_col])
        assert (table.loc[away, name].to_numpy() == expected).all(), name


def test_emocao_e_o_nome_do_quadrante(table):
    for kind in ("fusao_simples", "fusao_dimensao"):
        expected = table[f"quadrante_{kind}"].map(EMOTION_BY_QUADRANT).fillna("")
        assert (table[f"emocao_{kind}"] == expected).all()


def test_dissociacao_e_letra_menos_audio(table):
    both = table["lf2_llm_valencia"].notna()
    d_v = table.loc[both, "lf2_llm_valencia"] - table.loc[both, "lf1_audio_valencia"]
    assert np.allclose(d_v, table.loc[both, "dissociacao_valencia"], atol=2e-4)


def test_dissociacao_de_quadrante_e_consistente(table):
    both = table["quadrante_lf2"] != ""
    differs = table.loc[both, "quadrante_lf1"] != table.loc[both, "quadrante_lf2"]
    assert (table.loc[both, "dissociacao_quadrante"].astype(bool) == differs).all()


def test_dicionario_documenta_todas_as_colunas():
    readme = (DATASET_DIR / "README.md").read_text(encoding="utf-8")
    undocumented = [column for column in DATASET_COLUMNS if f"`{column}`" not in readme]
    assert not undocumented, undocumented


def test_proveniencia_bate_com_a_tabela(table):
    provenance = json.loads((DATASET_DIR / "proveniencia.json").read_text(encoding="utf-8"))
    assert provenance["n_faixas"] == len(table)
    assert provenance["n_instrumentais"] == int(table["instrumental"].sum())
    assert set(provenance["entradas_sha256"]) == {"lf1_raw.csv", "lf2_raw.csv"}
