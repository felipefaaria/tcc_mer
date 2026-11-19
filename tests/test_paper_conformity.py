"""Parâmetros e números declarados no artigo.

Se o código mudar um limiar, uma cota ou uma contagem que o artigo cita, estes testes avisam.
"""

import sys

import numpy as np
import pytest

from tcc_mer import lf2_llm
from tcc_mer.anthropic_adapter import PRICES
from tcc_mer.dataset import DATASET_PATH, read_dataset
from tcc_mer.rater_qc import (
    ANCHOR_MIN_ACCURACY,
    MAX_MODAL_SHARE,
    MIN_ITEM_REST_RHO,
    MIN_SECONDS_PER_ITEM,
)
from tcc_mer.selection import FAMILY_QUOTAS, SelectionParams
from tcc_mer.validation import CATEGORY_CENTER, MIN_RATERS
from tests.helpers import load_script


def test_parametros_da_selecao_das_faixas():
    params = SelectionParams()
    assert params.margin == 0.05
    assert (params.n, params.reserve, params.seed) == (50, 10, 2026)
    assert params.min_words == 40
    assert (params.min_voice, params.voice_threshold_db) == (0.5, 18)
    assert (params.cap_macro, params.cap_artist) == (9, 2)
    assert FAMILY_QUOTAS == {
        "A:letra→negativa": 0.40,
        "B:letra→positiva": 0.40,
        "C:só arousal": 0.20,
    }


def test_criterios_de_sinalizacao_de_avaliadores():
    assert ANCHOR_MIN_ACCURACY == 0.75
    assert MIN_ITEM_REST_RHO == 0.15
    assert MAX_MODAL_SHARE == 0.70
    assert MIN_SECONDS_PER_ITEM == 12


def test_codificacao_das_respostas_do_experimento():
    assert CATEGORY_CENTER == 3  # categorias 1 a 5 viram -2 a +2
    assert MIN_RATERS == 3


def test_lf2_usa_sonnet_5_com_esforco_baixo_e_prompt_v1(monkeypatch):
    module = load_script("scripts/rotulagem/gerar_lf2_llm.py")
    monkeypatch.setattr(sys, "argv", ["gerar_lf2_llm"])
    args = module.parse_args()
    assert (args.model, args.effort) == ("claude-sonnet-5", "low")
    assert args.model in PRICES
    assert lf2_llm.PROMPT_VERSION == "v1"
    assert (lf2_llm.LLM_LO, lf2_llm.LLM_HI) == (1.0, 9.0)


pytestmark_dataset = pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset não gerado")


@pytestmark_dataset
def test_numeros_do_corpus_do_artigo():
    table = read_dataset()
    assert len(table) == 606
    assert table["grupo"].value_counts().to_dict() == {"historica": 516, "gold": 53, "expansao": 37}
    assert int(table["instrumental"].sum()) == 18

    both = table[table["fusao_simples_valencia"].notna()]
    assert len(both) == 588
    assert (both["quadrante_lf1"] != both["quadrante_lf2"]).sum() == 354
    assert (both["quadrante_lf1"] != both["quadrante_fusao_simples"]).sum() == 265
    assert (np.sign(both["lf1_audio_valencia"]) != np.sign(both["lf2_llm_valencia"])).sum() == 265
    assert (np.sign(both["lf1_audio_arousal"]) != np.sign(both["lf2_llm_arousal"])).sum() == 232
    assert ((both["quadrante_lf1"] == "Q2").sum(), (both["quadrante_lf2"] == "Q2").sum()) == (
        6,
        128,
    )
    assert both["dissociacao_valencia"].mean() == pytest.approx(-0.12, abs=0.005)
    assert table["no_experimento"].sum() == 50
