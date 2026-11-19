"""Testes de src/lf2_lexicon.py — estimador léxico ANEW-Br (mini-léxico sintético)."""
import numpy as np
import pytest

from src.lf2_lexicon import (
    AnewBrLexicon,
    estimate_va_from_lyrics,
    make_spacy_lemmatizer,
    run_batch,
    tokenize,
)
from src.scale import to_unit


@pytest.fixture
def lexicon():
    # mini-léxico sintético (valência, arousal) em SAM 1-9
    return AnewBrLexicon({
        "feliz": (8.0, 6.0),
        "alegria": (8.5, 6.5),
        "triste": (2.0, 3.0),
        "morte": (1.5, 5.0),
        "festa": (7.5, 8.0),
        "calmo": (6.0, 2.0),
    })


def test_tokenize_mantem_acentos_remove_pontuacao():
    toks = tokenize("Alegria, alegria! Não há dor...")
    assert toks == ["alegria", "alegria", "não", "há", "dor"]


def test_tokenize_vazio():
    assert tokenize("") == []
    assert tokenize("123 456") == []


def test_estimativa_positiva(lexicon):
    res = estimate_va_from_lyrics("feliz alegria festa", lexicon)
    # média SAM de valência (8+8.5+7.5)/3 = 8.0 -> to_unit
    assert res.v_raw == pytest.approx(to_unit(8.0, 1, 9))
    assert res.v_raw > 0  # positivo em [-1,1]
    assert res.meta["n_matched"] == 3


def test_estimativa_negativa(lexicon):
    res = estimate_va_from_lyrics("triste morte", lexicon)
    assert res.v_raw < 0  # letra triste -> valência negativa
    assert res.meta["coverage"] == pytest.approx(1.0)


def test_instrumental_retorna_nan(lexicon):
    res = estimate_va_from_lyrics("qualquer coisa", lexicon, tem_letra=False)
    assert np.isnan(res.v_raw)
    assert np.isnan(res.a_raw)
    assert res.meta["reason"] == "instrumental"


def test_sem_match_retorna_nan_nao_zero(lexicon):
    """Nenhuma palavra no léxico -> NaN explícito, jamais zero."""
    res = estimate_va_from_lyrics("xyzzy plugh quux", lexicon)
    assert np.isnan(res.v_raw)
    assert res.v_raw != 0.0 or np.isnan(res.v_raw)
    assert res.meta["n_matched"] == 0


def test_cobertura_parcial(lexicon):
    res = estimate_va_from_lyrics("feliz xyzzy plugh", lexicon)
    assert res.meta["n_tokens"] == 3
    assert res.meta["n_matched"] == 1
    assert res.meta["coverage"] == pytest.approx(1 / 3)


def test_ponderacao_por_frequencia(lexicon):
    """weight_by_freq=True: repetições puxam a média; =False usa únicos."""
    lyrics = "triste triste triste feliz"
    freq = estimate_va_from_lyrics(lyrics, lexicon, weight_by_freq=True)
    uniq = estimate_va_from_lyrics(lyrics, lexicon, weight_by_freq=False)
    # ponderado por frequência: 3x triste -> mais negativo que a média de únicos
    assert freq.v_raw < uniq.v_raw


def test_lemmatizer_injetavel(lexicon):
    """Um lematizador que mapeia 'felizes'->'feliz' passa a casar no léxico."""
    def fake_lemmatizer(tokens):
        return ["feliz" if t == "felizes" else t for t in tokens]

    sem = estimate_va_from_lyrics("felizes", lexicon)
    com = estimate_va_from_lyrics("felizes", lexicon, lemmatizer=fake_lemmatizer)
    assert np.isnan(sem.v_raw)          # sem lematizar não casa
    assert com.meta["n_matched"] == 1   # lematizado casa
    assert com.meta["lemmatized"] is True


def test_make_spacy_lemmatizer_sem_spacy_da_import_error():
    """Sem spaCy, o factory falha de forma clara (a LF2 segue sem lematizador).

    Pulado se spaCy estiver instalado, pois aí tentaria carregar o modelo real.
    """
    import importlib.util
    if importlib.util.find_spec("spacy") is not None:
        pytest.skip("spaCy instalado neste ambiente; teste não se aplica")
    with pytest.raises(ImportError):
        make_spacy_lemmatizer()


def test_run_batch_preenche_ids(lexicon):
    tracks = [
        ("t1", "feliz festa", True),
        ("t2", "instrumental ignorada", False),
        ("t3", "triste morte", True),
    ]
    out = run_batch(tracks, lexicon)
    assert [o.track_id for o in out] == ["t1", "t2", "t3"]
    assert np.isnan(out[1].v_raw)       # instrumental
    assert out[0].v_raw > 0 > out[2].v_raw
