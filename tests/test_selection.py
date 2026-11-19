"""Testes de tcc_mer/selection.py: pool de discordantes e seleção diversificada."""

import numpy as np
import pandas as pd
import pytest

from tcc_mer.selection import (
    SelectionParams,
    apply_substitutions,
    attach_metadata,
    attach_voice,
    discordant_pool,
    english_share,
    macro_genre,
    select_diverse,
    word_count,
)


def _lfs(rows):
    """rows: (id, V1, A1, V3, A3) -> (lf1, lf2) no formato ``id, V_raw, A_raw``."""
    lf1 = pd.DataFrame([(r[0], r[1], r[2]) for r in rows], columns=["id", "V_raw", "A_raw"])
    lf2 = pd.DataFrame([(r[0], r[3], r[4]) for r in rows], columns=["id", "V_raw", "A_raw"])
    return lf1, lf2


@pytest.mark.parametrize(
    ("genre", "expected"),
    [
        ("samba-canção", "samba"),
        ("marcha-rancho", "marcha/frevo/maxixe"),
        ("baião", "nordestino"),
        ("rock nacional", "rock/pop"),
        ("MPB", "mpb"),
        (np.nan, "sem gênero"),
    ],
)
def test_macro_genre(genre, expected):
    assert macro_genre(genre) == expected


def test_english_share_distingue_idiomas():
    assert english_share("you and I love the way you are") > 0.8
    assert english_share("eu não sei o que você quer de mim") == 0.0
    assert english_share("123 !!!") == 0.0


def test_word_count():
    assert word_count("uma letra de cinco palavras") == 5


def test_pool_so_tem_faixas_discordantes_fora_do_gold():
    lf1, lf2 = _lfs(
        [
            ("a", 0.6, 0.6, -0.8, 0.6),  # áudio Q1, fusão (-0.1, 0.6) Q2: discordante
            ("b", 0.6, 0.6, 0.6, 0.6),  # sem divergência
            ("c", 0.6, 0.6, -0.8, 0.6),  # discordante, mas do gold
            ("d", 0.5, 0.5, -0.52, 0.5),  # fusão em V = -0.01: dentro da margem
        ]
    )
    pool = discordant_pool(lf1, lf2, gold_ids={"c"}, margin=0.05)
    assert list(pool.index) == ["a"]
    assert pool.loc["a", "familia"] == "A:letra→negativa"
    assert pool.loc["a", "q_audio"] == 1
    assert pool.loc["a", "q_fusao"] == 2


def test_pool_descarta_faixa_sem_um_dos_canais():
    lf1, lf2 = _lfs([("a", 0.6, 0.6, -0.8, 0.6), ("b", 0.6, 0.6, np.nan, np.nan)])
    assert list(discordant_pool(lf1, lf2, set(), margin=0.05).index) == ["a"]


def test_familias_de_transicao():
    lf1, lf2 = _lfs(
        [
            ("neg", 0.6, 0.6, -0.8, 0.6),  # V: + -> -
            ("pos", -0.6, 0.6, 0.8, 0.6),  # V: - -> +
            ("aro", 0.6, 0.6, 0.6, -0.9),  # só o arousal atravessa o eixo
        ]
    )
    pool = discordant_pool(lf1, lf2, set(), margin=0.05)
    assert pool["familia"].to_dict() == {
        "neg": "A:letra→negativa",
        "pos": "B:letra→positiva",
        "aro": "C:só arousal",
    }


def test_metadata_usa_estilo_provisorio_so_onde_falta_genero():
    lf1, lf2 = _lfs([("a", 0.6, 0.6, -0.8, 0.6), ("b", 0.6, 0.6, -0.8, 0.6)])
    pool = discordant_pool(lf1, lf2, set(), margin=0.05)
    metadata = pd.DataFrame(
        {
            "id": ["a", "b"],
            "grupo": ["historica", "gold"],
            "ano": [1955, 2010],
            "titulo": ["A", "B"],
            "interprete": ["x", "y"],
            "genero": [None, "rap"],
        }
    )
    result = attach_metadata(pool, metadata, pd.Series({"a": "samba", "b": "pop"}))
    assert result.loc["a", "macro"] == "samba"
    assert result.loc["a", "estilo_fonte"] == "provisório"
    assert result.loc["b", "macro"] == "rap"
    assert result.loc["a", "periodo"] == "1940-69"
    assert result.loc["b", "periodo"] == "2000+"
    assert result.loc["b", "origem"] == "gêneros"


def _candidates(n=24):
    """Pool sintético com períodos, famílias, macrogêneros e artistas variados."""
    periods = ["≤1939", "1940-69", "1970-99", "2000+"]
    families = ["A:letra→negativa", "B:letra→positiva", "C:só arousal"]
    return pd.DataFrame(
        {
            "periodo": [periods[i % 4] for i in range(n)],
            "familia": [families[i % 3] for i in range(n)],
            "macro": [f"estilo{i % 6}" for i in range(n)],
            "artista": [f"artista{i // 2}" for i in range(n)],
            "margem": np.linspace(0.1, 0.5, n),
            "palavras_letra": 100,
            "share_ingles": 0.0,
        },
        index=[f"t{i:02d}" for i in range(n)],
    )


def _with_voice(pool, voice=None):
    return attach_voice(pool, voice, threshold_db=18)


def test_selecao_e_deterministica():
    params = SelectionParams(n=8, reserve=2)
    first, _ = select_diverse(_with_voice(_candidates()), params)
    second, _ = select_diverse(_with_voice(_candidates()), params)
    assert list(first.index) == list(second.index)


def test_selecao_respeita_tamanho_e_nao_repete_faixas_entre_selecao_e_reserva():
    selected, reserve = select_diverse(_with_voice(_candidates()), SelectionParams(n=8, reserve=3))
    assert len(selected) == 8 and len(reserve) == 3
    assert not set(selected.index) & set(reserve.index)
    assert list(selected["ordem_selecao"]) == list(range(1, 9))
    assert list(reserve["ordem_selecao"]) == [9, 10, 11]


def test_selecao_respeita_tetos_por_artista_e_por_estilo():
    selected, _ = select_diverse(
        _with_voice(_candidates()), SelectionParams(n=10, reserve=0, cap_macro=2, cap_artist=1)
    )
    assert selected["artista"].value_counts().max() <= 1
    assert selected["macro"].value_counts().max() <= 2


def test_selecao_equilibra_periodos():
    selected, _ = select_diverse(
        _with_voice(_candidates()), SelectionParams(n=8, reserve=0, cap_macro=8)
    )
    assert selected["periodo"].value_counts().to_dict() == dict.fromkeys(
        selected["periodo"].unique(), 2
    )


def test_filtros_de_letra_curta_e_pouca_voz():
    pool = _candidates(8)
    pool.loc["t00", "palavras_letra"] = 10
    voice = pd.DataFrame({"frac_voz_18": 0.9}, index=pool.index)
    voice.loc["t01", "frac_voz_18"] = 0.1
    selected, reserve = select_diverse(_with_voice(pool, voice), SelectionParams(n=4, reserve=4))
    assert not {"t00", "t01"} & (set(selected.index) | set(reserve.index))


def test_filtro_de_ingles_e_opcional():
    pool = _with_voice(_candidates(8))
    pool.loc["t00", "share_ingles"] = 0.9
    unfiltered, _ = select_diverse(pool, SelectionParams(n=8, reserve=0))
    filtered, _ = select_diverse(pool, SelectionParams(n=7, reserve=0, max_english=0.30))
    assert "t00" in unfiltered.index
    assert "t00" not in filtered.index


def test_substituicao_manual_troca_a_faixa_e_mantem_a_posicao():
    pool = _with_voice(_candidates(12))
    selected, _ = select_diverse(pool, SelectionParams(n=6, reserve=0))
    removed = selected.index[2]
    added = next(track for track in pool.index if track not in selected.index)
    substitutions = pd.DataFrame({"removida": [removed], "incluida": [added], "motivo": ["teste"]})

    result = apply_substitutions(selected, pool, substitutions)

    assert removed not in result.index and added in result.index
    assert len(result) == len(selected)
    assert result.loc[added, "ordem_selecao"] == selected.loc[removed, "ordem_selecao"]
    assert list(result["ordem_selecao"]) == list(range(1, 7))


def test_substituicao_com_faixa_inexistente_levanta_erro():
    pool = _with_voice(_candidates(12))
    selected, _ = select_diverse(pool, SelectionParams(n=6, reserve=0))
    unknown = pd.DataFrame({"removida": ["nao-existe"], "incluida": [pool.index[0]]})
    with pytest.raises(ValueError, match="não está na seleção"):
        apply_substitutions(selected, pool, unknown)
