"""Seleção diversificada das faixas do experimento com ouvintes.

Pergunta do experimento: nas faixas em que o quadrante do áudio (LF1) difere do
quadrante da fusão simples, qual dos dois rótulos concorda mais com a percepção de
ouvintes brasileiros? As regras são fixadas antes da coleta e a seleção é
**determinística** (não aleatória):

* a faixa é *discordante* se ``quadrante(áudio) != quadrante(fusão simples)``, com
  ``|V|`` e ``|A|`` de ambos os pontos acima de uma margem mínima, para não contar
  cruzamentos de eixo por poucos centésimos;
* ficam de fora as faixas do gold set, as instrumentais, as de letra curta e as de
  trecho de 30 s com pouca voz (o LLM lê a letra inteira, o ouvinte escuta 30 s). Um
  filtro opcional (``max_english``) tira as de letra majoritariamente em inglês, já
  que os avaliadores são falantes de pt-BR;
* a escolha é gulosa, com cotas por período e por família de transição, teto por
  estilo e por artista, e desempate por margem e por um hash de ``(semente, id)``;
* substituições manuais explícitas (:func:`apply_substitutions`) trocam faixas já
  escolhidas por outras do pool, com o motivo registrado.

As colunas das saídas mantêm os nomes em português porque o formulário do experimento
as consome.
"""

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fusion import simple_fusion
from .quadrant import quadrants

__all__ = [
    "PERIODS",
    "FAMILY_QUOTAS",
    "SelectionParams",
    "macro_genre",
    "word_count",
    "english_share",
    "discordant_pool",
    "attach_metadata",
    "attach_voice",
    "select_diverse",
    "apply_substitutions",
]

PERIODS = [
    ("≤1939", 0, 1939),
    ("1940-69", 1940, 1969),
    ("1970-99", 1970, 1999),
    ("2000+", 2000, 3000),
]

# Fração das faixas por família de transição do rótulo de áudio para a fusão simples.
FAMILY_QUOTAS = {"A:letra→negativa": 0.40, "B:letra→positiva": 0.40, "C:só arousal": 0.20}

_ORIGIN = {"historica": "histórico", "gold": "gêneros", "expansao": "gêneros-exp"}

_ENGLISH_FUNCTION_WORDS = {
    "the",
    "you",
    "and",
    "to",
    "of",
    "in",
    "it",
    "is",
    "that",
    "my",
    "we",
    "your",
    "with",
    "for",
    "on",
    "be",
    "this",
    "not",
    "but",
    "are",
    "was",
    "all",
    "just",
    "have",
    "love",
    "don't",
    "i'm",
    "she",
    "he",
    "they",
    "what",
    "like",
    "when",
    "baby",
    "know",
    "yes",
    "up",
    "out",
    "can",
    "will",
    "from",
    "now",
}
_PORTUGUESE_FUNCTION_WORDS = {
    "o",
    "os",
    "as",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "e",
    "que",
    "em",
    "um",
    "uma",
    "para",
    "com",
    "não",
    "nao",
    "eu",
    "você",
    "voce",
    "te",
    "se",
    "por",
    "mais",
    "meu",
    "minha",
    "seu",
    "sua",
    "é",
    "foi",
    "tem",
    "já",
    "ja",
    "só",
    "mas",
    "como",
    "quando",
    "está",
    "esta",
    "na",
    "nos",
    "nas",
    "ao",
    "aos",
    "pra",
}

_MACRO_GENRE_KEYWORDS = [
    ("marcha/frevo/maxixe", ("marcha", "frevo", "maxixe")),
    ("samba", ("samba",)),
    ("nordestino", ("baião", "baiao", "toada", "embolada", "forró", "forro", "xote", "coco")),
    ("dança de salão/choro", ("valsa", "polca", "chótis", "chotis", "choro", "tango", "fox")),
    (
        "canção/modinha/bolero",
        ("canção", "cancao", "modinha", "romança", "balada", "bolero", "guarânia", "lundu"),
    ),
    ("rural/folclórico", ("cururu", "cateretê", "moda-de-viola", "batuque", "caipira")),
    ("rap", ("rap", "hip")),
    ("soul/funk", ("soul", "funk", "disco")),
    ("rock/pop", ("rock", "pop")),
]


@dataclass(frozen=True)
class SelectionParams:
    """Parâmetros da seleção, registrados no protocolo."""

    margin: float = 0.05
    n: int = 50
    reserve: int = 10
    seed: int = 2026
    min_words: int = 40
    min_voice: float = 0.5
    voice_threshold_db: int = 18
    max_english: float | None = None
    cap_macro: int = 9
    cap_artist: int = 2


def macro_genre(genre) -> str:
    """Agrupa o gênero do manifesto em um macrogênero por palavras-chave."""
    if pd.isna(genre):
        return "sem gênero"
    text = str(genre).lower()
    for macro, keywords in _MACRO_GENRE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return macro
    return text.strip() or "outro"


def word_count(text: str) -> int:
    """Número de palavras da letra."""
    return len(re.findall(r"\w+", text))


def english_share(text: str) -> float:
    """Fração de palavras-função em inglês entre as palavras-função em inglês e em português.

    Heurística simples, com listas de palavras disjuntas.
    """
    words = re.findall(r"[a-záàâãéêíóôõúç']+", text.lower())
    english = sum(word in _ENGLISH_FUNCTION_WORDS for word in words)
    portuguese = sum(word in _PORTUGUESE_FUNCTION_WORDS for word in words)
    return english / (english + portuguese) if english + portuguese else 0.0


def discordant_pool(
    lf1: pd.DataFrame, lf2: pd.DataFrame, gold_ids: set[str], margin: float
) -> pd.DataFrame:
    """Faixas discordantes, fora do gold set, indexadas por ``id``.

    Colunas: ``V1, A1`` (áudio), ``V3, A3`` (letra), ``Vf, Af`` (fusão simples),
    ``q_audio``, ``q_fusao`` (quadrantes 1 a 4), ``margem`` e ``familia``.
    """
    audio = lf1.set_index("id").rename(columns={"V_raw": "V1", "A_raw": "A1"})
    lyrics = lf2.set_index("id").rename(columns={"V_raw": "V3", "A_raw": "A3"})
    pool = audio.join(lyrics).dropna()
    pool["Vf"], pool["Af"] = simple_fusion(pool["V1"], pool["A1"], pool["V3"], pool["A3"])
    pool["q_audio"] = _quadrant_number(pool["V1"], pool["A1"])
    pool["q_fusao"] = _quadrant_number(pool["Vf"], pool["Af"])
    pool["margem"] = pool[["V1", "A1", "Vf", "Af"]].abs().min(axis=1)

    pool = pool[(pool["q_audio"] != pool["q_fusao"]) & (pool["margem"] >= margin)].copy()
    pool = pool[~pool.index.isin(gold_ids)]
    pool["familia"] = np.where(
        (pool["V1"] >= 0) & (pool["Vf"] < 0),
        "A:letra→negativa",
        np.where((pool["V1"] < 0) & (pool["Vf"] >= 0), "B:letra→positiva", "C:só arousal"),
    )
    return pool


def _quadrant_number(valence: pd.Series, arousal: pd.Series) -> np.ndarray:
    return np.array([int(q[1]) for q in quadrants(valence, arousal)])


def attach_metadata(
    pool: pd.DataFrame, metadata: pd.DataFrame, provisional_styles: pd.Series | None = None
) -> pd.DataFrame:
    """Acrescenta título, artista, gênero, ano, período e macrogênero de cada faixa.

    Quando o manifesto não traz gênero (metade das históricas), o macrogênero vem do
    estilo provisório atribuído à faixa, se houver.

    Args:
        pool: saída de :func:`discordant_pool`.
        metadata: saída de :func:`tcc_mer.dataset.load_metadata`.
        provisional_styles: estilo por ``id`` para faixas sem gênero no manifesto.
    """
    meta = metadata.set_index("id")
    info = pd.DataFrame(
        {
            "titulo": meta["titulo"],
            "artista": meta["interprete"],
            "genero": meta["genero"],
            "ano": meta["ano"].astype(float),
            "origem": meta["grupo"].map(_ORIGIN),
        }
    )
    info["macro"] = info["genero"].map(macro_genre)
    info["estilo_fonte"] = np.where(info["genero"].isna(), "provisório", "metadado")
    if provisional_styles is not None:
        use_style = info.index.isin(provisional_styles.index) & info["genero"].isna()
        info.loc[use_style, "macro"] = provisional_styles.reindex(info.index[use_style]).values
    info["periodo"] = [_period(year) for year in info["ano"]]
    return pool.join(info)


def _period(year: float) -> str:
    if np.isnan(year):
        return "?"
    return next((name for name, low, high in PERIODS if low <= year <= high), "?")


def _tie_break(seed: int, track_id: str) -> float:
    return int(hashlib.sha256(f"{seed}:{track_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def attach_voice(pool: pd.DataFrame, voice: pd.DataFrame | None, threshold_db: int) -> pd.DataFrame:
    """Acrescenta ``frac_voz``, a fração do trecho de 30 s com voz, no limiar de ``threshold_db``.

    Args:
        pool: faixas candidatas.
        voice: saída do medidor de voz, com ``frac_voz_<limiar>`` por ``id``; ``None`` deixa
            a coluna vazia e a seleção fica sem filtro de voz.
    """
    pool = pool.copy()
    pool["frac_voz"] = (
        voice[f"frac_voz_{threshold_db}"].reindex(pool.index) if voice is not None else np.nan
    )
    return pool


def select_diverse(
    pool: pd.DataFrame, params: SelectionParams
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Escolhe ``params.n`` faixas diversificadas e ``params.reserve`` de reserva.

    Args:
        pool: faixas discordantes com metadados, ``palavras_letra``, ``share_ingles`` e
            ``frac_voz`` (ver :func:`attach_voice`); sem voz medida, não há filtro de voz.
        params: parâmetros da seleção.

    Returns:
        ``(selecionadas, reserva)``, com ``ordem_selecao``.
    """
    keep = pool["palavras_letra"] >= params.min_words
    if params.max_english is not None:
        keep &= pool["share_ingles"] < params.max_english
    if pool["frac_voz"].notna().any():
        keep &= pool["frac_voz"].fillna(0) >= params.min_voice
    candidates = pool[keep].copy()

    period_target = {name: params.n / len(PERIODS) for name, _, _ in PERIODS}
    family_target = {family: share * params.n for family, share in FAMILY_QUOTAS.items()}
    counts = {"periodo": {}, "familia": {}, "macro": {}, "artista": {}}
    chosen: list[str] = []
    remaining = candidates.copy()

    for rank in range(min(params.n + params.reserve, len(candidates))):
        best, best_score = None, -1e9
        for track_id, row in remaining.iterrows():
            if counts["macro"].get(row["macro"], 0) >= params.cap_macro and rank < params.n:
                continue
            if counts["artista"].get(row["artista"], 0) >= params.cap_artist:
                continue
            score = (
                2.0
                * max(
                    0.0,
                    period_target.get(row["periodo"], 0) - counts["periodo"].get(row["periodo"], 0),
                )
                + 2.0
                * max(0.0, family_target[row["familia"]] - counts["familia"].get(row["familia"], 0))
                + (3.0 if counts["macro"].get(row["macro"], 0) == 0 else 0.0)
                + 0.5 * row["margem"]
                + 0.01 * _tie_break(params.seed, track_id)
            )
            if score > best_score:
                best, best_score = track_id, score
        if best is None:
            break
        row = remaining.loc[best]
        for key, value in (
            ("periodo", row["periodo"]),
            ("familia", row["familia"]),
            ("macro", row["macro"]),
            ("artista", row["artista"]),
        ):
            counts[key][value] = counts[key].get(value, 0) + 1
        chosen.append(best)
        remaining = remaining.drop(best)

    selected = candidates.loc[chosen[: params.n]].copy()
    selected["ordem_selecao"] = range(1, len(selected) + 1)
    reserve = candidates.loc[chosen[params.n :]].copy()
    reserve["ordem_selecao"] = range(params.n + 1, params.n + 1 + len(reserve))
    return selected, reserve


def apply_substitutions(
    selected: pd.DataFrame, pool: pd.DataFrame, substitutions: pd.DataFrame
) -> pd.DataFrame:
    """Troca faixas selecionadas por outras do pool, mantendo a posição de cada uma.

    Args:
        selected: faixas escolhidas por :func:`select_diverse`.
        pool: faixas candidatas, com as mesmas colunas de ``selected``.
        substitutions: colunas ``removida`` e ``incluida`` (ids), e ``motivo``.

    Raises:
        ValueError: se a faixa removida não estiver selecionada ou a incluída, no pool.
    """
    result = selected.copy()
    for removed, added in zip(substitutions["removida"], substitutions["incluida"]):
        if removed not in result.index:
            raise ValueError(f"faixa a remover não está na seleção: {removed}")
        if added not in pool.index:
            raise ValueError(f"faixa a incluir não está no pool: {added}")
        replacement = pool.loc[[added]].copy()
        replacement["ordem_selecao"] = result.loc[removed, "ordem_selecao"]
        result = pd.concat([result.drop(removed), replacement]).sort_values("ordem_selecao")
    return result
