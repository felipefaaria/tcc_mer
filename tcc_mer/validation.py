"""Análise do experimento de validação com ouvintes (música completa, 5 categorias).

Cada ouvinte avalia, por dimensão, uma entre cinco categorias (1 a 5), codificadas de
-2 a +2. O rótulo humano de uma faixa é a mediana das respostas em cada dimensão, e o
quadrante vem do sinal das duas medianas. Uma mediana neutra (zero) em qualquer
dimensão deixa a faixa **sem quadrante**, e ela fica fora do teste.

O teste pré-especificado conta, entre as faixas em que dois rótulos de modelo
(por padrão, o áudio e a fusão simples) dão quadrantes diferentes, quantas vezes cada
um coincide com o quadrante humano, e aplica um binomial exato bicaudal contra 50%.

As respostas têm uma linha por item respondido, com as colunas ``pid, itemId, item,
tipo, valencia, arousal, listenMs, conhece, bloco`` e ``respondidoEm``. O mapa de
itens (``data/validacao/mapa_itens.csv``) liga cada código de item à faixa e aos
rótulos dos modelos.
"""

from dataclasses import dataclass
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats

from .quadrant import quadrant_of, quadrants

__all__ = [
    "CATEGORY_CENTER",
    "DEFAULT_EXCLUDED",
    "MIN_RATERS",
    "DuelResult",
    "load_responses",
    "load_item_map",
    "human_quadrant",
    "track_labels",
    "analyzable",
    "duel",
    "anchor_accuracy",
    "repetition_agreement",
    "ordinal_alpha",
    "pairwise_quadrant_agreement",
]

CATEGORY_CENTER = 3
DEFAULT_EXCLUDED = ("eu",)
MIN_RATERS = 3

_MAP_NUMERIC = ("q_audio", "q_fusao", "V1", "A1", "V3", "A3", "Vf", "Af")


@dataclass(frozen=True)
class DuelResult:
    """Placar de dois rótulos de modelo contra o quadrante humano.

    ``first`` e ``second`` contam as faixas em que só o primeiro (ou só o segundo)
    rótulo coincide com o humano. ``proportion`` e o intervalo (Clopper-Pearson, 95%)
    dizem respeito à fração a favor do primeiro entre as duas.
    """

    n_tracks: int
    first: int
    second: int
    neither: int
    proportion: float
    ci_low: float
    ci_high: float
    p_value: float


def load_responses(path: str | Path, exclude: tuple[str, ...] = DEFAULT_EXCLUDED) -> pd.DataFrame:
    """Lê as respostas, tira os avaliadores excluídos e mantém a última resposta de cada item.

    Acrescenta ``rater`` (``pid`` normalizado) e ``V`` e ``A`` em -2..+2.
    """
    path = Path(path)
    df = (
        pd.read_excel(path, sheet_name="respostas")
        if path.suffix in (".xlsx", ".xls")
        else pd.read_csv(path)
    )
    df["rater"] = df["pid"].astype(str).str.strip().str.lower()
    df = df[~df["rater"].isin([name.lower() for name in exclude])]
    df = df.drop_duplicates(["rater", "itemId"], keep="last").copy()
    df["V"] = df["valencia"].astype(float) - CATEGORY_CENTER
    df["A"] = df["arousal"].astype(float) - CATEGORY_CENTER
    return df


def load_item_map(path: str | Path) -> pd.DataFrame:
    """Lê o mapa de itens, indexado pelo código do item.

    Acrescenta os quadrantes dos rótulos de modelo: ``q_lf1`` (áudio), ``q_simple``
    (fusão simples) e ``q_dimension`` (fusão por dimensão), como ``"Q1"``..``"Q4"``.
    """
    item_map = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False).set_index("codigo")
    for column in _MAP_NUMERIC:
        item_map[column] = pd.to_numeric(item_map[column], errors="coerce")
    item_map["q_lf1"] = item_map["q_audio"].map(_quadrant_name)
    item_map["q_simple"] = item_map["q_fusao"].map(_quadrant_name)
    item_map["q_dimension"] = quadrants(item_map["V3"], item_map["A1"])
    return item_map


def _quadrant_name(number: float) -> str:
    return f"Q{int(number)}" if pd.notna(number) else ""


def human_quadrant(valence_median: float, arousal_median: float) -> str:
    """Quadrante do rótulo humano; ``""`` se alguma mediana for neutra (zero)."""
    if valence_median == 0 or arousal_median == 0:
        return ""
    return quadrant_of(valence_median, arousal_median)


def track_labels(
    responses: pd.DataFrame, item_map: pd.DataFrame, aggregate: str = "median"
) -> pd.DataFrame:
    """Rótulo humano de cada faixa-alvo, com os rótulos dos modelos.

    Args:
        responses: saída de :func:`load_responses`.
        item_map: saída de :func:`load_item_map`.
        aggregate: ``"median"`` (regra pré-especificada) ou ``"mean"``.

    Returns:
        DataFrame indexado pelo código do item, com ``n`` (avaliadores), ``V`` e ``A``
        (agregados, em -2..+2), ``q_human`` e as colunas do mapa de itens.
    """
    targets = responses[responses["tipo"] == "alvo"].copy()
    targets["item"] = targets["item"].astype(str)
    labels = targets.groupby("item").agg(
        n=("V", "size"),
        V=("V", aggregate),
        A=("A", aggregate),
        V_mean=("V", "mean"),
        A_mean=("A", "mean"),
    )
    labels = labels.join(item_map.drop(columns=["tipo", "bloco"]))
    labels["q_human"] = [human_quadrant(v, a) for v, a in zip(labels["V"], labels["A"])]
    return labels


def analyzable(labels: pd.DataFrame, min_raters: int = MIN_RATERS) -> pd.DataFrame:
    """Faixas com avaliadores suficientes e quadrante humano definido."""
    return labels[(labels["n"] >= min_raters) & (labels["q_human"] != "")]


def duel(labels: pd.DataFrame, first: str = "q_simple", second: str = "q_lf1") -> DuelResult:
    """Compara dois rótulos de modelo com o quadrante humano nas faixas de ``labels``.

    Args:
        labels: faixas já filtradas (ver :func:`analyzable`).
        first: coluna do quadrante do primeiro rótulo (por padrão, a fusão simples).
        second: coluna do quadrante do segundo rótulo (por padrão, o áudio).
    """
    first_hits = labels[first] == labels["q_human"]
    second_hits = labels[second] == labels["q_human"]
    n_first = int((first_hits & ~second_hits).sum())
    n_second = int((~first_hits & second_hits).sum())
    n_neither = int((~first_hits & ~second_hits).sum())
    decided = n_first + n_second
    if decided == 0:
        return DuelResult(len(labels), n_first, n_second, n_neither, *(float("nan"),) * 4)
    test = stats.binomtest(n_first, decided, 0.5)
    ci = test.proportion_ci(confidence_level=0.95, method="exact")
    return DuelResult(
        len(labels),
        n_first,
        n_second,
        n_neither,
        n_first / decided,
        ci.low,
        ci.high,
        test.pvalue,
    )


def anchor_accuracy(responses: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por resposta a uma âncora, com os acertos de sinal e de quadrante.

    As âncoras são faixas do gold set de quadrante conhecido (``quadrante_consenso``).
    """
    anchors = responses[responses["tipo"] == "ancora"].join(
        item_map[["quadrante_consenso"]], on="item"
    )
    consensus = anchors["quadrante_consenso"].str[1].astype(int)
    anchors["q_human"] = [human_quadrant(v, a) for v, a in zip(anchors["V"], anchors["A"])]
    anchors["valence_ok"] = np.sign(anchors["V"]) == np.where(consensus.isin([1, 4]), 1, -1)
    anchors["arousal_ok"] = np.sign(anchors["A"]) == np.where(consensus.isin([1, 2]), 1, -1)
    anchors["exact_ok"] = anchors["q_human"] == anchors["quadrante_consenso"]
    return anchors


def repetition_agreement(responses: pd.DataFrame) -> pd.DataFrame:
    """Pares (resposta original, repetição) do mesmo avaliador.

    Colunas: ``rater, V_first, V_repeat, A_first, A_repeat``.
    """
    repeats = responses[responses["tipo"] == "repeticao"].copy()
    repeats["base"] = repeats["itemId"].str.replace("#r", "", regex=False)
    originals = responses[responses["tipo"] == "alvo"].set_index(["rater", "itemId"])
    rows = [
        (row["rater"], originals.loc[key, "V"], row["V"], originals.loc[key, "A"], row["A"])
        for _, row in repeats.iterrows()
        if (key := (row["rater"], row["base"])) in originals.index
    ]
    return pd.DataFrame(rows, columns=["rater", "V_first", "V_repeat", "A_first", "A_repeat"])


def ordinal_alpha(responses: pd.DataFrame, dimension: str) -> tuple[float, tuple[int, int]]:
    """Alfa de Krippendorff ordinal nas faixas-alvo, e o shape ``(avaliadores, faixas)``."""
    targets = responses[responses["tipo"] == "alvo"].copy()
    targets["item"] = targets["item"].astype(str)
    matrix = targets.pivot_table(index="rater", columns="item", values=dimension, aggfunc="first")
    if matrix.shape[0] < 2:
        return float("nan"), matrix.shape
    try:
        alpha = krippendorff.alpha(
            reliability_data=matrix.values.astype(float), level_of_measurement="ordinal"
        )
    except ValueError:
        alpha = float("nan")
    return float(alpha), matrix.shape


def pairwise_quadrant_agreement(responses: pd.DataFrame) -> float:
    """Fração dos pares de avaliadores que dão o mesmo quadrante à mesma faixa-alvo."""
    targets = responses[responses["tipo"] == "alvo"].copy()
    targets["item"] = targets["item"].astype(str)
    agree = []
    for _, group in targets.groupby("item"):
        labels = [human_quadrant(v, a) for v, a in zip(group["V"], group["A"])]
        agree.extend(
            labels[i] == labels[j] for i in range(len(labels)) for j in range(i + 1, len(labels))
        )
    return float(np.mean(agree))
