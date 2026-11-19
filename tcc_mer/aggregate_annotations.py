"""Agregação das anotações humanas do gold set.

Consome o export dos anotadores (aba ``respostas`` do Excel, uma linha por
resposta) e consolida por ``(faixa, modalidade)`` na **média** das respostas, mais
o número de anotadores da célula. O gold é a referência humana contra a qual os
rótulos de cada canal e a dissociação estimada são verificados.

Modalidade -> canal:

* ``melodia``  -> ``audio`` (áudio sem a voz);
* ``letra``    -> ``letra``;
* ``completa`` -> ``completa`` (referência bimodal; não entra em ``D``).

A escala coletada é o SAM 1-9, convertida para [-1, 1] na borda por
:func:`tcc_mer.scale.to_unit`. As respostas "não conheço a música" são mantidas
por padrão, porque a emoção percebida não exige conhecer a faixa.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .agreement import krippendorff_alpha
from .scale import to_unit

__all__ = [
    "MODALIDADE_TO_CANAL",
    "SAM_LO",
    "SAM_HI",
    "REQUIRED_COLUMNS",
    "load_annotations_xlsx",
    "to_gold_wide",
    "agreement_table",
]

SAM_LO, SAM_HI = 1.0, 9.0
MODALIDADE_TO_CANAL = {"melodia": "audio", "letra": "letra", "completa": "completa"}
REQUIRED_COLUMNS = ["pid", "modalidade", "trackId", "valencia", "arousal"]


def load_annotations_xlsx(path: str | Path, sheet: str = "respostas") -> pd.DataFrame:
    """Lê a aba de respostas do Excel e valida o schema mínimo.

    O ``pid`` é normalizado (sem espaços, minúsculo): o mesmo anotador aparece com
    grafias diferentes no export (``Pai`` e ``pai``) e não deve contar duas vezes.

    Returns:
        DataFrame longo com as colunas obrigatórias (e ``naoConheco``, criada com 0
        se ausente).

    Raises:
        ValueError: se faltar coluna obrigatória ou houver modalidade desconhecida.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"colunas obrigatórias ausentes em {path}: {missing}")
    unknown = set(df["modalidade"].dropna().unique()) - set(MODALIDADE_TO_CANAL)
    if unknown:
        raise ValueError(f"modalidade(s) desconhecida(s): {unknown}")
    df = df.assign(pid=df["pid"].astype(str).str.strip().str.lower())
    if "naoConheco" not in df.columns:
        df = df.assign(naoConheco=0)
    return df


def _track_by_annotator_matrix(df: pd.DataFrame, modalidade: str, dimension: str) -> np.ndarray:
    """Matriz faixa × anotador de uma dimensão numa modalidade (SAM 1-9).

    Respostas repetidas do mesmo anotador na mesma faixa são promediadas.
    """
    subset = df[df["modalidade"] == modalidade]
    return subset.pivot_table(
        index="trackId", columns="pid", values=dimension, aggfunc="mean"
    ).values


def to_gold_wide(df: pd.DataFrame, *, drop_naoconheco: bool = False) -> pd.DataFrame:
    """Consolida as respostas num gold largo, uma linha por faixa.

    A média é tirada primeiro por anotador (resolve teste-reteste) e depois entre
    anotadores.

    Args:
        df: DataFrame longo de :func:`load_annotations_xlsx`.
        drop_naoconheco: se ``True``, descarta as respostas com ``naoConheco == 1``.

    Returns:
        DataFrame indexado por ``trackId`` com ``V_<canal>``, ``A_<canal>`` (em
        [-1, 1]) e ``N_<canal>`` (anotadores distintos) para cada canal. Células sem
        resposta ficam ``NaN`` e ``N = 0``.
    """
    if drop_naoconheco:
        df = df[df["naoConheco"] != 1]

    per_annotator = df.groupby(["trackId", "modalidade", "pid"], as_index=False)[
        ["valencia", "arousal"]
    ].mean()

    tracks = sorted(df["trackId"].dropna().unique())
    gold = pd.DataFrame(index=pd.Index(tracks, name="trackId"))
    for modalidade, channel in MODALIDADE_TO_CANAL.items():
        cell = (
            per_annotator[per_annotator["modalidade"] == modalidade]
            .groupby("trackId")
            .agg(v=("valencia", "mean"), a=("arousal", "mean"), n=("pid", "nunique"))
        )
        gold[f"V_{channel}"] = to_unit(cell["v"].reindex(tracks), SAM_LO, SAM_HI)
        gold[f"A_{channel}"] = to_unit(cell["a"].reindex(tracks), SAM_LO, SAM_HI)
        gold[f"N_{channel}"] = cell["n"].reindex(tracks).fillna(0).astype(int)
    return gold


def agreement_table(df: pd.DataFrame, *, drop_naoconheco: bool = False) -> pd.DataFrame:
    """Alfa de Krippendorff intervalar por modalidade e dimensão.

    Espera-se alfa alto em áudio-arousal e letra-valência, e baixo nas células
    cruzadas, o que é o fenômeno da dissociação e não um defeito.

    Returns:
        DataFrame com ``modalidade, canal, dimensao, alpha, n_faixas, n_anotadores``.
    """
    if drop_naoconheco:
        df = df[df["naoConheco"] != 1]
    rows = []
    for modalidade, channel in MODALIDADE_TO_CANAL.items():
        for dimension in ("valencia", "arousal"):
            matrix = _track_by_annotator_matrix(df, modalidade, dimension)
            has_data = matrix.ndim == 2 and matrix.size > 0
            rows.append(
                {
                    "modalidade": modalidade,
                    "canal": channel,
                    "dimensao": dimension,
                    "alpha": krippendorff_alpha(matrix) if has_data else float("nan"),
                    "n_faixas": matrix.shape[0] if matrix.ndim == 2 else 0,
                    "n_anotadores": matrix.shape[1] if matrix.ndim == 2 else 0,
                }
            )
    return pd.DataFrame(rows)
