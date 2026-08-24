"""Agregação das anotações humanas → gold set (Fase C).

Consome o export dos anotadores (aba ``respostas`` do Excel, formato longo: uma
linha por resposta) e consolida por ``(trackId, modalidade)`` na **média** das
respostas + o **N de anotadores** por célula. É o gold contra o qual as LFs são
calibradas (Fase D).

Mapeamento de modalidade → canal:
- ``melodia``  → **audio**   (calibra a LF1)
- ``letra``    → **letra**   (calibra a LF2/LF3)
- ``completa`` → **completa** (referência bimodal; não entra em D)

A escala coletada é SAM 1-9; converte-se para [-1, 1] na borda (:func:`src.scale.to_unit`),
nunca antes. Respostas ``naoConheco`` são mantidas por padrão (emoção *percebida*
não exige conhecer a música), mas há flag para excluí-las e checar sensibilidade.

Regra metodológica: nunca combinar dimensões nem modalidades ao agregar; a média é
por célula (faixa × modalidade × dimensão), preservando o N.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .agreement import krippendorff_alpha
from .scale import to_unit

__all__ = [
    "MODALIDADE_TO_CANAL",
    "SAM_LO",
    "SAM_HI",
    "COLS_OBRIGATORIAS",
    "load_annotations_xlsx",
    "to_gold_wide",
    "agreement_table",
]

SAM_LO, SAM_HI = 1.0, 9.0
MODALIDADE_TO_CANAL = {"melodia": "audio", "letra": "letra", "completa": "completa"}
COLS_OBRIGATORIAS = ["pid", "modalidade", "trackId", "valencia", "arousal"]


def load_annotations_xlsx(
    path: str | Path, sheet: str = "respostas"
) -> pd.DataFrame:
    """Lê a aba de respostas do Excel e valida o schema mínimo.

    Args:
        path: caminho do .xlsx exportado do formulário.
        sheet: nome da aba de respostas.

    Returns:
        DataFrame longo com as colunas obrigatórias (+ ``naoConheco`` se houver).

    Raises:
        ValueError: se faltar alguma coluna obrigatória ou houver modalidade
            desconhecida (não em :data:`MODALIDADE_TO_CANAL`).
    """
    df = pd.read_excel(path, sheet_name=sheet)
    faltando = [c for c in COLS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"colunas obrigatórias ausentes em {path}: {faltando}")
    desconhecidas = set(df["modalidade"].dropna().unique()) - set(MODALIDADE_TO_CANAL)
    if desconhecidas:
        raise ValueError(f"modalidade(s) desconhecida(s): {desconhecidas}")
    if "naoConheco" not in df.columns:
        df = df.assign(naoConheco=0)
    return df


def _matriz_faixa_anotador(
    df: pd.DataFrame, modalidade: str, dim: str
) -> np.ndarray:
    """Matriz (faixa × anotador) de uma dimensão numa modalidade (SAM 1-9).

    Células duplicadas (teste-reteste: mesmo pid/faixa) são promediadas.
    """
    g = df[df["modalidade"] == modalidade]
    mat = g.pivot_table(index="trackId", columns="pid", values=dim, aggfunc="mean")
    return mat.values


def to_gold_wide(df: pd.DataFrame, *, drop_naoconheco: bool = False) -> pd.DataFrame:
    """Consolida as respostas em gold wide por faixa.

    Para cada faixa e canal (audio/letra/completa) calcula a média de valência e
    arousal (em [-1, 1]) e o N de anotadores distintos daquela célula.

    Args:
        df: DataFrame longo de :func:`load_annotations_xlsx`.
        drop_naoconheco: se ``True``, descarta respostas com ``naoConheco == 1``.

    Returns:
        DataFrame indexado por ``trackId`` com colunas ``V_<canal>``, ``A_<canal>``
        (em [-1, 1]) e ``N_<canal>`` para cada canal. Células sem resposta ficam
        ``NaN`` (N = 0). As colunas ``V_audio/V_letra/A_audio/A_letra`` alimentam
        :func:`src.dissociation.compute_dissociation` diretamente.
    """
    if drop_naoconheco:
        df = df[df["naoConheco"] != 1]

    # média por (faixa, modalidade, anotador) resolve teste-reteste antes de agregar
    por_anotador = (
        df.groupby(["trackId", "modalidade", "pid"], as_index=False)[
            ["valencia", "arousal"]
        ].mean()
    )

    faixas = sorted(df["trackId"].dropna().unique())
    out = pd.DataFrame(index=pd.Index(faixas, name="trackId"))
    for modalidade, canal in MODALIDADE_TO_CANAL.items():
        sub = por_anotador[por_anotador["modalidade"] == modalidade]
        agg = sub.groupby("trackId").agg(
            v=("valencia", "mean"), a=("arousal", "mean"), n=("pid", "nunique")
        )
        out[f"V_{canal}"] = to_unit(agg["v"].reindex(faixas), SAM_LO, SAM_HI)
        out[f"A_{canal}"] = to_unit(agg["a"].reindex(faixas), SAM_LO, SAM_HI)
        out[f"N_{canal}"] = agg["n"].reindex(faixas).fillna(0).astype(int)
    return out


def agreement_table(df: pd.DataFrame, *, drop_naoconheco: bool = False) -> pd.DataFrame:
    """Krippendorff alpha (intervalar) por modalidade e dimensão.

    A tabela de concordância é diagnóstico central do gold: espera-se alpha alto em
    audio→arousal e letra→valência, e baixo nas células cruzadas (o fenômeno da
    dissociação, não um defeito).

    Returns:
        DataFrame com colunas ``modalidade, canal, dimensao, alpha, n_faixas,
        n_anotadores``.
    """
    if drop_naoconheco:
        df = df[df["naoConheco"] != 1]
    linhas = []
    for modalidade, canal in MODALIDADE_TO_CANAL.items():
        for dim in ("valencia", "arousal"):
            mat = _matriz_faixa_anotador(df, modalidade, dim)
            linhas.append(
                {
                    "modalidade": modalidade,
                    "canal": canal,
                    "dimensao": dim,
                    "alpha": krippendorff_alpha(mat) if mat.size else float("nan"),
                    "n_faixas": mat.shape[0] if mat.ndim == 2 else 0,
                    "n_anotadores": mat.shape[1] if mat.ndim == 2 else 0,
                }
            )
    return pd.DataFrame(linhas)
