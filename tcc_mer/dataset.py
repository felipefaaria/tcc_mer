"""Tabela do corpus anotado: metadados, rótulos por canal, fusões, dissociação e emoção.

A tabela junta, por faixa, os metadados dos três grupos do corpus (históricas, gold
set e expansão) com as saídas brutas da LF1 (áudio) e da LF2 (letra, LLM) e deriva
o resto delas. Não há calibração: todos os valores estão em [-1, 1] e vêm direto
dos modelos.

A faixa marcada como instrumental no manifesto recebe ``NaN`` em tudo que depende
da letra, mesmo que o arquivo de letra exista (algumas faixas instrumentais têm um
texto de preenchimento e o LLM chegou a atribuir nota a ele).

Dicionário das colunas: ``dataset/README.md``.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .dissociation import compute_dissociation, magnitude
from .fusion import dimension_fusion, simple_fusion
from .paths import CORPUS_DIR, DATASET_DIR
from .quadrant import emotion_of, margin, quadrants

__all__ = [
    "GROUP_HISTORICAL",
    "GROUP_GOLD",
    "GROUP_EXPANSION",
    "DATASET_COLUMNS",
    "DATASET_PATH",
    "load_metadata",
    "build_annotated_corpus",
    "read_dataset",
]

GROUP_HISTORICAL = "historica"
GROUP_GOLD = "gold"
GROUP_EXPANSION = "expansao"

HISTORICAL_MANIFEST = CORPUS_DIR / "manifest_historico.csv"
GOLD_MANIFEST = CORPUS_DIR / "manifest_gold.csv"
EXPANSION_MANIFEST = CORPUS_DIR / "manifest_expansao.csv"
DATASET_PATH = DATASET_DIR / "corpus_anotado.csv"

METADATA_COLUMNS = [
    "id",
    "grupo",
    "ano",
    "ano_aproximado",
    "titulo",
    "interprete",
    "genero",
    "instrumental",
]

DATASET_COLUMNS = [
    *METADATA_COLUMNS,
    "lf1_audio_valencia",
    "lf1_audio_arousal",
    "lf2_llm_valencia",
    "lf2_llm_arousal",
    "fusao_simples_valencia",
    "fusao_simples_arousal",
    "fusao_dimensao_valencia",
    "fusao_dimensao_arousal",
    "dissociacao_valencia",
    "dissociacao_arousal",
    "dissociacao_magnitude",
    "dissociacao_quadrante",
    "quadrante_lf1",
    "quadrante_lf2",
    "quadrante_fusao_simples",
    "emocao_fusao_simples",
    "margem_fusao_simples",
    "quadrante_fusao_dimensao",
    "emocao_fusao_dimensao",
    "margem_fusao_dimensao",
    "no_experimento",
]

TEXT_COLUMNS = [
    "quadrante_lf1",
    "quadrante_lf2",
    "quadrante_fusao_simples",
    "emocao_fusao_simples",
    "quadrante_fusao_dimensao",
    "emocao_fusao_dimensao",
]

DECIMALS = 4


def _first_year(value) -> float:
    """Primeiro número de quatro dígitos de ``value`` (``NaN`` se não houver)."""
    match = re.search(r"\d{4}", str(value))
    return float(match.group()) if match else np.nan


def load_metadata(
    historical: Path = HISTORICAL_MANIFEST,
    gold: Path = GOLD_MANIFEST,
    expansion: Path = EXPANSION_MANIFEST,
) -> pd.DataFrame:
    """Une os manifestos dos três grupos num esquema único (:data:`METADATA_COLUMNS`).

    No gold set e na expansão o ano é aproximado (``ano_aproximado=True``) e o gênero é o
    macrogênero; nas históricas, o gênero é o do manifesto (específico e muitas vezes vazio).
    """
    historical_df = pd.read_csv(historical)
    gold_df = pd.read_csv(gold)
    expansion_df = pd.read_csv(expansion)

    frames = [
        pd.DataFrame(
            {
                "id": historical_df["id"],
                "grupo": GROUP_HISTORICAL,
                "ano": historical_df["ano"].map(_first_year),
                "ano_aproximado": False,
                "titulo": historical_df["faixa"],
                "interprete": historical_df["interprete"],
                "genero": historical_df["genero"],
                "instrumental": historical_df["instrumental"] == "sim",
            }
        ),
        pd.DataFrame(
            {
                "id": gold_df["id"],
                "grupo": GROUP_GOLD,
                "ano": gold_df["ano_aprox"].map(_first_year),
                "ano_aproximado": True,
                "titulo": gold_df["titulo"],
                "interprete": gold_df["artista"],
                "genero": gold_df["genero_macro"],
                "instrumental": gold_df["tem_letra"] == "não",
            }
        ),
        pd.DataFrame(
            {
                "id": expansion_df["id"],
                "grupo": GROUP_EXPANSION,
                "ano": expansion_df["ano_aprox"].map(_first_year),
                "ano_aproximado": True,
                "titulo": expansion_df["faixa"],
                "interprete": expansion_df["interprete"],
                "genero": expansion_df["genero_macro_ou_ritmo"],
                "instrumental": expansion_df["instrumental"].astype(str).str.lower() == "true",
            }
        ),
    ]
    metadata = pd.concat(frames, ignore_index=True)
    metadata["ano"] = metadata["ano"].astype("Int64")
    return metadata[METADATA_COLUMNS]


def _channel(raw: pd.DataFrame, ids: pd.Index, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(V_raw, A_raw)`` de uma LF, na ordem de ``ids``; exige uma linha por faixa."""
    if raw["id"].duplicated().any():
        raise ValueError(f"ids duplicados na saída de {name}")
    absent = ids.difference(pd.Index(raw["id"]))
    if len(absent):
        raise ValueError(f"{len(absent)} faixa(s) sem linha em {name}: {list(absent[:5])}")
    aligned = raw.set_index("id").reindex(ids)
    return aligned["V_raw"].to_numpy(float), aligned["A_raw"].to_numpy(float)


def build_annotated_corpus(
    metadata: pd.DataFrame,
    lf1: pd.DataFrame,
    lf2: pd.DataFrame,
    experiment_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Monta a tabela do corpus anotado (:data:`DATASET_COLUMNS`), uma linha por faixa.

    Args:
        metadata: saída de :func:`load_metadata`.
        lf1: ``id, V_raw, A_raw`` do áudio (Music2Emo).
        lf2: ``id, V_raw, A_raw`` da letra (LLM).
        experiment_ids: faixas do experimento de validação com ouvintes.

    Raises:
        ValueError: se alguma faixa dos metadados não tiver linha numa das LFs.
    """
    if metadata["id"].duplicated().any():
        raise ValueError("ids duplicados nos metadados")
    ids = pd.Index(metadata["id"])
    instrumental = metadata["instrumental"].to_numpy(bool)

    v_audio, a_audio = _channel(lf1, ids, "LF1")
    v_lyrics, a_lyrics = _channel(lf2, ids, "LF2")
    v_lyrics = np.where(instrumental, np.nan, v_lyrics)
    a_lyrics = np.where(instrumental, np.nan, a_lyrics)

    channels = (v_audio, a_audio, v_lyrics, a_lyrics)
    v_simple, a_simple = simple_fusion(*channels)
    v_dimension, a_dimension = dimension_fusion(*channels)
    d_v, d_a = compute_dissociation(v_audio, v_lyrics, a_audio, a_lyrics)

    q_audio = quadrants(v_audio, a_audio)
    q_lyrics = quadrants(v_lyrics, a_lyrics)
    q_simple = quadrants(v_simple, a_simple)
    q_dimension = quadrants(v_dimension, a_dimension)
    both_channels = (q_audio != "") & (q_lyrics != "")
    quadrant_differs = pd.Series(
        np.where(both_channels, q_audio != q_lyrics, None), dtype="boolean"
    )

    table = metadata.reset_index(drop=True).copy()
    columns = {
        "lf1_audio_valencia": v_audio,
        "lf1_audio_arousal": a_audio,
        "lf2_llm_valencia": v_lyrics,
        "lf2_llm_arousal": a_lyrics,
        "fusao_simples_valencia": v_simple,
        "fusao_simples_arousal": a_simple,
        "fusao_dimensao_valencia": v_dimension,
        "fusao_dimensao_arousal": a_dimension,
        "dissociacao_valencia": d_v,
        "dissociacao_arousal": d_a,
        "dissociacao_magnitude": magnitude(d_v, d_a),
        "margem_fusao_simples": margin(v_simple, a_simple),
        "margem_fusao_dimensao": margin(v_dimension, a_dimension),
    }
    for name, values in columns.items():
        table[name] = np.round(values, DECIMALS)
    table["dissociacao_quadrante"] = quadrant_differs
    table["quadrante_lf1"] = q_audio
    table["quadrante_lf2"] = q_lyrics
    table["quadrante_fusao_simples"] = q_simple
    table["emocao_fusao_simples"] = [emotion_of(q) for q in q_simple]
    table["quadrante_fusao_dimensao"] = q_dimension
    table["emocao_fusao_dimensao"] = [emotion_of(q) for q in q_dimension]
    table["no_experimento"] = table["id"].isin(set(experiment_ids))
    return table[DATASET_COLUMNS]


def read_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Lê a tabela gravada, com quadrantes e emoções ausentes como ``""``."""
    table = pd.read_csv(path, dtype=dict.fromkeys(TEXT_COLUMNS, "string"))
    table[TEXT_COLUMNS] = table[TEXT_COLUMNS].fillna("").astype(str)
    table["dissociacao_quadrante"] = table["dissociacao_quadrante"].astype("boolean")
    return table
