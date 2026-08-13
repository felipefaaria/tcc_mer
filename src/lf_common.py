"""Utilitários compartilhados pelas Labeling Functions (Fase B).

Cada LF produz um (V, A) por faixa e persiste sua saída **bruta** (já na escala
interna [-1, 1] via :func:`src.scale.to_unit`, mas **antes** da calibração afim ao
gold) em ``data/lf_outputs/lfN_raw.csv`` com as colunas ``id, V_raw, A_raw``.

A regra de ouro da Fase B é o **isolamento da dependência externa**: a lógica de
escala/instrumental/persistência vive aqui e nos módulos de cada LF (testável sem
o modelo pesado); o modelo/serviço externo entra por injeção (um ``predictor`` ou
``client``), nunca importado no topo do módulo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["LFRawOutput", "write_lf_raw", "read_lf_raw"]


@dataclass
class LFRawOutput:
    """Saída bruta de uma LF para uma faixa, na escala interna [-1, 1].

    ``v_raw``/``a_raw`` são ``np.nan`` quando a LF não se aplica (ex.: faixa
    instrumental para LFs de letra) ou não teve cobertura suficiente. ``meta``
    carrega diagnóstico livre (cobertura, moods, modelo/versão) que NÃO entra no
    csv de rótulo, mas é útil para reprodutibilidade.
    """

    track_id: str
    v_raw: float
    a_raw: float
    meta: dict = field(default_factory=dict)


def write_lf_raw(outputs: list[LFRawOutput], path: str | Path) -> pd.DataFrame:
    """Persiste uma lista de :class:`LFRawOutput` em ``id, V_raw, A_raw``.

    Cria o diretório-pai se necessário. Retorna o DataFrame escrito.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "id": [o.track_id for o in outputs],
            "V_raw": [o.v_raw for o in outputs],
            "A_raw": [o.a_raw for o in outputs],
        }
    )
    df.to_csv(path, index=False)
    return df


def read_lf_raw(path: str | Path) -> pd.DataFrame:
    """Lê um csv ``lfN_raw.csv`` (colunas ``id, V_raw, A_raw``)."""
    df = pd.read_csv(path)
    expected = {"id", "V_raw", "A_raw"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"colunas ausentes em {path}: {missing}")
    return df
