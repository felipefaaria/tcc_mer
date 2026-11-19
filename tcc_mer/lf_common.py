"""Contrato comum das funções de rotulagem (LF).

Cada LF produz um ``(V, A)`` por faixa, já em [-1, 1] e **sem calibração**, e o
persiste em ``data/outputs/<lf>_raw.csv`` com as colunas ``id, V_raw, A_raw``.
A dependência pesada (modelo, API) entra por injeção (um ``predictor`` ou
``client``) e nunca é importada no topo dos módulos, o que mantém a lógica
testável sem o serviço externo.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = ["LFRawOutput", "write_lf_raw", "read_lf_raw"]

RAW_COLUMNS = ("id", "V_raw", "A_raw")


@dataclass
class LFRawOutput:
    """Saída bruta de uma LF para uma faixa, em [-1, 1].

    ``v_raw`` e ``a_raw`` são ``NaN`` quando a LF não se aplica (faixa instrumental
    numa LF de letra) ou falhou. ``meta`` guarda diagnóstico livre (cobertura,
    moods, modelo) e não é gravado no csv.
    """

    track_id: str
    v_raw: float
    a_raw: float
    meta: dict = field(default_factory=dict)


def write_lf_raw(outputs: list[LFRawOutput], path: str | Path) -> pd.DataFrame:
    """Grava ``id, V_raw, A_raw`` em ``path`` (criando a pasta) e devolve o DataFrame."""
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
    """Lê um ``<lf>_raw.csv`` e valida as colunas ``id, V_raw, A_raw``."""
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"colunas ausentes em {path}: {missing}")
    return df
