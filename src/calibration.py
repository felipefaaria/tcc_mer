"""Calibração afim das LFs — ancorar a saída bruta à escala humana (gold).

Cada LF tem viés sistemático próprio (ex.: o viés de arousal-mais-baixo do
português). Aprendemos **no gold set** uma correção linear (afim) que mapeia a
saída bruta da LF para a escala humana:

    y_hat = a · lf_raw + b

- ``a`` corrige escala/compressão; ``b`` corrige o deslocamento (viés).
- Ajustada de forma **independente por LF e por dimensão** (valência e arousal
  têm vieses distintos e não devem compartilhar parâmetros).
- Ajuste por mínimos quadrados **só no gold**. Com gold pequeno, avalie sempre
  fora-da-amostra (k-fold) — ver :mod:`src.fusion`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AffineCalibration"]


@dataclass
class AffineCalibration:
    """Correção afim ``y_hat = a·y + b``, ajustada por mínimos quadrados no gold.

    Ajustada de forma independente por função de rotulagem e por dimensão
    (valência e arousal têm vieses distintos e não devem compartilhar parâmetros).
    """

    a: float
    b: float

    @classmethod
    def fit(cls, lf_raw: np.ndarray, gold: np.ndarray) -> "AffineCalibration":
        """Ajusta ``(a, b)`` minimizando ``||a·lf_raw + b - gold||²``.

        Pares com ``NaN`` (em ``lf_raw`` ou ``gold``) são descartados antes do
        ajuste — faixas sem rótulo humano ou sem saída da LF não contribuem.

        Args:
            lf_raw: saídas brutas da LF nas faixas do gold set, shape ``(n,)``.
            gold: rótulos humanos correspondentes, shape ``(n,)``.

        Returns:
            Calibração afim ajustada.

        Raises:
            ValueError: se restarem menos de 2 pares válidos (ajuste indefinido).
        """
        lf_raw = np.asarray(lf_raw, dtype=float).ravel()
        gold = np.asarray(gold, dtype=float).ravel()
        if lf_raw.shape != gold.shape:
            raise ValueError(f"shapes incompatíveis: {lf_raw.shape} vs {gold.shape}")
        mask = ~(np.isnan(lf_raw) | np.isnan(gold))
        x, y = lf_raw[mask], gold[mask]
        if x.size < 2:
            raise ValueError("mínimo de 2 pares válidos para ajustar a calibração")
        a, b = np.polyfit(x, y, deg=1)
        return cls(a=float(a), b=float(b))

    def apply(self, lf_raw: np.ndarray) -> np.ndarray:
        """Aplica a correção às saídas brutas da LF (``NaN`` propaga)."""
        lf_raw = np.asarray(lf_raw, dtype=float)
        return self.a * lf_raw + self.b
