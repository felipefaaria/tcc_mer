"""Calibração afim de uma LF contra o gold set.

Ajusta ``y_hat = a·lf_raw + b`` por mínimos quadrados: ``a`` corrige a compressão de
escala e ``b``, o deslocamento sistemático. O ajuste é independente por LF e por
dimensão. Com um gold pequeno, avalie sempre fora da amostra (ver
:mod:`tcc_mer.exploratory.pipeline`).
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["AffineCalibration"]


@dataclass
class AffineCalibration:
    """Correção afim ``y_hat = a·y + b``."""

    a: float
    b: float

    @classmethod
    def fit(cls, lf_raw: np.ndarray, gold: np.ndarray) -> "AffineCalibration":
        """Ajusta ``(a, b)`` minimizando ``||a·lf_raw + b - gold||²``.

        Pares com ``NaN`` em ``lf_raw`` ou em ``gold`` são descartados.

        Raises:
            ValueError: com shapes diferentes ou menos de 2 pares válidos.
        """
        lf_raw = np.asarray(lf_raw, dtype=float).ravel()
        gold = np.asarray(gold, dtype=float).ravel()
        if lf_raw.shape != gold.shape:
            raise ValueError(f"shapes incompatíveis: {lf_raw.shape} vs {gold.shape}")
        valid = ~(np.isnan(lf_raw) | np.isnan(gold))
        if valid.sum() < 2:
            raise ValueError("mínimo de 2 pares válidos para ajustar a calibração")
        a, b = np.polyfit(lf_raw[valid], gold[valid], deg=1)
        return cls(a=float(a), b=float(b))

    def apply(self, lf_raw: np.ndarray) -> np.ndarray:
        """Aplica a correção às saídas brutas (``NaN`` propaga)."""
        return self.a * np.asarray(lf_raw, dtype=float) + self.b
