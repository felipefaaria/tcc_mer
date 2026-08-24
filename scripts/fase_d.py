#!/usr/bin/env python
"""Fase D — calibra as LFs contra o gold, funde, e mede a dissociação.

Lê ``gold_wide.csv`` + ``lf{1,2,3}_raw.csv`` e reporta:
- **CCC fora-da-amostra** (k-fold aninhado: calibra+funde só no treino) por canal e
  dimensão — a métrica primária.
- Pesos de fusão (canal letra) por dimensão.
- **Dissociação calibrada** ``D = letra - áudio`` vs. ``D`` do gold; maiores
  dissociadores.

Mapeamento: LF1↔áudio(melodia), LF2/LF3↔letra. O canal áudio→valência tem gold
ruidoso (Krippendorff ~0.12) — CCC baixo ali é ESPERADO, não é bug.

    python scripts/fase_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import ccc  # noqa: E402
from src.pipeline import apply_channel, fit_channel, nested_kfold_ccc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "annotations" / "gold_wide.csv"
LFDIR = ROOT / "data" / "lf_outputs"
K = 5
SEED = 0


def carregar() -> pd.DataFrame:
    gold = pd.read_csv(GOLD).set_index("trackId")
    for lf in ("lf1", "lf2", "lf3"):
        d = pd.read_csv(LFDIR / f"{lf}_raw.csv").set_index("id")
        gold[f"{lf}_V"] = d["V_raw"]
        gold[f"{lf}_A"] = d["A_raw"]
    return gold


def oos(df, cols, ycol):
    X = df[cols].to_numpy(float)
    y = df[ycol].to_numpy(float)
    return nested_kfold_ccc(X, y, k=K, seed=SEED)


def main() -> None:
    if not GOLD.exists():
        sys.exit(f"[faseD] gold não encontrado: {GOLD} (rode scripts/gerar_gold.py)")
    df = carregar()

    print("=" * 64)
    print("CCC FORA-DA-AMOSTRA (k-fold aninhado; calibra+funde só no treino)")
    print("=" * 64)
    print(f"{'canal/modelo':28s} {'dim':4s} {'CCC_oos':>8s} {'n':>4s}")
    plan = [
        ("LF1 áudio",            ["lf1_V"], "V_audio", "V"),
        ("LF1 áudio",            ["lf1_A"], "A_audio", "A"),
        ("LF2 léxico (letra)",   ["lf2_V"], "V_letra", "V"),
        ("LF2 léxico (letra)",   ["lf2_A"], "A_letra", "A"),
        ("LF3 LLM (letra)",      ["lf3_V"], "V_letra", "V"),
        ("LF3 LLM (letra)",      ["lf3_A"], "A_letra", "A"),
        ("FUSÃO letra (LF2+LF3)", ["lf2_V", "lf3_V"], "V_letra", "V"),
        ("FUSÃO letra (LF2+LF3)", ["lf2_A", "lf3_A"], "A_letra", "A"),
        ("FUSÃO completa (LF1+2+3)", ["lf1_V", "lf2_V", "lf3_V"], "V_completa", "V"),
        ("FUSÃO completa (LF1+2+3)", ["lf1_A", "lf2_A", "lf3_A"], "A_completa", "A"),
    ]
    for nome, cols, ycol, dim in plan:
        c, w, n = oos(df, cols, ycol)
        wtxt = ""
        if len(cols) > 1 and not np.isnan(w).any():
            wtxt = "  pesos=" + "/".join(f"{x:.2f}" for x in w)
        print(f"{nome:28s} {dim:4s} {c:>8.2f} {n:>4d}{wtxt}")

    # ----- Dissociação calibrada -----
    print("\n" + "=" * 64)
    print("DISSOCIAÇÃO CALIBRADA  D = letra - áudio")
    print("=" * 64)
    est = {}
    for dim in ("V", "A"):
        cal_a, w_a = fit_channel(df[[f"lf1_{dim}"]].to_numpy(float),
                                 df[f"{dim}_audio"].to_numpy(float))
        cal_l, w_l = fit_channel(df[[f"lf2_{dim}", f"lf3_{dim}"]].to_numpy(float),
                                 df[f"{dim}_letra"].to_numpy(float))
        est[f"{dim}_audio_hat"] = apply_channel(cal_a, w_a, df[[f"lf1_{dim}"]].to_numpy(float))
        est[f"{dim}_letra_hat"] = apply_channel(cal_l, w_l,
                                                df[[f"lf2_{dim}", f"lf3_{dim}"]].to_numpy(float))
    E = pd.DataFrame(est, index=df.index)
    E["Dv_hat"] = E["V_letra_hat"] - E["V_audio_hat"]
    E["Da_hat"] = E["A_letra_hat"] - E["A_audio_hat"]
    E["Dv_gold"] = df["V_letra"] - df["V_audio"]
    E["Da_gold"] = df["A_letra"] - df["A_audio"]

    for dim, hat, gold in [("Valência", "Dv_hat", "Dv_gold"), ("Arousal", "Da_hat", "Da_gold")]:
        v = E.dropna(subset=[hat, gold])
        c = ccc(v[hat].to_numpy(), v[gold].to_numpy()) if len(v) >= 2 else float("nan")
        print(f"  D {dim}: CCC(D_hat, D_gold) = {c:+.2f}  (n={len(v)})")

    print("\n  Maiores dissociadores por D_V calibrado (letra - áudio):")
    top = E.dropna(subset=["Dv_hat"]).sort_values("Dv_hat")
    print("   letra«áudio (áudio alegre + letra triste):")
    for tid, r in top.head(5).iterrows():
        print(f"     {tid:7s} letra={r.V_letra_hat:+.2f} audio={r.V_audio_hat:+.2f} Dv={r.Dv_hat:+.2f}")
    print("   letra»áudio (letra positiva + áudio contido):")
    for tid, r in top.tail(5).iterrows():
        print(f"     {tid:7s} letra={r.V_letra_hat:+.2f} audio={r.V_audio_hat:+.2f} Dv={r.Dv_hat:+.2f}")

    out = LFDIR / "dissociation.csv"
    E.round(4).to_csv(out)
    print(f"\n[faseD] estimativas calibradas + D -> {out}")


if __name__ == "__main__":
    main()
