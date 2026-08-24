#!/usr/bin/env python
"""Runner do gold — consolida as anotações humanas em gold_wide + concordância.

Lê o Excel dos anotadores (aba ``respostas``), gera:
- ``data/annotations/gold_wide.csv``  (por faixa: V/A médios em [-1,1] + N, por canal)
- ``data/annotations/agreement.csv``  (Krippendorff por modalidade/dimensão)

Ambos ficam fora do git (PII/derivado). Exemplos::

    python scripts/gerar_gold.py
    python scripts/gerar_gold.py --xlsx data/annotations/Anotadores.xlsx --drop-naoconheco
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregate_annotations import (  # noqa: E402
    agreement_table,
    load_annotations_xlsx,
    to_gold_wide,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Gera o gold set a partir do Excel.")
    p.add_argument("--xlsx", type=Path,
                   default=ROOT / "data" / "annotations" / "Anotadores.xlsx")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "annotations")
    p.add_argument("--drop-naoconheco", action="store_true",
                   help="descarta respostas 'não conheço a música'.")
    args = p.parse_args()

    if not args.xlsx.exists():
        sys.exit(f"[gold] Excel não encontrado: {args.xlsx}")

    df = load_annotations_xlsx(args.xlsx)
    print(f"[gold] {len(df)} respostas, {df['pid'].nunique()} anotadores, "
          f"{df['trackId'].nunique()} faixas.")

    gold = to_gold_wide(df, drop_naoconheco=args.drop_naoconheco)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = args.out_dir / "gold_wide.csv"
    gold.to_csv(gold_path)

    agr = agreement_table(df, drop_naoconheco=args.drop_naoconheco)
    agr_path = args.out_dir / "agreement.csv"
    agr.to_csv(agr_path, index=False)

    print(f"[gold] gold_wide -> {gold_path}")
    for canal in ("audio", "letra", "completa"):
        col = f"N_{canal}"
        cov = int((gold[col] > 0).sum())
        print(f"    {canal:9s}: {cov} faixas com anotação "
              f"(N médio {gold.loc[gold[col] > 0, col].mean():.1f})")
    print(f"[gold] concordância -> {agr_path}")
    print(agr.to_string(index=False))


if __name__ == "__main__":
    main()
