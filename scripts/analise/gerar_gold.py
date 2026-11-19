#!/usr/bin/env python
"""Consolida as anotações humanas do gold set em ``gold_wide.csv`` e na tabela de concordância.

Lê o Excel dos anotadores (aba ``respostas``) e grava, em ``data/annotations``:

* ``gold_wide.csv``: por faixa, valência e arousal médios em [-1, 1] e o N de anotadores,
  por canal (``audio`` = áudio sem a voz, ``letra``, ``completa``);
* ``agreement.csv``: alfa de Krippendorff por modalidade e dimensão.

As anotações trazem dados pessoais dos anotadores e não são versionadas.

    python scripts/analise/gerar_gold.py
    python scripts/analise/gerar_gold.py --xlsx data/annotations/Anotadores.xlsx --drop-naoconheco
"""

import argparse
import sys
from pathlib import Path

from tcc_mer.aggregate_annotations import agreement_table, load_annotations_xlsx, to_gold_wide
from tcc_mer.paths import ANNOTATIONS_DIR

CHANNELS = ("audio", "letra", "completa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xlsx", type=Path, default=ANNOTATIONS_DIR / "Anotadores.xlsx")
    parser.add_argument("--out-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument(
        "--drop-naoconheco",
        action="store_true",
        help="descarta as respostas 'não conheço a música'",
    )
    args = parser.parse_args()
    if not args.xlsx.exists():
        sys.exit(f"[gold] Excel não encontrado: {args.xlsx}")

    responses = load_annotations_xlsx(args.xlsx)
    print(
        f"[gold] {len(responses)} respostas, {responses['pid'].nunique()} anotadores, "
        f"{responses['trackId'].nunique()} faixas."
    )

    gold = to_gold_wide(responses, drop_naoconheco=args.drop_naoconheco)
    agreement = agreement_table(responses, drop_naoconheco=args.drop_naoconheco)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = args.out_dir / "gold_wide.csv"
    agreement_path = args.out_dir / "agreement.csv"
    gold.to_csv(gold_path)
    agreement.to_csv(agreement_path, index=False)

    print(f"[gold] gold_wide -> {gold_path}")
    for channel in CHANNELS:
        n_annotators = gold[f"N_{channel}"]
        covered = n_annotators > 0
        print(
            f"    {channel:9s}: {covered.sum()} faixas com anotação "
            f"(N médio {n_annotators[covered].mean():.1f})"
        )
    print(f"[gold] concordância -> {agreement_path}")
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
