#!/usr/bin/env python
"""Gera ``dataset/corpus_anotado.csv``, a tabela do corpus anotado.

Junta os manifestos dos três grupos do corpus (históricas, gold set e expansão) com as
saídas brutas da LF1 (áudio, Music2Emo) e da LF2 (letra, LLM), e deriva as fusões, a
dissociação e os rótulos de emoção. Grava também ``dataset/proveniencia.json``, com o
hash das entradas e o modelo e a versão do prompt da LF2.

    python scripts/gerar_dataset.py
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from tcc_mer.dataset import DATASET_PATH, build_annotated_corpus, load_metadata
from tcc_mer.lf_common import read_lf_raw
from tcc_mer.paths import DATASET_DIR, OUTPUTS_DIR, VALIDATION_DIR


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lf1", type=Path, default=OUTPUTS_DIR / "lf1_raw.csv")
    parser.add_argument("--lf2", type=Path, default=OUTPUTS_DIR / "lf2_raw.csv")
    parser.add_argument("--item-map", type=Path, default=VALIDATION_DIR / "mapa_itens.csv")
    parser.add_argument("--out", type=Path, default=DATASET_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.lf1, args.lf2):
        if not path.exists():
            sys.exit(f"[dataset] saída de LF não encontrada: {path}")

    metadata = load_metadata()
    lf1, lf2 = read_lf_raw(args.lf1), read_lf_raw(args.lf2)

    item_map = pd.read_csv(args.item_map, encoding="utf-8-sig", keep_default_na=False)
    experiment_ids = item_map.loc[item_map["tipo"] == "alvo", "trackId"]

    table = build_annotated_corpus(metadata, lf1, lf2, experiment_ids)

    instrumentals = set(metadata.loc[metadata["instrumental"], "id"])
    rated = lf2.dropna(subset=["V_raw"])
    discarded = sorted(instrumentals & set(rated["id"]))
    if discarded:
        print(
            f"[dataset] {len(discarded)} instrumental(is) com nota de letra descartada(s): "
            f"{', '.join(discarded)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    provenance = {
        "n_faixas": len(table),
        "n_por_grupo": table["grupo"].value_counts().to_dict(),
        "n_instrumentais": int(table["instrumental"].sum()),
        "n_com_os_dois_canais": int(table["fusao_simples_valencia"].notna().sum()),
        "entradas_sha256": {path.name: sha256(path) for path in (args.lf1, args.lf2)},
        "lf1": read_json(OUTPUTS_DIR / "lf1_meta.json"),
        "lf2": read_json(OUTPUTS_DIR / "lf2_meta.json"),
    }
    provenance_path = DATASET_DIR / "proveniencia.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[dataset] {args.out}  ({len(table)} faixas, {provenance['n_com_os_dois_canais']} "
        f"com os dois canais)"
    )
    print(f"[dataset] {provenance_path}")


if __name__ == "__main__":
    main()
