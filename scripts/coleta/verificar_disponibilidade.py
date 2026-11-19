#!/usr/bin/env python
"""Verifica se as candidatas existem no Genius (letra) e no Deezer (prévia de áudio).

Lê um csv de candidatas (colunas ``ano``, ``faixa`` e ``compositor_interprete``) e só BUSCA: não
baixa nada. Serve para saber, antes de gastar esforço de extração, quantas candidatas de uma
fase são utilizáveis (músicas antigas muitas vezes não têm prévia digital). Grava um relatório
com uma linha por música: se achou em cada fonte e o título que casou, para conferência humana.

A chave do Genius vem de ``GENIUS_ACCESS_TOKEN``, do ambiente ou de um ``.env`` na raiz.

    python scripts/coleta/verificar_disponibilidade.py candidatas.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from tcc_mer.env import load_env
from tcc_mer.sources import NETWORK_ERRORS, deezer_hits, first_artist, genius_hits

SAVE_EVERY = 20


def check_genius(performer: str, title: str, token: str) -> tuple[bool, str]:
    """``(achou, "artista - título")``; em caso de erro, ``(False, "erro: ...")``."""
    try:
        hits = genius_hits(f"{first_artist(performer)} {title}", token)
    except NETWORK_ERRORS as error:
        return False, f"erro: {error}"
    if not hits:
        return False, ""
    top = hits[0]["result"]
    return True, f"{top.get('primary_artist', {}).get('name', '?')} - {top.get('title', '?')}"


def check_deezer(performer: str, title: str) -> tuple[bool, str]:
    """``(achou, "artista - título")``; em caso de erro, ``(False, "erro: ...")``."""
    try:
        hits = deezer_hits(f"{first_artist(performer)} {title}")
    except NETWORK_ERRORS as error:
        return False, f"erro: {error}"
    if not hits or not hits[0].get("preview"):
        return False, ""
    return True, f"{hits[0].get('artist', {}).get('name', '?')} - {hits[0].get('title', '?')}"


def write_report(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_in", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    out_path = args.out or args.csv_in.with_name(args.csv_in.stem + "_disponibilidade.csv")

    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        sys.exit("[verificar] defina GENIUS_ACCESS_TOKEN no .env")
    with args.csv_in.open(encoding="utf-8") as file:
        candidates = list(csv.DictReader(file))
    if args.limit:
        candidates = candidates[: args.limit]

    fields = [*candidates[0].keys(), "genius_achou", "genius_match", "deezer_achou", "deezer_match"]
    report = []
    for position, row in enumerate(candidates, start=1):
        performer, title = row["compositor_interprete"], row["faixa"]
        genius_ok, genius_match = check_genius(performer, title, token)
        time.sleep(args.sleep)
        deezer_ok, deezer_match = check_deezer(performer, title)
        time.sleep(args.sleep)
        report.append(
            {
                **row,
                "genius_achou": genius_ok,
                "genius_match": genius_match,
                "deezer_achou": deezer_ok,
                "deezer_match": deezer_match,
            }
        )
        print(
            f"[{position}/{len(candidates)}] {row['ano']} {title[:35]:35s} "
            f"genius={'OK' if genius_ok else '--'} deezer={'OK' if deezer_ok else '--'}"
        )
        if position % SAVE_EVERY == 0:
            write_report(out_path, fields, report)

    write_report(out_path, fields, report)
    n_genius = sum(r["genius_achou"] for r in report)
    n_deezer = sum(r["deezer_achou"] for r in report)
    n_both = sum(r["genius_achou"] and r["deezer_achou"] for r in report)
    print(f"\n[verificar] {len(report)} verificadas.")
    print(
        f"  Genius: {n_genius}/{len(report)}  Deezer: {n_deezer}/{len(report)}  "
        f"Ambos: {n_both}/{len(report)}"
    )
    print(f"[verificar] relatório em {out_path}")


if __name__ == "__main__":
    main()
