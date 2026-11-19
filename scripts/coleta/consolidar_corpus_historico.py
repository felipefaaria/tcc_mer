#!/usr/bin/env python
"""Consolida as candidatas históricas no corpus final.

Fica só com as músicas que têm áudio e letra (ou instrumental confirmado), renomeia os ids para
``<ano>-<seq>`` e grava ``data/corpus/manifest_historico.csv``.

Por padrão é um ensaio (dry-run) que mostra o que faria. Com ``--apply``, renomeia de fato os
arquivos de áudio e de letra, apaga os arquivos parciais das candidatas incompletas e grava o csv.
"""

import argparse
import csv
import shutil

from tcc_mer.corpus_files import has_audio, has_lyrics_text
from tcc_mer.paths import AUDIO_DIR, CORPUS_DIR, LYRICS_DIR

SOURCE_CSV = CORPUS_DIR / "candidatas_historico.csv"
TARGET_CSV = CORPUS_DIR / "manifest_historico.csv"

# O csv de origem tem rótulos de fonte com a acentuação corrompida.
SOURCE_FIXES = {
    "Can\ufffdo da Hist\ufffdria (letra manual: MPB Cifrantiga/IMS)": (
        "Canção da História (letra manual: MPB Cifrantiga/IMS)"
    ),
    "Can\ufffdo da Hist\ufffdria": "Canção da História",
}

FINAL_COLUMNS = [
    "id",
    "ano",
    "faixa",
    "interprete",
    "genero",
    "contexto",
    "fonte",
    "instrumental",
    "arquivo_audio",
    "arquivo_letra",
]


def select_complete(rows: list[dict]) -> list[dict]:
    """Candidatas com áudio e letra (ou instrumental confirmado), com a marca ``_instrumental``."""
    complete = []
    for row in rows:
        track_id = row["id_temp"]
        instrumental = "instrumental" in row["status"].lower()
        if has_audio(track_id) and (instrumental or has_lyrics_text(track_id)):
            complete.append({**row, "_instrumental": instrumental})
    return complete


def final_row(row: dict, new_id: str) -> dict:
    source = row["fonte"].strip()
    return {
        "id": new_id,
        "ano": row["ano"],
        "faixa": row["faixa"],
        "interprete": row["interprete_compositor"],
        "genero": row["genero"],
        "contexto": row["nota_sucesso"].strip(),
        "fonte": SOURCE_FIXES.get(source, source),
        "instrumental": "sim" if row["_instrumental"] else "não",
        "arquivo_audio": f"data/corpus/audio/{new_id}.mp3",
        "arquivo_letra": f"data/corpus/lyrics/{new_id}.txt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="executa de fato (renomeia arquivos e grava o csv); sem isto é um ensaio",
    )
    args = parser.parse_args()

    with SOURCE_CSV.open(encoding="utf-8") as file:
        candidates = list(csv.DictReader(file))
    complete = sorted(select_complete(candidates), key=lambda r: (int(r["ano"]), r["faixa"]))

    per_year: dict[int, int] = {}
    final_rows, renames = [], []
    for row in complete:
        year = int(row["ano"])
        per_year[year] = per_year.get(year, 0) + 1
        new_id = f"{year}-{per_year[year]:02d}"
        renames.append((row["id_temp"], new_id))
        final_rows.append(final_row(row, new_id))

    print(f"[consolidar] {len(final_rows)} músicas completas (de {len(candidates)} candidatas)")
    print(f"[consolidar] anos cobertos: {min(per_year)}–{max(per_year)}")
    if not args.apply:
        print("\n--- ENSAIO (nada foi alterado) ---")
        for old, new in renames[:10]:
            print(f"  {old} -> {new}")
        print("  ...")
        return

    for old_id, new_id in renames:
        for folder, extension in ((AUDIO_DIR, ".mp3"), (LYRICS_DIR, ".txt")):
            source = folder / f"{old_id}{extension}"
            if source.exists():
                shutil.move(str(source), str(folder / f"{new_id}{extension}"))

    kept = {old for old, _ in renames}
    removed = 0
    for track_id in {row["id_temp"] for row in candidates} - kept:
        for folder, extension in ((AUDIO_DIR, ".mp3"), (LYRICS_DIR, ".txt")):
            partial = folder / f"{track_id}{extension}"
            if partial.exists():
                partial.unlink()
                removed += 1
    print(f"[consolidar] {removed} arquivo(s) parcial(is) de músicas incompletas removido(s)")

    with TARGET_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FINAL_COLUMNS)
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"[consolidar] arquivos renomeados e csv final gravado em {TARGET_CSV}")


if __name__ == "__main__":
    main()
