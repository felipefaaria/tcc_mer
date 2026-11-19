#!/usr/bin/env python
"""Extrai letra (Genius) e prévia de áudio (Deezer, depois iTunes) das candidatas históricas.

Lê ``data/corpus/candidatas_historico.csv`` (as candidatas de 1901 a 2025, de fontes
documentadas: *A Canção no Tempo* vol. 1 e 2, as mais tocadas no rádio e as retrospectivas do
Spotify) e grava a letra em ``data/corpus/lyrics`` e o áudio em ``data/corpus/audio``. O nome do
arquivo é o ``id_temp`` da consolidação. Grava também ``data/corpus/coleta_historico_status.csv``
com o status de cada busca. É retomável: o que já tem arquivo salvo é mantido.

A chave do Genius vem de ``GENIUS_ACCESS_TOKEN``, do ambiente ou de um ``.env`` na raiz.

    python scripts/coleta/extrair_corpus_historico.py --dry-run    # só mostra o plano
    python scripts/coleta/extrair_corpus_historico.py --limit 5    # testa em 5
    python scripts/coleta/extrair_corpus_historico.py              # roda tudo
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from tcc_mer.env import load_env, use_utf8_console
from tcc_mer.paths import AUDIO_DIR, CORPUS_DIR, LYRICS_DIR
from tcc_mer.sources import ensure_lyrics, ensure_preview, is_concrete_track

MANIFEST_FIELDS = [
    "id_temp",
    "ano",
    "faixa",
    "interprete_compositor",
    "genius_status",
    "deezer_status",
]


def read_candidates(csv_path: Path) -> tuple[list[dict], list[dict]]:
    """``(processáveis, puladas)``: as puladas não têm faixa concreta."""
    with csv_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    processable = [row for row in rows if is_concrete_track(row["faixa"])]
    return processable, [row for row in rows if not is_concrete_track(row["faixa"])]


def write_manifest(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-csv", type=Path, default=CORPUS_DIR / "candidatas_historico.csv")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--lyrics-dir", type=Path, default=LYRICS_DIR)
    parser.add_argument(
        "--manifest-out", type=Path, default=CORPUS_DIR / "coleta_historico_status.csv"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=1.2, help="pausa (s) entre faixas")
    parser.add_argument("--only-lyrics", action="store_true")
    parser.add_argument("--only-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    use_utf8_console()
    load_env()
    args = parse_args()
    candidates, skipped = read_candidates(args.corpus_csv)
    print(
        f"[historico] {len(candidates)} processável(is), {len(skipped)} pulada(s) "
        "(faixa não concreta)"
    )
    for row in skipped:
        print(f"  PULADA: {row.get('id_temp')} {row['interprete_compositor']} — {row['faixa']!r}")
    if args.limit:
        candidates = candidates[: args.limit]
        print(f"[historico] --limit {args.limit}: processando só essas")

    if args.dry_run:
        for row in candidates:
            print(
                f"  {row['id_temp']}  {row['interprete_compositor']} — "
                f"{row['faixa']} ({row['ano']})"
            )
        print("[historico] --dry-run: nenhuma rede ou arquivo tocado")
        return

    token = None
    if not args.only_audio:
        token = os.environ.get("GENIUS_ACCESS_TOKEN")
        if not token:
            sys.exit("[historico] defina GENIUS_ACCESS_TOKEN no .env (ou use --only-audio)")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.lyrics_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    if args.manifest_out.exists():
        with args.manifest_out.open(encoding="utf-8") as file:
            manifest = list(csv.DictReader(file))

    for position, row in enumerate(candidates, start=1):
        track_id, performer, title = row["id_temp"], row["interprete_compositor"], row["faixa"]
        print(f"[{position}/{len(candidates)}] {track_id}  {performer} — {title} ({row['ano']})")

        if args.only_audio:
            genius_status = "pulado (--only-audio)"
        else:
            genius_status = ensure_lyrics(
                args.lyrics_dir / f"{track_id}.txt", performer, title, token
            )
        print(f"    letra: {genius_status}")

        if args.only_lyrics:
            deezer_status = "pulado (--only-lyrics)"
        else:
            deezer_status = ensure_preview(args.audio_dir / f"{track_id}.mp3", performer, title)
        print(f"    áudio: {deezer_status}")

        record = {
            "id_temp": track_id,
            "ano": row["ano"],
            "faixa": title,
            "interprete_compositor": performer,
            "genius_status": genius_status,
            "deezer_status": deezer_status,
        }
        manifest = [m for m in manifest if m["id_temp"] != track_id] + [record]
        write_manifest(manifest, args.manifest_out)
        time.sleep(args.sleep)

    lyrics_ok = sum(m["genius_status"] in ("ok", "já existia") for m in manifest)
    audio_ok = sum(m["deezer_status"] in ("ok_deezer", "ok_itunes", "já existia") for m in manifest)
    print(
        f"\n[historico] concluído: {lyrics_ok}/{len(manifest)} letras, "
        f"{audio_ok}/{len(manifest)} áudios. Manifesto em {args.manifest_out}"
    )


if __name__ == "__main__":
    main()
