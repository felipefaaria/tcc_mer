#!/usr/bin/env python
"""Extrai letra (Genius) e áudio (Deezer/iTunes) para o CORPUS HISTÓRICO
(``docs/corpus_historico_final.csv`` — 498 músicas, 1901-2025, todas de fontes
documentadas: "A Canção no Tempo" Vol.1/2, Mais Tocadas, Spotify).

Reaproveita os buscadores e o limpador de letra já testados em
``extrair_corpus.py`` (mesmo tratamento de bugs: User-Agent do Genius, blurb de
bio, busca livre no Deezer). Os IDs finais são o ``id_temp`` já atribuído na
consolidação (H0001..H0498) — não reatribui nada, só usa como nome de arquivo.

Escreve em ``data/corpus/audio/`` e ``data/corpus/lyrics/`` (mesmas pastas do
resto do projeto) e grava ``docs/corpus_historico_manifest.csv`` com o status
de cada busca — retomável: pula o que já tem arquivo salvo.

Exemplos::

    python scripts/extrair_corpus_historico.py --dry-run       # so mostra o plano
    python scripts/extrair_corpus_historico.py --limit 5        # testa em 5
    python scripts/extrair_corpus_historico.py                  # roda tudo
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# console do Windows costuma vir em cp1252, que não sabe imprimir todo Unicode
# (ex.: um título estrangeiro raro num "match_suspeito") — força UTF-8 com
# substituição em vez de deixar o print derrubar o processo no meio da rodada.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from extrair_corpus import (  # noqa: E402
    buscar_letra_genius, buscar_preview_deezer, buscar_preview_itunes,
    faixa_e_concreta,
)


def ler_corpus(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    processaveis, pulados = [], []
    for row in linhas:
        if not faixa_e_concreta(row["faixa"]):
            pulados.append(row)
            continue
        processaveis.append(row)
    return processaveis, pulados


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus-csv", type=Path,
                   default=ROOT / "docs" / "corpus_historico_final.csv")
    p.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "corpus" / "audio")
    p.add_argument("--lyrics-dir", type=Path, default=ROOT / "data" / "corpus" / "lyrics")
    p.add_argument("--manifest-out", type=Path,
                   default=ROOT / "docs" / "corpus_historico_manifest.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=1.2,
                   help="pausa (s) entre músicas, por etiqueta com Genius/Deezer.")
    p.add_argument("--only-lyrics", action="store_true")
    p.add_argument("--only-audio", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    processaveis, pulados = ler_corpus(args.corpus_csv)
    print(f"[historico] {len(processaveis)} processável(is), {len(pulados)} pulada(s) "
          f"(faixa não concreta).")
    for r in pulados:
        print(f"  PULADA: {r.get('id_temp')} {r['interprete_compositor']} — {r['faixa']!r}")

    if args.limit:
        processaveis = processaveis[:args.limit]
        print(f"[historico] --limit {args.limit}: processando só essas.")

    if args.dry_run:
        for r in processaveis:
            print(f"  {r['id_temp']}  {r['interprete_compositor']} — {r['faixa']} ({r['ano']})")
        print("[historico] --dry-run: nenhuma rede/arquivo tocado.")
        return

    token = None
    if not args.only_audio:
        import os
        token = os.environ.get("GENIUS_ACCESS_TOKEN")
        if not token:
            sys.exit("[historico] defina GENIUS_ACCESS_TOKEN no .env "
                     "(ou use --only-audio para pular a letra).")

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.lyrics_dir.mkdir(parents=True, exist_ok=True)

    manifest_campos = ["id_temp", "ano", "faixa", "interprete_compositor",
                       "genius_status", "deezer_status"]
    manifest: list[dict] = []
    if args.manifest_out.exists():
        with args.manifest_out.open(encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
    ja_feito = {m["id_temp"] for m in manifest}

    for i, row in enumerate(processaveis, start=1):
        tid = row["id_temp"]
        interprete = row["interprete_compositor"]
        faixa = row["faixa"]
        print(f"[{i}/{len(processaveis)}] {tid}  {interprete} — {faixa} ({row['ano']})")

        lyrics_path = args.lyrics_dir / f"{tid}.txt"
        audio_path = args.audio_dir / f"{tid}.mp3"

        if args.only_audio:
            genius_status = "pulado (--only-audio)"
        elif lyrics_path.exists() and lyrics_path.read_text(encoding="utf-8").strip():
            genius_status = "já existia"
            print("    letra: já existia, mantida")
        else:
            texto, genius_status = buscar_letra_genius(interprete, faixa, token)
            lyrics_path.write_text(texto, encoding="utf-8")
            print(f"    letra: {genius_status} ({len(texto)} chars)")

        if args.only_lyrics:
            deezer_status = "pulado (--only-lyrics)"
        elif audio_path.exists():
            deezer_status = "já existia"
            print("    áudio: já existia, mantido")
        else:
            conteudo, deezer_status = buscar_preview_deezer(interprete, faixa)
            if conteudo is None:
                conteudo, deezer_status = buscar_preview_itunes(interprete, faixa)
            if conteudo is not None:
                audio_path.write_bytes(conteudo)
            print(f"    áudio: {deezer_status}"
                  f"{'' if conteudo is None else f' ({len(conteudo)} bytes)'}")

        registro = {"id_temp": tid, "ano": row["ano"], "faixa": faixa,
                   "interprete_compositor": interprete,
                   "genius_status": genius_status, "deezer_status": deezer_status}
        manifest = [m for m in manifest if m["id_temp"] != tid] + [registro]
        with args.manifest_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=manifest_campos)
            w.writeheader()
            w.writerows(manifest)

        time.sleep(args.sleep)

    ok_letra = sum(1 for m in manifest if m["genius_status"] in ("ok", "já existia"))
    ok_audio = sum(1 for m in manifest if m["deezer_status"] in
                   ("ok_deezer", "ok_itunes", "já existia"))
    print(f"\n[historico] concluído: {ok_letra}/{len(manifest)} letras, "
          f"{ok_audio}/{len(manifest)} áudios. Manifest em {args.manifest_out}")


if __name__ == "__main__":
    main()
