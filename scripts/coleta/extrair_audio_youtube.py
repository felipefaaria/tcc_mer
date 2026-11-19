#!/usr/bin/env python
"""Último recurso de áudio: um trecho de 30 s do YouTube, para o que o Deezer e o iTunes não têm.

Só deve rodar DEPOIS de ``extrair_corpus_historico.py`` ter tentado o Deezer e o iTunes. Busca
automaticamente (``ytsearch``) e reaproveita a validação de título das outras fontes
(:func:`tcc_mer.sources.title_matches`), com mais cuidado, porque o YouTube tem muito mais lixo
(covers, playlists inteiras, vídeos sem relação) do que uma API de catálogo:

* rejeita vídeos fora de uma duração plausível para uma gravação (30 s a 12 min), o que
  descarta compilações e vídeos não musicais;
* baixa só um trecho de 30 s, o mesmo formato da prévia das outras fontes;
* mostra o vídeo escolhido (URL, título, canal) para a proveniência ser registrada no manifesto
  e conferida depois: é a fonte com maior risco de falso positivo.

Usa o ffmpeg empacotado em ``imageio-ffmpeg``; não precisa instalar o ffmpeg no sistema.

    python scripts/coleta/extrair_audio_youtube.py --interprete "Eduardo das Neves" \\
        --faixa "A Conquista do Ar" --id H0502 --dry-run
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

from tcc_mer.env import use_utf8_console
from tcc_mer.paths import AUDIO_DIR
from tcc_mer.sources import first_artist, title_matches

MIN_DURATION_S = 30
MAX_DURATION_S = 12 * 60
CLIP_SECONDS = 30

# Canais que reeditam 78 rpm e cilindros com título de arquivo correto (compositor, intérprete,
# ano). Um resultado deles é muito mais confiável do que um título curto e genérico que bate
# por coincidência com o de outra música.
TRUSTED_CHANNELS = {
    "mpb cifrantiga",
    "som em bom tom",
    "78rpm gramophone records and cylinders",
    "estudio78rpm (otacílio)",
    "luciano hortencio",
}


def youtube_title_matches(title: str, video_title: str) -> bool:
    """Validação de título adaptada ao YouTube.

    Títulos de vídeo costumam vir com contexto extra ("Farrula (valsa, 1905) - Anacleto de
    Medeiros"), o que dilui a similaridade de :func:`title_matches`, calibrada para títulos
    limpos. Aceita se o nome da faixa aparece inteiro dentro do título do vídeo, ou se passa na
    checagem original.
    """

    def normalize(text: str) -> str:
        # Tira o acento antes de filtrar; senão "ô" some em vez de virar "o".
        plain = "".join(
            c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
        )
        return re.sub(r"[^a-z0-9 ]", "", plain.lower()).strip()

    wanted = normalize(title)
    return bool(wanted and wanted in normalize(video_title)) or title_matches(title, video_title)


def search_candidates(performer: str, title: str, n: int = 3) -> list[dict]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "default_search": "ytsearch",
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{first_artist(performer)} {title}", download=False)
    return info.get("entries") or []


def choose_candidate(candidates: list[dict], title: str) -> tuple[dict | None, str]:
    """O melhor candidato que passa na validação de título e de duração, com o status."""
    valid = [
        c
        for c in candidates
        if youtube_title_matches(title, c.get("title", ""))
        and MIN_DURATION_S <= (c.get("duration") or 0) <= MAX_DURATION_S
    ]
    for candidate in valid:
        if (candidate.get("channel") or "").strip().lower() in TRUSTED_CHANNELS:
            return candidate, "ok"
    if valid:
        return valid[0], "ok"
    if candidates:
        top = candidates[0]
        return None, (
            f"nenhum candidato passou na validação (título e duração); o primeiro "
            f"resultado foi {top.get('title')!r} ({top.get('duration')}s)"
        )
    return None, "sem_resultado"


def download_clip(video_id: str, destination: Path) -> None:
    options = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(destination.with_suffix("")) + ".%(ext)s",
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
        ],
        "postprocessor_args": {"ffmpeg": ["-t", str(CLIP_SECONDS)]},
        "overwrites": True,
        # O cliente "web" padrão do yt-dlp costuma receber 403 (assinatura bloqueada); o
        # cliente "android" contorna isso na maioria dos casos, sem cookies nem login.
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])


def main() -> None:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interprete", required=True)
    parser.add_argument("--faixa", required=True)
    parser.add_argument(
        "--id", required=True, help="id da faixa (ex.: H0502), nome do arquivo final"
    )
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--dry-run", action="store_true", help="mostra o candidato, sem baixar")
    parser.add_argument("--n", type=int, default=3, help="quantos resultados de busca considerar")
    args = parser.parse_args()

    candidates = search_candidates(args.interprete, args.faixa, n=args.n)
    chosen, status = choose_candidate(candidates, args.faixa)

    print(f"[youtube] busca: {args.interprete!r} / {args.faixa!r}")
    for c in candidates:
        mark = " <-- escolhido" if chosen and c.get("id") == chosen.get("id") else ""
        print(
            f"  - {c.get('title')!r} ({c.get('duration')}s) canal={c.get('channel')!r} "
            f"url=https://www.youtube.com/watch?v={c.get('id')}{mark}"
        )
    if chosen is None:
        sys.exit(f"[youtube] resultado: {status}")

    url = f"https://www.youtube.com/watch?v={chosen['id']}"
    print(f"[youtube] escolhido: {chosen.get('title')!r} — {url}")
    if args.dry_run:
        print("[youtube] --dry-run: nenhum download feito")
        return

    destination = args.audio_dir / f"{args.id}.mp3"
    download_clip(chosen["id"], destination)
    print(
        f"[youtube] salvo em {destination} (fonte: {url}, título: {chosen.get('title')!r}, "
        f"canal: {chosen.get('channel')!r}). Registre essa proveniência no manifesto."
    )


if __name__ == "__main__":
    main()
