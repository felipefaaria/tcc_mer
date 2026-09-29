#!/usr/bin/env python
"""Fallback de áudio via YouTube para músicas que Deezer/iTunes não encontraram.

Só deve ser usado DEPOIS que ``extrair_corpus_historico.py`` (ou o fluxo
equivalente) já tentou Deezer/iTunes e falhou — é o último recurso descrito
pelo usuário: "extraindo o audio via youtube".

Busca automaticamente (``ytsearch3:``), reaproveitando a MESMA validação de
similaridade de título usada em Genius/Deezer/iTunes (``_titulo_bate`` de
``extrair_corpus.py``) para reduzir o risco de pegar um vídeo errado — risco
que aqui é maior que nas outras fontes, já que o YouTube tem muito mais lixo
(covers, playlists inteiras, vídeos sem relação nenhuma) do que uma API de
catálogo musical. Por isso também:

- rejeita vídeos fora de uma faixa de duração plausível para uma gravação de
  época (30s-12min) — descarta compilações/álbuns inteiros e vídeos
  claramente não-musicais;
- baixa só um TRECHO de ~30s (mesmo padrão de preview usado no resto do
  corpus, via Deezer/iTunes), não a faixa inteira;
- grava sempre no manifest qual vídeo foi usado (URL, título, canal) para
  auditoria manual posterior — como esse é o fallback com maior risco de
  falso positivo, precisa ficar fácil de conferir depois.

Usa o ffmpeg empacotado pelo pacote ``imageio-ffmpeg`` (não precisa instalar
ffmpeg à parte no sistema).

Exemplos::

    python scripts/extrair_audio_youtube.py --interprete "Eduardo das Neves" --faixa "A Conquista do Ar" --id H0502 --dry-run
    python scripts/extrair_audio_youtube.py --interprete "Eduardo das Neves" --faixa "A Conquista do Ar" --id H0502
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent

from extrair_corpus import _titulo_bate, _primeiro_artista  # noqa: E402

DURACAO_MIN_S = 30
DURACAO_MAX_S = 12 * 60
CLIPE_S = 30


def _titulo_bate_youtube(faixa: str, titulo_video: str) -> bool:
    """Validação de título adaptada pro YouTube.

    Títulos de vídeo (esp. de canais como MPB CIFRANTIGA) costumam vir cheios
    de contexto extra — "Farrula (valsa, 1905) - Anacleto de Medeiros" — que
    dilui a razão de similaridade do ``_titulo_bate`` original (calibrado pra
    títulos "limpos" de Genius/Deezer) e rejeita vídeos corretos por engano.
    Aceita se o nome da faixa aparecer inteiro como substring do título
    normalizado (cobre esse caso comum), OU se passar na checagem original.
    """
    import re as _re
    import unicodedata as _ud

    def norm(s: str) -> str:
        # tira acento (NFKD + remove marcas combinantes) ANTES de filtrar —
        # senão "ô"/"é" etc. somem em vez de virar "o"/"e", e uma palavra como
        # "Pierrô" nunca bate com "Pierrot" ou com o mesmo nome sem acento.
        sem_acento = "".join(c for c in _ud.normalize("NFKD", s)
                              if not _ud.combining(c))
        return _re.sub(r"[^a-z0-9 ]", "", sem_acento.lower()).strip()

    if norm(faixa) and norm(faixa) in norm(titulo_video):
        return True
    return _titulo_bate(faixa, titulo_video)


def _ffmpeg_path() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def buscar_candidatos(interprete: str, faixa: str, n: int = 3) -> list[dict]:
    import yt_dlp
    query = f"{_primeiro_artista(interprete)} {faixa}"
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist",
            "default_search": "ytsearch"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return info.get("entries") or []


# Canais especializados em reeditar 78rpm/cilindros de época com título
# arquivístico correto (compositor/intérprete/ano) — um match nesses canais é
# muito mais confiável que um título curto e genérico bater por coincidência
# (ex.: "Fadário!" de um canal qualquer, que não tem nada a ver com "O
# Fadário" de Anacleto de Medeiros/Catulo, mas passa na checagem de substring).
CANAIS_CONFIAVEIS = {
    "mpb cifrantiga", "som em bom tom", "78rpm gramophone records and cylinders",
    "estudio78rpm (otacílio)", "luciano hortencio",
}


def escolher_candidato(candidatos: list[dict], faixa: str) -> tuple[dict | None, str]:
    validos = []
    for c in candidatos:
        titulo = c.get("title", "")
        duracao = c.get("duration") or 0
        if not _titulo_bate_youtube(faixa, titulo):
            continue
        if not (DURACAO_MIN_S <= duracao <= DURACAO_MAX_S):
            continue
        validos.append(c)
    if validos:
        # entre os válidos, prioriza canal arquivístico conhecido antes de
        # cair pra "primeiro que bateu" (mais sujeito a coincidência de título curto).
        for c in validos:
            if (c.get("channel") or "").strip().lower() in CANAIS_CONFIAVEIS:
                return c, "ok"
        return validos[0], "ok"
    if candidatos:
        return None, (f"nenhum candidato passou na validação "
                       f"(título/duração) — top resultado foi "
                       f"{candidatos[0].get('title')!r} "
                       f"({candidatos[0].get('duration')}s)")
    return None, "sem_resultado"


def baixar_clipe(video_id: str, destino: Path) -> None:
    import yt_dlp
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "quiet": True, "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(destino.with_suffix("")) + ".%(ext)s",
        "ffmpeg_location": _ffmpeg_path(),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                             "preferredquality": "128"}],
        "postprocessor_args": {"ffmpeg": ["-t", str(CLIPE_S)]},
        "overwrites": True,
        # YouTube costuma devolver 403 pro cliente "web" padrão do yt-dlp
        # (assinatura/cifra bloqueada) — o cliente "android" contorna isso
        # na maioria dos casos, sem precisar de cookies/login.
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--interprete", required=True)
    p.add_argument("--faixa", required=True)
    p.add_argument("--id", required=True, help="id_temp (ex.: H0502) — nome do arquivo final.")
    p.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "corpus" / "audio")
    p.add_argument("--dry-run", action="store_true", help="só mostra o candidato escolhido, não baixa.")
    p.add_argument("--n", type=int, default=3, help="quantos resultados de busca considerar.")
    args = p.parse_args()

    candidatos = buscar_candidatos(args.interprete, args.faixa, n=args.n)
    escolhido, status = escolher_candidato(candidatos, args.faixa)

    print(f"[youtube] busca: {args.interprete!r} / {args.faixa!r}")
    for c in candidatos:
        marca = " <-- escolhido" if escolhido and c.get("id") == escolhido.get("id") else ""
        print(f"  - {c.get('title')!r} ({c.get('duration')}s) "
              f"canal={c.get('channel')!r} url=https://www.youtube.com/watch?v={c.get('id')}{marca}")

    if escolhido is None:
        print(f"[youtube] RESULTADO: {status}")
        sys.exit(1)

    url = f"https://www.youtube.com/watch?v={escolhido['id']}"
    print(f"[youtube] escolhido: {escolhido.get('title')!r} — {url}")

    if args.dry_run:
        print("[youtube] --dry-run: nenhum download feito.")
        return

    destino = args.audio_dir / f"{args.id}.mp3"
    baixar_clipe(escolhido["id"], destino)
    print(f"[youtube] salvo em {destino} "
          f"(fonte: {url}, título: {escolhido.get('title')!r}, "
          f"canal: {escolhido.get('channel')!r}) — REGISTRE essa proveniência no manifest.")


if __name__ == "__main__":
    main()
