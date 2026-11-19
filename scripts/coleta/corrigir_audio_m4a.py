#!/usr/bin/env python
"""Corrige áudios salvos com extensão ``.mp3`` que na verdade são M4A/AAC.

O download direto da prévia do iTunes serve AAC bruto, seja qual for a extensão pedida, e o
``mpg123`` (usado pelo Music2Emo dentro do container) não lê esse formato como MP3: o resultado
é ``NaN``. O script detecta pela assinatura real do arquivo, e não pela extensão, e recodifica
para MP3 de verdade com o ffmpeg do ``imageio-ffmpeg``, sobrescrevendo o original só se a
conversão der certo.
"""

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

from tcc_mer.corpus_files import MIN_AUDIO_BYTES
from tcc_mer.paths import AUDIO_DIR


def is_real_mp3(path: Path) -> bool:
    """Se o arquivo começa como MP3: etiqueta ``ID3`` ou quadro de sincronia ``0xFFEx``/``0xFFFx``.

    O M4A e o M4V começam com o tamanho do bloco seguido de ``ftyp``.
    """
    head = path.read_bytes()[:4]
    return head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)


def convert_to_mp3(path: Path) -> bool:
    """Recodifica ``path`` para MP3 e o substitui; devolve ``False`` se a conversão falhar."""
    temporary = path.with_suffix(".fix.mp3")
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(temporary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        result.returncode != 0
        or not temporary.exists()
        or temporary.stat().st_size < MIN_AUDIO_BYTES
    ):
        print(f"  FALHOU: {path.name}\n{result.stderr[-500:]}")
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(path)
    return True


def main() -> None:
    targets = [p for p in sorted(AUDIO_DIR.glob("*.mp3")) if not is_real_mp3(p)]
    print(f"[corrigir] {len(targets)} arquivo(s) não MP3 encontrados (pela assinatura)")
    failures = 0
    for path in targets:
        if convert_to_mp3(path):
            print(f"  ok: {path.name}")
        else:
            failures += 1
    print(f"[corrigir] concluído: {len(targets) - failures} corrigido(s), {failures} falha(s)")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
