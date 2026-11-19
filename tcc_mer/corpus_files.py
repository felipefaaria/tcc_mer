"""Verificações sobre os arquivos de áudio e de letra do corpus em disco."""

from .paths import AUDIO_DIR, LYRICS_DIR

__all__ = ["MIN_AUDIO_BYTES", "TARGET_PER_YEAR", "has_audio", "has_lyrics_text"]

# Abaixo disso o arquivo é um download truncado ou vazio, não um áudio de 30 s.
MIN_AUDIO_BYTES = 1000

TARGET_PER_YEAR = 5


def has_audio(track_id: str) -> bool:
    """Se a faixa tem um arquivo de áudio de tamanho plausível."""
    path = AUDIO_DIR / f"{track_id}.mp3"
    return path.exists() and path.stat().st_size > MIN_AUDIO_BYTES


def has_lyrics_text(track_id: str) -> bool:
    """Se a faixa tem um arquivo de letra com algum texto (vazio marca instrumental)."""
    path = LYRICS_DIR / f"{track_id}.txt"
    return path.exists() and bool(path.read_text(encoding="utf-8", errors="ignore").strip())
