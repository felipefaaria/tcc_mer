#!/usr/bin/env python
"""Mede quanto do trecho de 30 s tem voz cantada, com o Demucs (stem ``vocals``).

O LLM lê a letra inteira, mas o avaliador ouve só 30 s. Trechos instrumentais ou com
pouca voz não permitem julgar a letra e devem ficar de fora da seleção.

Para cada faixa grava ``dur``, ``voz_db_med`` e ``frac_voz_<limiar>``: a fração de janelas
de 0,5 s em que a voz está acima do limiar (em dB, relativo ao nível médio da mix). As
faixas instrumentais entram como controle para escolher o limiar. É retomável: pula as
faixas já medidas.

    python scripts/validacao_humana/medir_voz.py                 # pool + instrumentais
    python scripts/validacao_humana/medir_voz.py --ids A B C     # só algumas
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from tcc_mer.dataset import load_metadata
from tcc_mer.paths import AUDIO_DIR, OUTPUTS_DIR

WINDOW_SECONDS = 0.5
THRESHOLDS_DB = (-24, -18, -12)
SAVE_EVERY = 10


def decode(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    """Decodifica o áudio com o ffmpeg para um array ``(canais, amostras)`` em float32."""
    raw = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "quiet",
            "-i",
            str(path),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        stdout=subprocess.PIPE,
        check=False,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, channels).T.copy()


def measure_voice(model, path: Path) -> dict:
    """Mede a presença de voz no trecho: duração, mediana em dB e fração acima de cada limiar."""
    wave = torch.from_numpy(decode(path, model.samplerate, model.audio_channels))
    reference = wave.mean(0)
    normalized = (wave - reference.mean()) / (reference.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(
            model, normalized[None], device="cpu", progress=False, split=True, overlap=0.1
        )[0]
    sources = sources * reference.std() + reference.mean()
    vocals = sources[model.sources.index("vocals")].mean(0).numpy()
    mix = wave.mean(0).numpy()

    window = int(WINDOW_SECONDS * model.samplerate)
    n_windows = len(mix) // window

    def window_rms(signal: np.ndarray) -> np.ndarray:
        return np.sqrt((signal[: n_windows * window].reshape(n_windows, window) ** 2).mean(1))

    level_db = 20 * np.log10((window_rms(vocals) + 1e-9) / (window_rms(mix).mean() + 1e-9))

    result = {
        "dur": round(n_windows * WINDOW_SECONDS, 1),
        "voz_db_med": round(float(np.median(level_db)), 1),
    }
    for threshold in THRESHOLDS_DB:
        result[f"frac_voz_{abs(threshold)}"] = round(float((level_db > threshold).mean()), 3)
    return result


def tracks_to_measure(explicit: list[str] | None, done: set[str]) -> list[str]:
    """Faixas ainda não medidas e com áudio disponível."""
    if explicit:
        wanted = explicit
    else:
        pool = pd.read_csv(OUTPUTS_DIR / "validacao_pool.csv")["id"].tolist()
        metadata = load_metadata()
        controls = metadata.loc[metadata["instrumental"], "id"].tolist()
        wanted = pool + [track for track in controls if track not in pool]
    return [
        track for track in wanted if track not in done and (AUDIO_DIR / f"{track}.mp3").exists()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "validacao_voz.csv")
    args = parser.parse_args()

    measured = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame()
    pending = tracks_to_measure(args.ids, set(measured.get("id", [])))
    print(f"{len(pending)} faixa(s) a medir (~10 s cada na CPU)")
    model = get_model("htdemucs")
    model.eval()

    rows = []
    for position, track in enumerate(pending, 1):
        try:
            rows.append({"id": track, **measure_voice(model, AUDIO_DIR / f"{track}.mp3")})
            print(f"[{position}/{len(pending)}] {track}: {rows[-1]}", flush=True)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"[{position}/{len(pending)}] {track}: ERRO {error!r}", flush=True)
        if position % SAVE_EVERY == 0 or position == len(pending):
            new_rows = pd.DataFrame(rows)
            table = (
                pd.concat([measured, new_rows], ignore_index=True) if len(measured) else new_rows
            )
            table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("salvo:", args.out)


if __name__ == "__main__":
    main()
