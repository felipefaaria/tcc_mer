#!/usr/bin/env python
"""LF1: roda o Music2Emo sobre os trechos de áudio e grava ``lf1_raw.csv``.

Roda DENTRO do container (ambiente pesado). Descobre os áudios em
``$DATA_DIR/corpus/audio/``, infere em lote e escreve:

* ``$DATA_DIR/outputs/lf1_raw.csv``: ``id, V_raw, A_raw`` em [-1, 1];
* ``$DATA_DIR/outputs/lf1_meta.json``: modelo, data e contagens.

Variáveis de ambiente:

* ``DATA_DIR``: raiz dos dados montados (padrão ``/app/data``);
* ``LF1_LIMIT``: se maior que 0, processa só as N primeiras faixas (teste rápido).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tcc_mer.lf1_audio import make_music2emo_predictor, run_batch
from tcc_mer.lf_common import write_lf_raw

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
AUDIO_DIR = DATA_DIR / "corpus" / "audio"
OUT_CSV = DATA_DIR / "outputs" / "lf1_raw.csv"
META_JSON = DATA_DIR / "outputs" / "lf1_meta.json"
LIMIT = int(os.environ.get("LF1_LIMIT", "0"))

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
SAMPLE_SIZE = 10


def discover_tracks() -> list[tuple[str, str]]:
    """``(id, caminho)`` dos áudios; o id é o nome do arquivo sem extensão."""
    if not AUDIO_DIR.exists():
        sys.exit(f"[lf1] pasta de áudio não existe: {AUDIO_DIR} (ver docker/README.md)")
    files = sorted(p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    return [(p.stem, str(p)) for p in (files[:LIMIT] if LIMIT > 0 else files)]


def main() -> None:
    tracks = discover_tracks()
    if not tracks:
        sys.exit(f"[lf1] nenhum áudio em {AUDIO_DIR} (extensões: {', '.join(AUDIO_EXTENSIONS)})")

    mode = f"teste rápido ({LIMIT} faixas)" if LIMIT > 0 else "corpus completo"
    print(f"[lf1] {len(tracks)} faixa(s), {mode}")
    print("[lf1] carregando o Music2Emo (na 1ª vez baixa os pesos do MERT)...")
    predictor = make_music2emo_predictor()

    print("[lf1] rodando a inferência...")
    outputs = run_batch(tracks, predictor, on_error="nan")
    write_lf_raw(outputs, OUT_CSV)

    n_rated = sum(not math.isnan(o.v_raw) for o in outputs)
    meta = {
        "modelo": "music2emo (AMAAI-Lab)",
        "fonte": "https://github.com/AMAAI-Lab/Music2Emotion",
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "n_faixas": len(outputs),
        "n_com_saida": n_rated,
        "limite_smoke": LIMIT,
        "escala": "modelo 1-9 -> to_unit -> [-1, 1] (V_raw, A_raw)",
    }
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[lf1] {OUT_CSV}: {n_rated}/{len(outputs)} com V/A válidos")
    print("[lf1] amostra (confira se faz sentido antes de escalar):")
    for output in outputs[:SAMPLE_SIZE]:
        print(
            f"    {output.track_id}: V={output.v_raw:+.3f}  A={output.a_raw:+.3f}  "
            f"moods={output.meta.get('predicted_moods')}"
        )


if __name__ == "__main__":
    main()
