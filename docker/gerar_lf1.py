#!/usr/bin/env python
"""Usina da LF1 — roda o Music2Emo sobre os excertos e grava lf1_raw.csv.

Este script roda DENTRO do container (ambiente pesado). NÃO importar no repo de
análise. Ele descobre os áudios em ``$DATA_DIR/corpus/audio/``, roda o modelo em
lote e escreve:

- ``$DATA_DIR/outputs/lf1_raw.csv``  (id, V_raw, A_raw em [-1, 1])
- ``$DATA_DIR/outputs/lf1_meta.json`` (modelo, data, contagens — reprodutibilidade)

Variáveis de ambiente:
- ``DATA_DIR``  raiz dos dados montados (padrão ``/app/data``).
- ``LF1_LIMIT`` se > 0, processa só as N primeiras faixas (smoke test). Padrão 0 = todas.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.lf1_audio import make_music2emo_predictor, run_batch
from src.lf_common import write_lf_raw

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
AUDIO_DIR = DATA_DIR / "corpus" / "audio"
OUT_CSV = DATA_DIR / "outputs" / "lf1_raw.csv"
META_JSON = DATA_DIR / "outputs" / "lf1_meta.json"
LIMIT = int(os.environ.get("LF1_LIMIT", "0"))

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


def descobrir_faixas() -> list[tuple[str, str]]:
    """Lista (id, caminho) dos áudios; o id é o nome do arquivo sem extensão."""
    if not AUDIO_DIR.exists():
        sys.exit(f"[lf1] pasta de áudio não existe: {AUDIO_DIR}\n"
                 f"      monte os excertos em {AUDIO_DIR} (ver docker/README.md).")
    arquivos = sorted(p for p in AUDIO_DIR.iterdir()
                      if p.suffix.lower() in AUDIO_EXTS)
    if LIMIT > 0:
        arquivos = arquivos[:LIMIT]
    return [(p.stem, str(p)) for p in arquivos]


def main() -> None:
    faixas = descobrir_faixas()
    if not faixas:
        sys.exit(f"[lf1] nenhum áudio encontrado em {AUDIO_DIR} "
                 f"(extensões aceitas: {', '.join(AUDIO_EXTS)}).")

    modo = f"SMOKE TEST ({LIMIT} faixas)" if LIMIT > 0 else "corpus completo"
    print(f"[lf1] {len(faixas)} faixa(s) — {modo}.")
    print("[lf1] carregando Music2Emo (na 1ª vez baixa os pesos do MERT)...")
    predictor = make_music2emo_predictor()

    print("[lf1] modelo pronto. Rodando inferência...")
    saidas = run_batch(faixas, predictor, on_error="nan")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_lf_raw(saidas, OUT_CSV)

    n_ok = sum(1 for s in saidas if not math.isnan(s.v_raw))
    META_JSON.write_text(
        json.dumps(
            {
                "modelo": "music2emo (AMAAI-Lab)",
                "fonte": "https://github.com/AMAAI-Lab/Music2Emotion",
                "gerado_em": datetime.now(timezone.utc).isoformat(),
                "n_faixas": len(saidas),
                "n_com_saida": n_ok,
                "limite_smoke": LIMIT,
                "escala": "modelo 1-9 -> to_unit -> [-1, 1] (V_raw, A_raw)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[lf1] pronto: {OUT_CSV}  ({n_ok}/{len(saidas)} com V/A válidos)")
    print("[lf1] amostra (confira se faz sentido antes de escalar):")
    for s in saidas[:10]:
        print(f"    {s.track_id}: V={s.v_raw:+.3f}  A={s.a_raw:+.3f}  "
              f"moods={s.meta.get('predicted_moods')}")


if __name__ == "__main__":
    main()
