#!/usr/bin/env python
"""Reprocessa na LF1 uma lista de ids e atualiza só essas linhas do ``lf1_raw.csv``.

Serve para corrigir faixas que falharam (``NaN``) sem refazer o lote inteiro. Roda no
container, como ``gerar_lf1.py``::

    python docker/reprocessar_ids.py ID1 ID2 ID3 ...
"""

import os
import sys
from pathlib import Path

from tcc_mer.lf1_audio import estimate_va_from_audio, make_music2emo_predictor
from tcc_mer.lf_common import read_lf_raw

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
AUDIO_DIR = DATA_DIR / "corpus" / "audio"
OUT_CSV = DATA_DIR / "outputs" / "lf1_raw.csv"


def main() -> None:
    ids = sys.argv[1:]
    if not ids:
        sys.exit("uso: python reprocessar_ids.py ID1 ID2 ...")

    print(f"[reprocessar] {len(ids)} id(s): {ids}")
    predictor = make_music2emo_predictor()

    updated = {}
    for track_id in ids:
        try:
            result = estimate_va_from_audio(str(AUDIO_DIR / f"{track_id}.mp3"), predictor)
            updated[track_id] = (result.v_raw, result.a_raw)
            print(f"  {track_id}: V={result.v_raw:+.3f}  A={result.a_raw:+.3f}")
        except Exception as error:
            print(f"  {track_id}: FALHOU ({error!r})")
            updated[track_id] = (float("nan"), float("nan"))

    table = read_lf_raw(OUT_CSV).set_index("id")
    for track_id, (valence, arousal) in updated.items():
        table.loc[track_id, ["V_raw", "A_raw"]] = (valence, arousal)
    table.reset_index().to_csv(OUT_CSV, index=False)
    print(f"[reprocessar] {OUT_CSV} atualizado.")


if __name__ == "__main__":
    main()
