#!/usr/bin/env python
"""Runner da LF2 (léxico ANEW-Br) — lê as letras e gera lf2_raw.csv.

Roda no Python normal do projeto (sem Docker): a LF2 é leve. Descobre as letras
em ``data/stimuli/lyrics/*.txt`` (uma por faixa, nome do arquivo = id), casa os
lemas com o léxico ANEW-Br e escreve:

- ``data/lf_outputs/lf2_raw.csv``   (id, V_raw, A_raw em [-1, 1])
- ``data/lf_outputs/lf2_meta.json`` (léxico, lematizador, cobertura por faixa)

O ANEW-Br é material de terceiros com licença própria — **não é versionado** no
repo. Aponte ``--lexicon`` para o csv local.

Instrumentais (``tem_letra=não``): não têm arquivo de letra, então simplesmente
não aparecem no csv — o join com as demais LFs produz ``NaN`` (nunca zero), como
a metodologia exige. Para marcar um instrumental explicitamente, deixe um .txt
vazio: a LF2 devolve ``NaN`` com ``meta['reason']=='instrumental'``.

Exemplos::

    # arquivo oficial anew-br.txt (preset padrão):
    python scripts/gerar_lf2.py --lexicon data/lexicons/anew-br.txt
    # um CSV padrão seu, sem lematizador:
    python scripts/gerar_lf2.py --lexicon meu_lexico.csv --preset csv --no-lemmatize
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# Permite rodar como `python scripts/gerar_lf2.py` a partir da raiz do repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lf2_lexicon import (  # noqa: E402
    AnewBrLexicon,
    make_spacy_lemmatizer,
    run_batch,
)
from src.lf_common import write_lf_raw  # noqa: E402

# Presets de formato de léxico. 'anew-br-txt' = o anew-br.txt oficial
# (github.com/affective-computing/anew-br): espaços + vírgula decimal + colunas
# em português. 'csv' = CSV padrão (vírgula, ponto decimal).
PRESETS = {
    "csv": dict(word_col="word", valence_col="valence_mean",
                arousal_col="arousal_mean", sep=",", decimal=".", encoding="utf-8"),
    "anew-br-txt": dict(word_col="Palavra", valence_col="Valencia-Media",
                        arousal_col="Alerta-Media", sep=r"\s+", decimal=",",
                        encoding="utf-8"),
}

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera lf2_raw.csv a partir das letras.")
    p.add_argument("--lexicon", required=True, type=Path,
                   help="csv do ANEW-Br (não versionado).")
    p.add_argument("--lyrics-dir", type=Path,
                   default=ROOT / "data" / "stimuli" / "lyrics",
                   help="pasta com <id>.txt (padrão: data/stimuli/lyrics).")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "lf_outputs" / "lf2_raw.csv",
                   help="csv de saída (padrão: data/lf_outputs/lf2_raw.csv).")
    p.add_argument("--preset", choices=list(PRESETS), default="anew-br-txt",
                   help="formato do léxico (padrão: anew-br-txt oficial).")
    p.add_argument("--word-col", help="sobrescreve a coluna de palavra do preset.")
    p.add_argument("--valence-col", help="sobrescreve a coluna de valência (SAM 1-9).")
    p.add_argument("--arousal-col", help="sobrescreve a coluna de arousal (SAM 1-9).")
    p.add_argument("--sep", help="sobrescreve o separador de colunas do preset.")
    p.add_argument("--decimal", help="sobrescreve o caractere decimal do preset.")
    p.add_argument("--no-lemmatize", action="store_true",
                   help="desliga o lematizador spaCy (usa normalização simples).")
    p.add_argument("--no-weight-by-freq", action="store_true",
                   help="agrega por lemas únicos em vez de ponderar por frequência.")
    return p.parse_args()


def descobrir_letras(lyrics_dir: Path) -> list[tuple[str, str, bool]]:
    """Lista (id, texto, tem_letra) a partir dos .txt; vazio => instrumental."""
    if not lyrics_dir.exists():
        sys.exit(f"[lf2] pasta de letras não existe: {lyrics_dir}")
    arquivos = sorted(lyrics_dir.glob("*.txt"))
    tracks: list[tuple[str, str, bool]] = []
    for f in arquivos:
        texto = f.read_text(encoding="utf-8", errors="replace").strip()
        tracks.append((f.stem, texto, bool(texto)))
    return tracks


def escolher_lematizador(no_lemmatize: bool):
    if no_lemmatize:
        print("[lf2] lematizador: DESLIGADO (normalização simples).")
        return None
    try:
        lem = make_spacy_lemmatizer()
        print("[lf2] lematizador: spaCy pt_core_news_sm.")
        return lem
    except ImportError as e:
        print(f"[lf2] spaCy indisponível ({e.__class__.__name__}); "
              f"seguindo com normalização simples. Detalhe: {e}")
        return None


def main() -> None:
    args = parse_args()
    if not args.lexicon.exists():
        sys.exit(f"[lf2] léxico não encontrado: {args.lexicon}")

    fmt = dict(PRESETS[args.preset])  # cópia; flags explícitas sobrescrevem
    for k in ("word_col", "valence_col", "arousal_col", "sep", "decimal"):
        v = getattr(args, k)
        if v is not None:
            fmt[k] = v
    lexicon = AnewBrLexicon.from_csv(args.lexicon, **fmt)
    print(f"[lf2] léxico ({args.preset}) carregado: {len(lexicon.norms)} palavras.")

    tracks = descobrir_letras(args.lyrics_dir)
    if not tracks:
        sys.exit(f"[lf2] nenhuma letra .txt em {args.lyrics_dir}.")
    n_instr = sum(1 for _, _, tl in tracks if not tl)
    print(f"[lf2] {len(tracks)} faixa(s) ({n_instr} instrumental/vazia).")

    lemmatizer = escolher_lematizador(args.no_lemmatize)

    saidas = run_batch(
        tracks, lexicon,
        lemmatizer=lemmatizer,
        weight_by_freq=not args.no_weight_by_freq,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_lf_raw(saidas, args.out)

    n_ok = sum(1 for s in saidas if not math.isnan(s.v_raw))
    coberturas = [s.meta.get("coverage", 0.0) for s in saidas
                  if not math.isnan(s.v_raw)]
    cob_media = sum(coberturas) / len(coberturas) if coberturas else 0.0

    meta = {
        "lexico": str(args.lexicon),
        "n_palavras_lexico": len(lexicon.norms),
        "lematizador": "spacy:pt_core_news_sm" if lemmatizer else "normalizacao_simples",
        "weight_by_freq": not args.no_weight_by_freq,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "n_faixas": len(saidas),
        "n_com_saida": n_ok,
        "cobertura_media": round(cob_media, 3),
        "escala": "SAM 1-9 -> to_unit -> [-1, 1] (V_raw, A_raw)",
    }
    (args.out.parent / "lf2_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[lf2] pronto: {args.out}  ({n_ok}/{len(saidas)} com V/A; "
          f"cobertura média {cob_media:.1%})")
    if cob_media < 0.10 and n_ok:
        print("[lf2] AVISO: cobertura baixa (<10%). Verifique as colunas do "
              "léxico e considere ligar o lematizador (--lemmatize/spaCy).")
    for s in saidas[:10]:
        if math.isnan(s.v_raw):
            print(f"    {s.track_id}: — ({s.meta.get('reason', 'sem match')})")
        else:
            print(f"    {s.track_id}: V={s.v_raw:+.3f}  A={s.a_raw:+.3f}  "
                  f"cob={s.meta['coverage']:.0%} ({s.meta['n_matched']} palavras)")


if __name__ == "__main__":
    main()
