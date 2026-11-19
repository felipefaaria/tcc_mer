#!/usr/bin/env python
"""Gera ``lexicon_raw.csv`` com valência e arousal da letra pelo léxico ANEW-Br.

Fonte de rótulo **descartada** do artigo (Seção 4.1): o script existe para reproduzir a
comparação. Casa os lemas de cada letra em ``data/corpus/lyrics/<id>.txt`` com as normas do
ANEW-Br e grava:

* ``data/outputs/lexicon_raw.csv``: ``id, V_raw, A_raw`` em [-1, 1];
* ``data/outputs/lexicon_meta.json``: léxico, lematizador e cobertura.

O ANEW-Br tem licença própria e não é versionado: aponte ``--lexicon`` para o arquivo local.
Faixa instrumental (manifesto, ou letra vazia) fica ``NaN``.

    python scripts/rotulagem/gerar_lexicon.py --lexicon data/lexicons/anew-br.txt
    python scripts/rotulagem/gerar_lexicon.py --lexicon meu.csv --preset csv --no-lemmatize
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from tcc_mer.dataset import load_metadata
from tcc_mer.lexicon import AnewBrLexicon, make_spacy_lemmatizer, run_batch
from tcc_mer.lf_common import write_lf_raw
from tcc_mer.paths import LEXICONS_DIR, LYRICS_DIR, OUTPUTS_DIR

TAG = "[lexicon]"

# Formatos de arquivo de léxico: o ``anew-br.txt`` oficial (espaços, vírgula decimal,
# colunas em português) e um CSV padrão.
PRESETS = {
    "anew-br-txt": {
        "word_col": "Palavra",
        "valence_col": "Valencia-Media",
        "arousal_col": "Alerta-Media",
        "sep": r"\s+",
        "decimal": ",",
    },
    "csv": {
        "word_col": "word",
        "valence_col": "valence_mean",
        "arousal_col": "arousal_mean",
        "sep": ",",
        "decimal": ".",
    },
}
FORMAT_OPTIONS = ("word_col", "valence_col", "arousal_col", "sep", "decimal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lexicon", type=Path, default=LEXICONS_DIR / "anew-br.txt")
    parser.add_argument("--lyrics-dir", type=Path, default=LYRICS_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "lexicon_raw.csv")
    parser.add_argument("--preset", choices=list(PRESETS), default="anew-br-txt")
    for option in FORMAT_OPTIONS:
        parser.add_argument(f"--{option.replace('_', '-')}", help=f"sobrescreve {option} do preset")
    parser.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="não usa o lematizador spaCy (usa os tokens como estão)",
    )
    parser.add_argument(
        "--no-weight-by-freq",
        action="store_true",
        help="média sobre lemas únicos em vez de ocorrências",
    )
    return parser.parse_args()


def discover_lyrics(lyrics_dir: Path) -> list[tuple[str, str, bool]]:
    """``(id, texto, tem_letra)`` de cada faixa; o manifesto marca as instrumentais."""
    if not lyrics_dir.exists():
        sys.exit(f"{TAG} pasta de letras não existe: {lyrics_dir}")
    metadata = load_metadata()
    instrumental = set(metadata.loc[metadata["instrumental"], "id"])
    tracks = []
    for path in sorted(lyrics_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        tracks.append((path.stem, text, bool(text) and path.stem not in instrumental))
    return tracks


def choose_lemmatizer(disabled: bool):
    if disabled:
        print(f"{TAG} lematizador desligado (tokens como estão)")
        return None
    try:
        lemmatizer = make_spacy_lemmatizer()
    except ImportError as error:
        print(f"{TAG} spaCy indisponível, seguindo sem lematizar: {error}")
        return None
    print(f"{TAG} lematizador: spaCy pt_core_news_sm")
    return lemmatizer


def main() -> None:
    args = parse_args()
    if not args.lexicon.exists():
        sys.exit(f"{TAG} léxico não encontrado: {args.lexicon}")

    options = dict(PRESETS[args.preset])
    options.update({k: v for k in FORMAT_OPTIONS if (v := getattr(args, k)) is not None})
    lexicon = AnewBrLexicon.from_csv(args.lexicon, **options)
    print(f"{TAG} léxico ({args.preset}): {len(lexicon.norms)} palavras")

    tracks = discover_lyrics(args.lyrics_dir)
    if not tracks:
        sys.exit(f"{TAG} nenhuma letra .txt em {args.lyrics_dir}")
    lemmatizer = choose_lemmatizer(args.no_lemmatize)
    outputs = run_batch(
        tracks, lexicon, lemmatizer=lemmatizer, weight_by_freq=not args.no_weight_by_freq
    )
    write_lf_raw(outputs, args.out)

    rated = [o for o in outputs if not math.isnan(o.v_raw)]
    coverage = sum(o.meta["coverage"] for o in rated) / len(rated) if rated else 0.0
    meta = {
        "lexico": str(args.lexicon),
        "n_palavras_lexico": len(lexicon.norms),
        "lematizador": "spacy:pt_core_news_sm" if lemmatizer else "normalizacao_simples",
        "weight_by_freq": not args.no_weight_by_freq,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "n_faixas": len(outputs),
        "n_com_saida": len(rated),
        "cobertura_media": round(coverage, 3),
        "escala": "SAM 1-9 -> to_unit -> [-1, 1] (V_raw, A_raw)",
    }
    (args.out.parent / "lexicon_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{TAG} {args.out}: {len(rated)}/{len(outputs)} com V/A; cobertura média {coverage:.1%}")
    if rated and coverage < 0.10:
        print(f"{TAG} aviso: cobertura baixa (<10%). Confira as colunas e use o lematizador.")


if __name__ == "__main__":
    main()
