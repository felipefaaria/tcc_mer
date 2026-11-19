#!/usr/bin/env python
"""Seleciona as faixas do experimento de validação com ouvintes (Seção 3.7 do artigo).

As regras estão em :mod:`tcc_mer.selection`; a seleção é determinística. Ordem de uso::

    python scripts/validacao_humana/selecionar_faixas.py --pool         # validacao_pool.csv
    python scripts/validacao_humana/medir_voz.py                        # validacao_voz.csv
    python scripts/validacao_humana/selecionar_faixas.py --selecionar   # seleção e protocolo

Precisa das letras em ``data/corpus/lyrics`` (contagem de palavras e idioma), das saídas
das LFs e do gold set (``data/annotations/gold_wide.csv``); nenhum deles é versionado.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tcc_mer.dataset import load_metadata
from tcc_mer.lf_common import read_lf_raw
from tcc_mer.paths import ANNOTATIONS_DIR, CORPUS_DIR, LYRICS_DIR, OUTPUTS_DIR, VALIDATION_DIR
from tcc_mer.selection import (
    SelectionParams,
    apply_substitutions,
    attach_metadata,
    attach_voice,
    discordant_pool,
    english_share,
    select_diverse,
    word_count,
)

OUTPUT_COLUMNS = [
    "ordem_selecao",
    "titulo",
    "artista",
    "ano",
    "periodo",
    "genero",
    "macro",
    "origem",
    "estilo_fonte",
    "familia",
    "q_audio",
    "q_fusao",
    "V1",
    "A1",
    "V3",
    "A3",
    "Vf",
    "Af",
    "margem",
    "palavras_letra",
    "share_ingles",
    "frac_voz",
]
EXCLUSIONS = [
    "faixas do gold set",
    "instrumentais",
    "sem nota da letra",
    "letra com poucas palavras",
    "trecho de 30 s com pouca voz (se medido)",
]


def read_lyrics_stats(track_ids: pd.Index) -> pd.DataFrame:
    """Número de palavras e fração de inglês da letra de cada faixa (0 se não houver arquivo)."""
    rows = {}
    for track_id in track_ids:
        path = LYRICS_DIR / f"{track_id}.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rows[track_id] = (word_count(text), english_share(text) if text else 0.0)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["palavras_letra", "share_ingles"])


def build_pool(params: SelectionParams) -> pd.DataFrame:
    lf1 = read_lf_raw(OUTPUTS_DIR / "lf1_raw.csv")
    lf2 = read_lf_raw(OUTPUTS_DIR / "lf2_raw.csv")
    gold_ids = set(pd.read_csv(ANNOTATIONS_DIR / "gold_wide.csv")["trackId"])
    styles_path = CORPUS_DIR / "estilo_provisorio.csv"
    styles = pd.read_csv(styles_path).set_index("id")["estilo"] if styles_path.exists() else None

    pool = attach_metadata(
        discordant_pool(lf1, lf2, gold_ids, params.margin), load_metadata(), styles
    )
    return pool.join(read_lyrics_stats(pool.index)).round(4)


def sha256_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_protocol(
    params: SelectionParams,
    substitutions: pd.DataFrame,
    selected: pd.DataFrame,
    reserve: pd.DataFrame,
) -> None:
    protocol = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pergunta": "nas faixas discordantes (quadrante do áudio != quadrante da fusão), quem "
        "concorda mais com o quadrante humano da música completa?",
        "fusao": "média simples de LF1 e LF2 (valores brutos), valência e arousal",
        "quadrante": "Q1 V>=0,A>=0; Q2 V<0,A>=0; Q3 V<0,A<0; Q4 V>=0,A<0",
        "parametros": asdict(params),
        "exclusoes": EXCLUSIONS,
        "substituicoes_manuais": substitutions.to_dict("records"),
        "diversificacao": "cotas por período e família, teto por macrogênero e artista",
        "insumos_sha256": {
            name: sha256_prefix(OUTPUTS_DIR / name) for name in ("lf1_raw.csv", "lf2_raw.csv")
        },
        "n_selecionadas": len(selected),
        "n_reserva": len(reserve),
    }
    (OUTPUTS_DIR / "validacao_protocolo.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    defaults = SelectionParams()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", action="store_true", help="grava só o pool de discordantes")
    parser.add_argument("--selecionar", action="store_true", help="faz a seleção")
    parser.add_argument("--margem", type=float, default=defaults.margin)
    parser.add_argument("-n", type=int, default=defaults.n)
    parser.add_argument("--reserva", type=int, default=defaults.reserve)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--min-palavras", type=int, default=defaults.min_words, help="palavras na letra completa"
    )
    parser.add_argument(
        "--min-voz", type=float, default=defaults.min_voice, help="fração do trecho de 30 s com voz"
    )
    parser.add_argument(
        "--max-ingles",
        type=float,
        default=defaults.max_english,
        help="exclui letras com fração de palavras-função em inglês a partir "
        "deste valor (padrão: sem filtro; as duas faixas em inglês do "
        "experimento entram em --substituicoes)",
    )
    parser.add_argument(
        "--substituicoes",
        type=Path,
        default=VALIDATION_DIR / "substituicoes_manuais.csv",
        help="faixas trocadas à mão depois da seleção automática",
    )
    parser.add_argument(
        "--limiar-voz",
        type=int,
        default=defaults.voice_threshold_db,
        choices=[12, 18, 24],
        help="limiar (dB) que define janela com voz",
    )
    parser.add_argument("--teto-macro", type=int, default=defaults.cap_macro)
    parser.add_argument("--teto-artista", type=int, default=defaults.cap_artist)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = SelectionParams(
        margin=args.margem,
        n=args.n,
        reserve=args.reserva,
        seed=args.seed,
        min_words=args.min_palavras,
        min_voice=args.min_voz,
        voice_threshold_db=args.limiar_voz,
        max_english=args.max_ingles,
        cap_macro=args.teto_macro,
        cap_artist=args.teto_artista,
    )
    pool = build_pool(params)

    if args.pool or not args.selecionar:
        path = OUTPUTS_DIR / "validacao_pool.csv"
        pool.rename_axis("id").to_csv(path, encoding="utf-8-sig")
        print(f"{path}  ({len(pool)} candidatas discordantes, margem {params.margin})")
        print(pd.crosstab(pool["periodo"], pool["familia"], margins=True).to_string())
        if not args.selecionar:
            return

    voice_path = OUTPUTS_DIR / "validacao_voz.csv"
    voice = pd.read_csv(voice_path).set_index("id") if voice_path.exists() else None
    if voice is None:
        print("AVISO: validacao_voz.csv não existe, a seleção não terá filtro de voz.")
    pool = attach_voice(pool, voice, params.voice_threshold_db)
    selected, reserve = select_diverse(pool, params)
    substitutions = pd.read_csv(args.substituicoes)
    selected = apply_substitutions(selected, pool, substitutions)

    selected[OUTPUT_COLUMNS].rename_axis("id").to_csv(
        OUTPUTS_DIR / "validacao_selecao.csv", encoding="utf-8-sig"
    )
    reserve[OUTPUT_COLUMNS].rename_axis("id").to_csv(
        OUTPUTS_DIR / "validacao_reserva.csv", encoding="utf-8-sig"
    )
    write_protocol(params, substitutions, selected, reserve)

    print(f"\nselecionadas={len(selected)} reserva={len(reserve)}")
    print("\nPeríodo × família:")
    print(pd.crosstab(selected["periodo"], selected["familia"], margins=True).to_string())
    print("\nMacrogênero:", selected["macro"].value_counts().to_dict())
    print(
        "Origem:",
        selected["origem"].value_counts().to_dict(),
        "| artistas distintos:",
        selected["artista"].nunique(),
    )


if __name__ == "__main__":
    main()
