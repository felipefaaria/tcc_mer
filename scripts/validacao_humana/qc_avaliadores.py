#!/usr/bin/env python
"""Controle de qualidade por avaliador do experimento com ouvintes (Seção 4.3 do artigo).

Sinaliza avaliadores por critérios objetivos sobre a qualidade da resposta (âncoras,
concordância com os demais, linha reta e pressa; ver :mod:`tcc_mer.rater_qc`), depois
refaz o teste principal retirando cada avaliador (jackknife) e retirando os sinalizados.

    python scripts/validacao_humana/qc_avaliadores.py --respostas "Tcc Anotação.xlsx"
"""

import argparse
from pathlib import Path

import pandas as pd

from tcc_mer.paths import OUTPUTS_DIR, VALIDATION_DIR
from tcc_mer.rater_qc import (
    MIN_TARGETS_FOR_FLAGS,
    flagged_raters,
    jackknife,
    main_test_without,
    rater_quality,
)
from tcc_mer.validation import DuelResult, load_item_map, load_responses

TABLE_COLUMNS = [
    "bloco",
    "n",
    "anc_V",
    "anc_A",
    "rho_V",
    "rho_A",
    "quad_ok",
    "neutra_V",
    "neutra_A",
    "modal_V",
    "modal_A",
    "sd_V",
    "sd_A",
    "rep_V_ident",
    "listen_med_s",
    "tempo_item_med_s",
]
FLAG_COLUMNS = ["flag_anc", "flag_conc", "flag_reta", "flag_pressa"]


def describe(result: DuelResult) -> str:
    return (
        f"n={result.n_tracks} fusão={result.first} áudio={result.second} -> {result.proportion:.0%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--respostas", required=True, type=Path)
    parser.add_argument("--mapa", type=Path, default=VALIDATION_DIR / "mapa_itens.csv")
    parser.add_argument("--excluir", nargs="*", default=["eu"])
    args = parser.parse_args()

    responses = load_responses(args.respostas, exclude=tuple(args.excluir))
    item_map = load_item_map(args.mapa)
    quality = rater_quality(responses, item_map)

    pd.set_option("display.width", 250)
    print("=== POR AVALIADOR ===")
    print(quality[TABLE_COLUMNS].to_string())

    print("\n=== SINALIZAÇÕES (anc<75% | rho<0,15 em V e A | linha reta>70% | <12 s/item) ===")
    flagged = quality[quality["n_flags"] > 0]
    print(
        flagged[["alvos", *FLAG_COLUMNS]].to_string()
        if len(flagged)
        else "nenhum avaliador sinalizado"
    )
    partial = quality[quality["alvos"] < MIN_TARGETS_FOR_FLAGS].index.tolist()
    print(f"avaliadores com <{MIN_TARGETS_FOR_FLAGS} alvos (parciais, pouca informação):", partial)

    print("\n=== TESTE PRINCIPAL SEM CADA AVALIADOR (jackknife) ===")
    print(f"todos: {describe(main_test_without(responses, item_map, []))}")
    removal = jackknife(responses, item_map)
    print(removal.round(2).to_string())
    print(
        f"faixa da proporção a favor da fusão: {removal['prop_fusao'].min():.0%} – "
        f"{removal['prop_fusao'].max():.0%}"
    )

    flagged_ids = flagged_raters(quality)
    fastest = quality["tempo_item_med_s"].idxmin()
    for label, excluded in (
        (f"SEM os sinalizados (com dados suficientes) {flagged_ids}", flagged_ids),
        (f"SEM os sinalizados e o avaliador mais rápido ({fastest})", [*flagged_ids, fastest]),
        (f"SEM os parciais {partial}", partial),
        ("SEM sinalizados e parciais", flagged_ids + partial),
    ):
        print(f"\n{label}:\n  {describe(main_test_without(responses, item_map, excluded))}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    quality.to_csv(OUTPUTS_DIR / "validacao_qc_avaliadores.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
