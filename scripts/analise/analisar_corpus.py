#!/usr/bin/env python
"""Dissociação áudio-letra no corpus inteiro (Seção 4.2 do artigo).

Usa só a tabela ``dataset/corpus_anotado.csv``, sem nenhum dado privado. Calcula, nas
faixas com os dois canais: a frequência de quadrantes e de sinais diferentes entre
áudio e letra, o efeito das fusões sobre o quadrante do áudio e as faixas de maior
dissociação.

    python scripts/analise/analisar_corpus.py [--top 6]
"""

import argparse

import numpy as np
import pandas as pd

from tcc_mer.dataset import DATASET_PATH, read_dataset


def percent(mask: pd.Series) -> str:
    return f"{mask.sum()} ({mask.mean():.1%})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--top", type=int, default=6, help="faixas de maior |D_V| a listar")
    args = parser.parse_args()

    table = read_dataset(args.dataset)
    both = table[table["fusao_simples_valencia"].notna()]
    print(
        f"faixas com os dois canais: {len(both)} de {len(table)} "
        f"({int(table['instrumental'].sum())} instrumentais)\n"
    )

    print("=== Áudio contra letra ===")
    print("quadrantes diferentes:        ", percent(both["quadrante_lf1"] != both["quadrante_lf2"]))
    print(
        "sinal de valência oposto:     ",
        percent(np.sign(both["lf1_audio_valencia"]) != np.sign(both["lf2_llm_valencia"])),
    )
    print(
        "sinal de arousal oposto:      ",
        percent(np.sign(both["lf1_audio_arousal"]) != np.sign(both["lf2_llm_arousal"])),
    )
    print(
        "em Q2 (tensão), áudio / letra:",
        f"{(both['quadrante_lf1'] == 'Q2').sum()} / {(both['quadrante_lf2'] == 'Q2').sum()}",
    )
    print(
        f"média de D_V = {both['dissociacao_valencia'].mean():+.2f} | "
        f"média de D_A = {both['dissociacao_arousal'].mean():+.2f}"
    )

    print("\n=== Efeito das fusões sobre o rótulo do áudio ===")
    print(
        "fusão simples muda o quadrante:     ",
        percent(both["quadrante_fusao_simples"] != both["quadrante_lf1"]),
    )
    print(
        "fusão simples muda o sinal de V:    ",
        percent(np.sign(both["fusao_simples_valencia"]) != np.sign(both["lf1_audio_valencia"])),
    )
    print(
        "fusão por dimensão muda o quadrante:",
        percent(both["quadrante_fusao_dimensao"] != both["quadrante_lf1"]),
    )
    print(
        "fusões com quadrantes diferentes:   ",
        percent(both["quadrante_fusao_simples"] != both["quadrante_fusao_dimensao"]),
    )

    print("\n=== Emoção por fusão ===")
    emotions = pd.DataFrame(
        {
            "fusão simples": both["emocao_fusao_simples"].value_counts(),
            "fusão por dimensão": both["emocao_fusao_dimensao"].value_counts(),
        }
    )
    print(emotions.to_string())

    print(f"\n=== {args.top} maiores |D_V| (letra mais triste ou mais alegre que o áudio) ===")
    top = both.reindex(both["dissociacao_valencia"].abs().sort_values(ascending=False).index)
    columns = [
        "id",
        "ano",
        "titulo",
        "interprete",
        "lf1_audio_valencia",
        "lf2_llm_valencia",
        "dissociacao_valencia",
    ]
    print(top.head(args.top)[columns].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
