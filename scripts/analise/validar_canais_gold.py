#!/usr/bin/env python
"""Rótulos brutos de cada canal e dissociação estimada contra o gold set (Seções 4.1 e 4.2).

Compara, sem calibração, cada canal com a condição correspondente do gold set:

* LF1 (áudio) contra a condição ``melodia`` (áudio sem a voz) e contra ``completa``;
* LF2 (letra, LLM) contra a condição ``letra``;
* o léxico ANEW-Br (fonte descartada) contra a condição ``letra``, se houver saída;
* a dissociação estimada ``D = LF2 - LF1`` contra a humana ``D = letra - melodia``.

Para cada par reporta Spearman (com IC de 95% por bootstrap), CCC e a fração de faixas
com o mesmo sinal. Os valores dos modelos vêm da tabela ``dataset/corpus_anotado.csv``.
O gold set (``gold_wide.csv``, de ``scripts/analise/gerar_gold.py``) não é versionado.

    python scripts/analise/validar_canais_gold.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tcc_mer.dataset import DATASET_PATH, read_dataset
from tcc_mer.metrics import bootstrap_ci, ccc
from tcc_mer.paths import ANNOTATIONS_DIR, OUTPUTS_DIR

N_BOOT = 3000
SEED = 0


def compare(name: str, estimate: pd.Series, human: pd.Series, rng: np.random.Generator) -> None:
    """Imprime Spearman (com IC), CCC e acordo de sinal entre ``estimate`` e ``human``."""
    pairs = pd.DataFrame({"estimate": estimate, "human": human}).dropna()
    x, y = pairs["estimate"].to_numpy(), pairs["human"].to_numpy()
    rho = stats.spearmanr(x, y).statistic
    low, high = bootstrap_ci(
        pairs.to_numpy(),
        lambda m: stats.spearmanr(m[:, 0], m[:, 1]).statistic,
        n_boot=N_BOOT,
        seed=rng,
    )
    same_sign = (np.sign(x) == np.sign(y)).mean()
    print(
        f"{name:46s} n={len(pairs):2d}  Spearman={rho:+.2f} [{low:+.2f},{high:+.2f}]  "
        f"CCC={ccc(x, y):+.2f}  mesmo sinal={same_sign:.0%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=ANNOTATIONS_DIR / "gold_wide.csv")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--lexicon", type=Path, default=OUTPUTS_DIR / "lexicon_raw.csv")
    args = parser.parse_args()
    if not args.gold.exists():
        sys.exit(f"[gold] não encontrado: {args.gold} (rode scripts/analise/gerar_gold.py)")

    model = read_dataset(args.dataset).set_index("id")
    gold = pd.read_csv(args.gold).set_index("trackId")
    gold["lf1_V"], gold["lf1_A"] = model["lf1_audio_valencia"], model["lf1_audio_arousal"]
    gold["lf2_V"], gold["lf2_A"] = model["lf2_llm_valencia"], model["lf2_llm_arousal"]
    has_lexicon = args.lexicon.exists()
    if has_lexicon:
        lexicon = pd.read_csv(args.lexicon).set_index("id")
        gold["lex_V"], gold["lex_A"] = lexicon["V_raw"], lexicon["A_raw"]

    rng = np.random.default_rng(SEED)
    print("=== Canais contra o gold set, sem calibração (valores brutos em [-1, 1]) ===")
    compare("LF1 áudio x humano melodia - valência", gold["lf1_V"], gold["V_audio"], rng)
    compare("LF1 áudio x humano melodia - arousal", gold["lf1_A"], gold["A_audio"], rng)
    compare("LF1 áudio x humano completa - valência", gold["lf1_V"], gold["V_completa"], rng)
    compare("LF1 áudio x humano completa - arousal", gold["lf1_A"], gold["A_completa"], rng)
    compare("LF2 LLM x humano letra - valência", gold["lf2_V"], gold["V_letra"], rng)
    compare("LF2 LLM x humano letra - arousal", gold["lf2_A"], gold["A_letra"], rng)
    if has_lexicon:
        compare(
            "[descartado] léxico x humano letra - valência", gold["lex_V"], gold["V_letra"], rng
        )
        compare("[descartado] léxico x humano letra - arousal", gold["lex_A"], gold["A_letra"], rng)

    print("\n=== Dissociação estimada (LF2 - LF1) contra a humana (letra - melodia) ===")
    for dimension, name in (("V", "valência"), ("A", "arousal")):
        estimated = gold[f"lf2_{dimension}"] - gold[f"lf1_{dimension}"]
        human = gold[f"{dimension}_letra"] - gold[f"{dimension}_audio"]
        compare(f"D {name}", estimated, human, rng)
        pairs = pd.DataFrame({"e": estimated, "h": human}).dropna()
        slope = np.polyfit(pairs["e"], pairs["h"], 1)[0]
        print(
            f"    inclinação (D humano sobre D estimado) = {slope:.2f} | "
            f"média estimada {pairs['e'].mean():+.2f}, média humana {pairs['h'].mean():+.2f}"
        )


if __name__ == "__main__":
    main()
