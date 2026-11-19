#!/usr/bin/env python
"""Fusão calibrada no gold set: análise exploratória abandonada (Seção 3.7 do artigo).

O artigo usa rótulos brutos. Aqui se reproduz a alternativa que foi explorada e
descartada, em duas partes:

1. **CCC fora da amostra** (k-fold aninhado: a calibração afim e os pesos de fusão são
   ajustados só no treino) de cada canal contra a condição correspondente do gold set;
2. **concordância de quadrante** nas faixas do experimento com ouvintes: a calibração e a
   fusão ajustadas no gold set inteiro são aplicadas a essas faixas (nenhuma está no
   gold) e comparadas com o áudio e a fusão simples.

    python scripts/exploratorio/fusao_calibrada.py --respostas "Tcc Anotação.xlsx"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from tcc_mer.dataset import DATASET_PATH, read_dataset
from tcc_mer.exploratory.pipeline import apply_channel, fit_channel, nested_kfold_ccc
from tcc_mer.paths import ANNOTATIONS_DIR, VALIDATION_DIR
from tcc_mer.quadrant import quadrants
from tcc_mer.validation import analyzable, duel, load_item_map, load_responses, track_labels

FOLDS = 5
SEED = 0

# (rótulo, colunas das LFs, coluna do gold, dimensão)
CROSS_VALIDATION_PLAN = [
    ("LF1 áudio (melodia)", ["lf1_V"], "V_audio"),
    ("LF1 áudio (melodia)", ["lf1_A"], "A_audio"),
    ("LF2 letra (LLM)", ["lf2_V"], "V_letra"),
    ("LF2 letra (LLM)", ["lf2_A"], "A_letra"),
    ("Fusão LF1+LF2 (completa)", ["lf1_V", "lf2_V"], "V_completa"),
    ("Fusão LF1+LF2 (completa)", ["lf1_A", "lf2_A"], "A_completa"),
]


def load_gold_with_labels(gold_path: Path, dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gold set e rótulos brutos de todas as faixas (colunas ``lf1_*`` e ``lf2_*``)."""
    model = read_dataset(dataset_path).set_index("id")
    model = model.rename(
        columns={
            "lf1_audio_valencia": "lf1_V",
            "lf1_audio_arousal": "lf1_A",
            "lf2_llm_valencia": "lf2_V",
            "lf2_llm_arousal": "lf2_A",
        }
    )[["lf1_V", "lf1_A", "lf2_V", "lf2_A"]]
    gold = pd.read_csv(gold_path).set_index("trackId").join(model)
    return gold, model


def print_cross_validation(gold: pd.DataFrame) -> None:
    print("=== CCC FORA DA AMOSTRA (k-fold aninhado) ===")
    print(f"{'canal':28s} {'dim':4s} {'CCC_oos':>8s} {'n':>4s}")
    for name, columns, target in CROSS_VALIDATION_PLAN:
        ccc_oos, weights, n = nested_kfold_ccc(
            gold[columns].to_numpy(float), gold[target].to_numpy(float), k=FOLDS, seed=SEED
        )
        shown = "  pesos=" + "/".join(f"{w:.2f}" for w in weights) if len(columns) > 1 else ""
        print(f"{name:28s} {target[0]:4s} {ccc_oos:>8.2f} {n:>4d}{shown}")


def calibrated_quadrants(
    gold: pd.DataFrame, model: pd.DataFrame, track_ids: pd.Index
) -> pd.DataFrame:
    """Quadrantes do áudio calibrado e da fusão calibrada, ajustados no gold set inteiro."""
    fitted = {}
    for dimension in ("V", "A"):
        audio = fit_channel(
            gold[[f"lf1_{dimension}"]].to_numpy(float), gold[f"{dimension}_audio"].to_numpy(float)
        )
        fusion = fit_channel(
            gold[[f"lf1_{dimension}", f"lf2_{dimension}"]].to_numpy(float),
            gold[f"{dimension}_completa"].to_numpy(float),
        )
        tracks = model.loc[track_ids]
        fitted[f"audio_{dimension}"] = apply_channel(
            *audio, tracks[[f"lf1_{dimension}"]].to_numpy(float)
        )
        fitted[f"fusion_{dimension}"] = apply_channel(
            *fusion, tracks[[f"lf1_{dimension}", f"lf2_{dimension}"]].to_numpy(float)
        )
    return pd.DataFrame(
        {
            "q_cal_audio": quadrants(fitted["audio_V"], fitted["audio_A"]),
            "q_cal_fusion": quadrants(fitted["fusion_V"], fitted["fusion_A"]),
        },
        index=track_ids,
    )


def print_duel(title: str, labels: pd.DataFrame, first: str, second: str) -> None:
    result = duel(labels, first, second)
    print(
        f"   {title}: {result.first} x {result.second} -> {result.proportion:.0%} "
        f"(IC95% {result.ci_low:.0%}–{result.ci_high:.0%}, p={result.p_value:.4f})"
    )


def print_experiment_agreement(
    gold: pd.DataFrame, model: pd.DataFrame, responses: pd.DataFrame, item_map: pd.DataFrame
) -> None:
    labels = analyzable(track_labels(responses, item_map))
    if set(labels["trackId"]) & set(gold.index):
        sys.exit(
            "[exploratório] há faixas do experimento no gold set: a avaliação não seria "
            "fora da amostra"
        )
    labels = labels.join(
        calibrated_quadrants(gold, model, pd.Index(labels["trackId"])), on="trackId"
    )
    print(f"\n=== CONCORDÂNCIA DE QUADRANTE NAS {len(labels)} FAIXAS DO EXPERIMENTO ===")
    for name, column in (
        ("áudio bruto", "q_lf1"),
        ("fusão simples bruta", "q_simple"),
        ("áudio calibrado", "q_cal_audio"),
        ("fusão calibrada", "q_cal_fusion"),
    ):
        hits = labels[column] == labels["q_human"]
        print(f"   {name:22s} {hits.sum():2d}/{len(labels)} = {hits.mean():.0%}")
    print_duel("fusão calibrada contra áudio bruto", labels, "q_cal_fusion", "q_lf1")
    print_duel("fusão calibrada contra áudio calibrado", labels, "q_cal_fusion", "q_cal_audio")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--respostas", required=True, type=Path)
    parser.add_argument("--mapa", type=Path, default=VALIDATION_DIR / "mapa_itens.csv")
    parser.add_argument("--gold", type=Path, default=ANNOTATIONS_DIR / "gold_wide.csv")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--excluir", nargs="*", default=["eu"])
    args = parser.parse_args()
    if not args.gold.exists():
        sys.exit(
            f"[exploratório] gold não encontrado: {args.gold} (rode scripts/analise/gerar_gold.py)"
        )

    gold, model = load_gold_with_labels(args.gold, args.dataset)
    print_cross_validation(gold)

    responses = load_responses(args.respostas, exclude=tuple(args.excluir))
    print_experiment_agreement(gold, model, responses, load_item_map(args.mapa))


if __name__ == "__main__":
    main()
