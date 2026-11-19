#!/usr/bin/env python
"""Análise do experimento de validação com ouvintes (Seção 4.3 do artigo).

Regras fixadas antes de olhar os resultados:

* rótulo humano da faixa = mediana, entre avaliadores, de cada dimensão (categorias 1 a
  5 -> -2 a +2); o quadrante vem do sinal, e mediana neutra deixa a faixa sem quadrante;
* só entram faixas-alvo com ``--min-avaliadores`` ou mais avaliações; âncoras e
  repetições são controle de qualidade;
* o pesquisador (``--excluir``, padrão ``eu``) fica de fora;
* **teste principal**: nas faixas em que áudio e fusão simples dão quadrantes diferentes,
  conta quando o quadrante humano coincide com o de cada um (binomial exato bicaudal e
  IC de Clopper-Pearson para a proporção a favor da fusão).

Também reporta, como análises secundárias, a concordância entre avaliadores, os
controles de qualidade, as famílias de transição, as notas contínuas e, como
**exploratória**, a fusão por dimensão (valência da letra, arousal do áudio).

    python scripts/validacao_humana/analisar_experimento.py --respostas "Tcc Anotação.xlsx"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tcc_mer.metrics import ccc
from tcc_mer.paths import OUTPUTS_DIR, VALIDATION_DIR
from tcc_mer.validation import (
    MIN_RATERS,
    DuelResult,
    analyzable,
    anchor_accuracy,
    duel,
    load_item_map,
    load_responses,
    ordinal_alpha,
    pairwise_quadrant_agreement,
    repetition_agreement,
    track_labels,
)


def print_quality(responses: pd.DataFrame, item_map: pd.DataFrame) -> None:
    print("\n=== QUALIDADE ===")
    anchors = anchor_accuracy(responses, item_map)
    print(
        f"âncoras: sinal da valência correto {anchors['valence_ok'].mean():.0%} | sinal do "
        f"arousal correto {anchors['arousal_ok'].mean():.0%} | quadrante exato "
        f"{anchors['exact_ok'].mean():.0%} (n={len(anchors)})"
    )
    print(anchors.groupby("item")[["valence_ok", "arousal_ok"]].mean().round(2).to_string())
    by_rater = anchors.groupby("rater")["valence_ok"].mean().round(2)
    poor = by_rater[by_rater < 0.5]
    if len(poor):
        print(
            "avaliadores com <50% de acerto de sinal de valência nas âncoras:", poor.index.tolist()
        )

    pairs = repetition_agreement(responses)
    if len(pairs):
        v_diff = (pairs["V_first"] - pairs["V_repeat"]).abs()
        a_diff = (pairs["A_first"] - pairs["A_repeat"]).abs()
        print(
            f"repetições (n={len(pairs)}): valência idêntica {(v_diff == 0).mean():.0%}, "
            f"a ±1 cat {(v_diff <= 1).mean():.0%} | arousal idêntico {(a_diff == 0).mean():.0%}, "
            f"a ±1 cat {(a_diff <= 1).mean():.0%}"
        )
    print(
        f"tempo de escuta: mediana {responses['listenMs'].median() / 1000:.0f}s | "
        f"<10s: {(responses['listenMs'] < 10000).sum()}"
    )
    print(
        "conhecia a música:", responses["conhece"].value_counts(normalize=True).round(2).to_dict()
    )


def print_agreement(responses: pd.DataFrame) -> None:
    print("\n=== CONCORDÂNCIA ENTRE AVALIADORES (alfa ordinal, só faixas-alvo) ===")
    for dimension in ("V", "A"):
        alpha, (n_raters, n_tracks) = ordinal_alpha(responses, dimension)
        print(f"{dimension}: alpha = {alpha:.2f}  ({n_raters} avaliadores x {n_tracks} faixas)")
    print(
        f"concordância par-a-par de quadrante: {pairwise_quadrant_agreement(responses):.0%} "
        f"(acaso ~25–30%)"
    )


def print_duel(title: str, result: DuelResult, first: str, second: str) -> None:
    print(f"{title}: {first} {result.first} | {second} {result.second} | nenhum {result.neither}")
    if result.first + result.second:
        decided = result.first + result.second
        print(
            f"  proporção a favor de {first}: {result.first}/{decided} = {result.proportion:.0%} "
            f"(IC95% Clopper-Pearson {result.ci_low:.0%}–{result.ci_high:.0%}; "
            f"binomial exato p = {result.p_value:.4f})"
        )


def print_main_test(labels: pd.DataFrame, min_raters: int) -> dict:
    print("\n=== TESTE PRINCIPAL: onde áudio e fusão discordam, quem concorda com o humano? ===")
    enough = labels[labels["n"] >= min_raters]
    kept = analyzable(labels, min_raters)
    result = duel(kept)
    print(
        f"faixas analisáveis: {len(kept)} (sem quadrante humano por 'neutra': "
        f"{len(enough) - len(kept)})"
    )
    print_duel("teste principal", result, "fusão", "áudio")
    print(
        f"acerto de quadrante sobre todas as analisáveis: fusão "
        f"{(kept['q_simple'] == kept['q_human']).mean():.0%} | áudio "
        f"{(kept['q_lf1'] == kept['q_human']).mean():.0%}"
    )

    for column, title in (("familia", "por família de transição"), ("periodo", "por período")):
        print(f"\n--- {title} (exploratório) ---")
        for name, group in kept.groupby(column):
            fusion_hits = int((group["q_simple"] == group["q_human"]).sum())
            audio_hits = int((group["q_lf1"] == group["q_human"]).sum())
            print(
                f"{name:18s} n={len(group):2d}  fusão={fusion_hits:2d}  áudio={audio_hits:2d}  "
                f"nenhum={len(group) - fusion_hits - audio_hits}"
            )

    print("\n--- EXPLORATÓRIO: fusão por dimensão (valência da letra, arousal do áudio) ---")
    print(
        f"acertos de quadrante: áudio {(kept['q_lf1'] == kept['q_human']).sum()} | "
        f"fusão simples {(kept['q_simple'] == kept['q_human']).sum()} | "
        f"fusão por dimensão {(kept['q_dimension'] == kept['q_human']).sum()} de {len(kept)}"
    )
    print_duel("contra o áudio", duel(kept, "q_dimension", "q_lf1"), "fusão por dimensão", "áudio")
    print_duel(
        "contra a fusão simples",
        duel(kept, "q_dimension", "q_simple"),
        "fusão por dimensão",
        "fusão simples",
    )

    return {
        "n_faixas": len(kept),
        "fusao": result.first,
        "audio": result.second,
        "nenhum": result.neither,
        "sem_quadrante": len(enough) - len(kept),
        "prop_fusao": result.proportion,
        "ic_low": result.ci_low,
        "ic_high": result.ci_high,
        "p": result.p_value,
    }


def print_continuous(labels: pd.DataFrame, min_raters: int) -> None:
    print("\n=== SECUNDÁRIA: humano (média) x modelo, em escala -1..+1 ===")
    enough = labels[labels["n"] >= min_raters]
    human = {"V": enough["V_mean"] / 2, "A": enough["A_mean"] / 2}
    names = {"V": "Valência", "A": "Arousal"}
    for dimension in ("V", "A"):
        model_columns = [
            ("áudio LF1", f"{dimension}1"),
            ("letra LF2", f"{dimension}3"),
            ("fusão", f"{dimension}f"),
        ]
        for name, column in model_columns:
            rho = stats.spearmanr(human[dimension], enough[column]).statistic
            pearson = np.corrcoef(human[dimension], enough[column])[0, 1]
            print(
                f"{names[dimension]:9s} {name:10s} Spearman={rho:+.2f}  Pearson={pearson:+.2f}  "
                f"CCC={ccc(human[dimension], enough[column]):+.2f}"
            )

    for dimension, audio, fusion in (("V", "V1", "Vf"), ("A", "A1", "Af")):
        audio_error = (enough[audio] - human[dimension]).abs()
        fusion_error = (enough[fusion] - human[dimension]).abs()
        diff = (audio_error - fusion_error).to_numpy()
        boot = np.random.default_rng(0).choice(diff, (5000, len(diff))).mean(1)
        print(
            f"{dimension}: erro abs. médio áudio={audio_error.mean():.3f} "
            f"fusão={fusion_error.mean():.3f}  (áudio-fusão={diff.mean():+.3f}, "
            f"IC95% [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]; "
            f"positivo = fusão melhor)"
        )


def print_sensitivity(responses: pd.DataFrame, labels: pd.DataFrame, min_raters: int) -> None:
    print("\n=== SENSIBILIDADE ===")
    for stricter in (4, 5):
        subset = analyzable(labels, stricter)
        if len(subset):
            result = duel(subset)
            print(
                f">= {stricter} avaliadores: n={len(subset)} fusão={result.first} "
                f"áudio={result.second}"
            )
    targets = responses[responses["tipo"] == "alvo"].copy()
    targets["item"] = targets["item"].astype(str)
    unknown = targets.groupby("item")["conhece"].apply(lambda s: (s == "nao").mean())
    unfamiliar = labels[labels["n"] >= min_raters]
    unfamiliar = unfamiliar[unknown.reindex(unfamiliar.index) >= 0.5]
    subset = unfamiliar[unfamiliar["q_human"] != ""]
    if len(subset):
        result = duel(subset)
        print(
            f"faixas desconhecidas por >=50% dos avaliadores: n={len(subset)} "
            f"fusão={result.first} áudio={result.second}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--respostas", required=True, type=Path)
    parser.add_argument("--mapa", type=Path, default=VALIDATION_DIR / "mapa_itens.csv")
    parser.add_argument("--excluir", nargs="*", default=["eu"])
    parser.add_argument("--min-avaliadores", type=int, default=MIN_RATERS)
    args = parser.parse_args()

    everyone = load_responses(args.respostas, exclude=())
    responses = load_responses(args.respostas, exclude=tuple(args.excluir))
    item_map = load_item_map(args.mapa)
    print(
        f"avaliadores: {responses['rater'].nunique()} (excluídos: {args.excluir}; "
        f"antes: {everyone['rater'].nunique()}) | respostas: {len(responses)}"
    )
    print(
        "respostas por avaliador:",
        responses.groupby("rater").size().sort_values(ascending=False).to_dict(),
    )

    labels = track_labels(responses, item_map)
    enough = labels[labels["n"] >= args.min_avaliadores]
    print_quality(responses, item_map)
    print(
        f"\nfaixas-alvo com avaliações: {len(labels)} | com >= {args.min_avaliadores} "
        f"avaliadores: {len(enough)} | avaliadores por faixa: mediana {enough['n'].median():.0f}, "
        f"min {enough['n'].min()}, max {enough['n'].max()}"
    )
    print_agreement(responses)
    results = print_main_test(labels, args.min_avaliadores)
    print_continuous(labels, args.min_avaliadores)
    print_sensitivity(responses, labels, args.min_avaliadores)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    labels.round(3).to_csv(OUTPUTS_DIR / "validacao_rotulos_humanos.csv", encoding="utf-8-sig")
    (OUTPUTS_DIR / "validacao_resultados.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsalvo em {OUTPUTS_DIR}: validacao_rotulos_humanos.csv e validacao_resultados.json")


if __name__ == "__main__":
    main()
