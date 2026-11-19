#!/usr/bin/env python
"""Robustez do teste principal do experimento com ouvintes (Seção 4.3 e Apêndice A).

1. retirada de um avaliador por vez (proporção a favor da fusão e valor de p);
2. regra alternativa de quadrante (pós-hoc): média em vez de mediana, sem excluir
   faixas neutras;
3. histórico das análises parciais da coleta.

  python scripts/validacao_humana/robustez.py --respostas "Tcc Anotação.xlsx"
"""

import argparse
from pathlib import Path

from scipy import stats

from tcc_mer.paths import VALIDATION_DIR
from tcc_mer.rater_qc import jackknife, main_test_without
from tcc_mer.validation import analyzable, duel, load_item_map, load_responses, track_labels

# Análises feitas durante a coleta, com a mesma regra: (nome, avaliadores, fusão, áudio).
# Os dados intermediários não são versionados; os placares ficam registrados aqui.
PARTIAL_ANALYSES = [
    ("1ª análise", 11, 20, 11),
    ("2ª análise", 13, 18, 12),
    ("3ª análise", 16, 20, 8),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--respostas", required=True, type=Path)
    parser.add_argument("--mapa", type=Path, default=VALIDATION_DIR / "mapa_itens.csv")
    parser.add_argument("--excluir", nargs="*", default=["eu"])
    args = parser.parse_args()

    responses = load_responses(args.respostas, exclude=tuple(args.excluir))
    item_map = load_item_map(args.mapa)
    base = main_test_without(responses, item_map, [])
    print(
        f"BASE: avaliadores={responses['rater'].nunique()} | faixas analisáveis={base.n_tracks} "
        f"| fusão={base.first} áudio={base.second} -> {base.proportion:.0%} "
        f"(IC95% {base.ci_low:.0%}–{base.ci_high:.0%}, p={base.p_value:.3f})"
    )

    print("\n1) Sem cada avaliador:")
    removal = jackknife(responses, item_map)
    print(removal[["fusao", "audio", "prop_fusao", "p"]].round(3).to_string())
    print(
        f"proporção: {removal['prop_fusao'].min():.0%}–{removal['prop_fusao'].max():.0%} | "
        f"p<0,05 em {(removal['p'] < 0.05).sum()} de {len(removal)} retiradas"
    )

    print("\n2) Regra alternativa pós-hoc (média, sem excluir neutras):")
    mean_result = duel(analyzable(track_labels(responses, item_map, aggregate="mean")))
    print(
        f"   n={mean_result.n_tracks} fusão={mean_result.first} áudio={mean_result.second} -> "
        f"{mean_result.proportion:.0%} (IC95% {mean_result.ci_low:.0%}–{mean_result.ci_high:.0%}, "
        f"p={mean_result.p_value:.3f})"
    )

    print("\n3) Histórico de análises (mesma regra):")
    for name, n_raters, fusion, audio in PARTIAL_ANALYSES:
        p_value = stats.binomtest(fusion, fusion + audio, 0.5).pvalue
        print(f"   {name}: {n_raters} avaliadores, fusão {fusion} x áudio {audio}, p={p_value:.3f}")
    print(
        f"   análise atual: {responses['rater'].nunique()} avaliadores, fusão {base.first} x "
        f"áudio {base.second}, p={base.p_value:.3f}"
    )


if __name__ == "__main__":
    main()
