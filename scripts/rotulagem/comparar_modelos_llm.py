#!/usr/bin/env python
"""Compara modelos Claude como LF2, contra a condição ``letra`` do gold set (Seção 3.4).

Roda o mesmo prompt versionado (``prompts/lf2_zero_shot.md``) em cada modelo, sobre as mesmas
faixas do gold que têm letra, e mede o CCC de cada dimensão contra a média humana. A escolha
do modelo sai do CCC e do custo, e não de uma impressão sobre qual "parece melhor".

Freio de orçamento, em duas camadas: antes de gastar, estima o custo de toda a rodada com
``count_tokens`` e recusa rodar acima de ``--budget``; durante a rodada, soma o custo real e
para se passar do teto (o que já rodou vira resultado parcial).

    python scripts/rotulagem/comparar_modelos_llm.py --n 40 --dry-run      # só o custo
    python scripts/rotulagem/comparar_modelos_llm.py --n 40 --budget 2.0
    python scripts/rotulagem/comparar_modelos_llm.py --models claude-sonnet-5 --n 30
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_mer.anthropic_adapter import PRICES, estimate_cost, make_client
from tcc_mer.env import load_env
from tcc_mer.lf2_llm import build_prompt, estimate_va_from_lyrics_llm
from tcc_mer.metrics import ccc
from tcc_mer.paths import ANNOTATIONS_DIR, LYRICS_DIR, OUTPUTS_DIR

TAG = "[comparar]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(PRICES), help="modelos separados por vírgula")
    parser.add_argument("--n", type=int, default=15, help="faixas (com gold de letra) por modelo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--budget", type=float, default=2.0, help="teto em USD, somando os modelos")
    parser.add_argument("--lyrics-dir", type=Path, default=LYRICS_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "avaliacao_modelos_lf2.csv")
    parser.add_argument("--dry-run", action="store_true", help="só estima o custo")
    parser.add_argument("--yes", action="store_true", help="não pede confirmação antes de gastar")
    return parser.parse_args()


def load_gold_tracks(n: int, seed: int, lyrics_dir: Path) -> list[tuple[str, str, float, float]]:
    """``(id, letra, V_humano, A_humano)`` de ``n`` faixas do gold, sorteadas com ``seed``."""
    gold = pd.read_csv(ANNOTATIONS_DIR / "gold_wide.csv")
    usable = []
    for row in gold[gold["V_letra"].notna()].itertuples():
        path = lyrics_dir / f"{row.trackId}.txt"
        text = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""
        if text:
            usable.append((row.trackId, text, float(row.V_letra), float(row.A_letra)))
    if not usable:
        sys.exit(f"{TAG} nenhuma faixa com gold de letra e .txt em {lyrics_dir}")
    random.Random(seed).shuffle(usable)
    return usable[:n]


def evaluate_model(sdk, model: str, effort: str, tracks, budget: float) -> dict:
    """Roda o modelo nas faixas e devolve CCC, custo e tempo (para no meio se passar do teto)."""
    client, usage = make_client(sdk, model, effort)
    predicted, human = [], []
    started = time.monotonic()
    stopped_by_budget = False
    for track_id, text, v_human, a_human in tracks:
        try:
            result = estimate_va_from_lyrics_llm(text, client, model_name=model)
        except Exception as error:
            print(f"    {track_id}: erro ({error}); pulando")
            continue
        predicted.append((result.v_raw, result.a_raw))
        human.append((v_human, a_human))
        cost = usage.cost(model)
        print(
            f"    {track_id}: V={result.v_raw:+.2f} A={result.a_raw:+.2f} "
            f"(humano V={v_human:+.2f} A={a_human:+.2f})  custo ${cost:.4f}"
        )
        if cost > budget:
            print(f"    custo real passou o --budget (${budget:.2f}): parando aqui")
            stopped_by_budget = True
            break
    predicted, human = np.array(predicted).reshape(-1, 2), np.array(human).reshape(-1, 2)
    cost = usage.cost(model)
    return {
        "model": model,
        "n_faixas": len(predicted),
        "ccc_valencia": ccc(predicted[:, 0], human[:, 0]),
        "ccc_arousal": ccc(predicted[:, 1], human[:, 1]),
        "custo_total_usd": round(cost, 4),
        "custo_por_musica_usd": round(cost / max(len(predicted), 1), 5),
        "tempo_total_s": round(time.monotonic() - started, 1),
        "parou_por_orcamento": stopped_by_budget,
    }


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in PRICES]
    if unknown:
        sys.exit(f"{TAG} modelo(s) desconhecido(s): {unknown} (conhecidos: {list(PRICES)})")
    load_env()
    import anthropic

    sdk = anthropic.Anthropic()
    tracks = load_gold_tracks(args.n, args.seed, args.lyrics_dir)
    print(f"{TAG} {len(tracks)} faixa(s) com gold de letra (seed={args.seed})")

    prompts = [build_prompt(text) for _, text, _, _ in tracks]
    estimates = {model: estimate_cost(sdk, model, prompts) for model in models}
    for model, cost in estimates.items():
        print(f"    {model}: ~${cost:.4f}")
    total = sum(estimates.values())
    print(f"{TAG} total estimado: ~${total:.4f} (orçamento: ${args.budget:.2f})")
    if total > args.budget:
        sys.exit(
            f"{TAG} abortado: a estimativa passa do --budget. Reduza --n/--models ou suba o teto."
        )
    if args.dry_run:
        return
    if not args.yes and input(f"Gastar até ~${total:.4f}? [s/N] ").strip().lower() != "s":
        sys.exit(f"{TAG} abortado.")

    results = []
    for model in models:
        print(f"\n{TAG} === {model} (effort={args.effort}) ===")
        results.append(evaluate_model(sdk, model, args.effort, tracks, args.budget))

    table = pd.DataFrame(results)
    print(f"\n{TAG} === RESULTADO ===")
    print(table.round(3).to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\n{TAG} salvo em {args.out}")
    print(f"{TAG} leitura: escolha o modelo mais barato que não perca CCC de forma relevante.")


if __name__ == "__main__":
    main()
