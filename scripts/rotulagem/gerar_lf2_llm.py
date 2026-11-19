#!/usr/bin/env python
"""LF2: gera ``lf2_raw.csv`` com valência e arousal da letra, estimados por um LLM zero-shot.

Lê a letra completa de cada faixa em ``data/corpus/lyrics/<id>.txt``, envia ao modelo com
o prompt versionado (``prompts/lf2_zero_shot.md``) e grava:

* ``data/outputs/lf2_raw.csv``: ``id, V_raw, A_raw`` em [-1, 1];
* ``data/outputs/lf2_meta.json``: modelo, esforço, versão do prompt, contagens e custo.

Faixa instrumental (marcada no manifesto, ou com letra vazia) não consome chamada e fica
``NaN``. O script retoma de onde parou: pula as faixas que já têm valor. Para refazer
tudo (outro modelo ou outro prompt), use ``--force``, que evita misturar instrumentos
de medição na mesma coluna.

A chave da API vem de ``ANTHROPIC_API_KEY``, do ambiente ou de um ``.env`` na raiz.

Freio de orçamento: antes de gastar, estima o custo com ``count_tokens`` e recusa rodar
acima de ``--budget``; durante a rodada, soma o custo real e para se passar do teto.

    python scripts/rotulagem/gerar_lf2_llm.py --dry-run     # só a estimativa de custo
    python scripts/rotulagem/gerar_lf2_llm.py               # roda o que falta
    python scripts/rotulagem/gerar_lf2_llm.py --force       # recalcula tudo
    python scripts/rotulagem/gerar_lf2_llm.py --ids 1901-01 1901-02 --out teste.csv
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tcc_mer.anthropic_adapter import PRICES, estimate_cost, make_client
from tcc_mer.dataset import load_metadata
from tcc_mer.env import load_env
from tcc_mer.lf2_llm import PROMPT_VERSION, build_prompt, estimate_va_from_lyrics_llm
from tcc_mer.lf_common import LFRawOutput, write_lf_raw
from tcc_mer.paths import LYRICS_DIR, OUTPUTS_DIR

TAG = "[lf2]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lyrics-dir", type=Path, default=LYRICS_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "lf2_raw.csv")
    parser.add_argument("--model", default="claude-sonnet-5", choices=list(PRICES))
    parser.add_argument("--effort", default="low", help="output_config.effort (low basta aqui)")
    parser.add_argument("--budget", type=float, default=2.0, help="teto em USD para a rodada")
    parser.add_argument("--sleep", type=float, default=0.3, help="pausa (s) entre chamadas")
    parser.add_argument(
        "--force",
        action="store_true",
        help="recalcula todas as faixas, ignorando os valores existentes",
    )
    parser.add_argument("--ids", nargs="*", help="processa só estas faixas (teste rápido)")
    parser.add_argument("--dry-run", action="store_true", help="só estima o custo")
    parser.add_argument("--yes", action="store_true", help="não pede confirmação antes de gastar")
    return parser.parse_args()


def discover_lyrics(lyrics_dir: Path) -> list[tuple[str, str, bool]]:
    """``(id, texto, tem_letra)`` de cada faixa.

    O manifesto manda: faixa instrumental não tem letra mesmo que o arquivo exista (alguns
    têm um texto de preenchimento).
    """
    if not lyrics_dir.exists():
        sys.exit(f"{TAG} pasta de letras não existe: {lyrics_dir}")
    metadata = load_metadata()
    instrumental = set(metadata.loc[metadata["instrumental"], "id"])
    tracks = []
    for path in sorted(lyrics_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        tracks.append((path.stem, text, bool(text) and path.stem not in instrumental))
    return tracks


def load_done(path: Path, force: bool) -> dict[str, tuple[float, float]]:
    """``{id: (V, A)}`` das faixas já rotuladas no csv de saída."""
    if force or not path.exists():
        return {}
    df = pd.read_csv(path).dropna(subset=["V_raw"])
    return {str(row.id): (row.V_raw, row.A_raw) for row in df.itertuples()}


def save(path: Path, tracks: list[tuple[str, str, bool]], values: dict) -> None:
    nan = float("nan")
    write_lf_raw(
        [LFRawOutput(track_id, *values.get(track_id, (nan, nan))) for track_id, _, _ in tracks],
        path,
    )


def main() -> None:
    args = parse_args()
    load_env()
    import anthropic

    sdk = anthropic.Anthropic()
    tracks = discover_lyrics(args.lyrics_dir)
    if not tracks:
        sys.exit(f"{TAG} nenhuma letra .txt em {args.lyrics_dir}")

    values = load_done(args.out, args.force)
    pending = [t for t in tracks if t[2] and t[0] not in values]
    if args.ids:
        pending = [t for t in pending if t[0] in args.ids]
    n_instrumental = sum(not has_lyrics for _, _, has_lyrics in tracks)
    print(
        f"{TAG} {len(pending)} faixa(s) a processar com {args.model} (de {len(tracks)}; "
        f"{n_instrumental} instrumentais; {len(values)} já prontas)"
    )
    if not pending:
        save(args.out, tracks, values)
        return

    estimated = estimate_cost(sdk, args.model, (build_prompt(text) for _, text, _ in pending))
    print(f"{TAG} custo estimado: ~${estimated:.4f} (orçamento: ${args.budget:.2f})")
    if estimated > args.budget:
        sys.exit(f"{TAG} abortado: a estimativa passa do --budget. Suba o teto ou rode em partes.")
    if args.dry_run:
        return
    if not args.yes and input(f"Gastar até ~${estimated:.4f}? [s/N] ").strip().lower() != "s":
        sys.exit(f"{TAG} abortado.")

    client, usage = make_client(sdk, args.model, args.effort)
    over_budget = False
    for track_id, text, _ in pending:
        try:
            result = estimate_va_from_lyrics_llm(text, client, model_name=args.model)
        except Exception:
            save(args.out, tracks, values)
            raise
        if pd.isna(result.v_raw):
            print(
                f"  {track_id}: erro ({result.meta.get('error', '?')[:60]}); tente de novo depois"
            )
        else:
            values[track_id] = (result.v_raw, result.a_raw)
            print(
                f"  {track_id}: V={result.v_raw:+.2f} A={result.a_raw:+.2f}  "
                f"({len(values)} ok, custo ${usage.cost(args.model):.4f})"
            )
            save(args.out, tracks, values)
        if usage.cost(args.model) > args.budget:
            print(f"{TAG} custo real passou o --budget: parando aqui.")
            over_budget = True
            break
        time.sleep(args.sleep)

    save(args.out, tracks, values)
    missing = [t[0] for t in tracks if t[2] and t[0] not in values]
    meta = {
        "provider": "anthropic",
        "modelo": args.model,
        "effort": args.effort,
        "prompt_version": PROMPT_VERSION,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "n_faixas": len(tracks),
        "n_com_saida": len(values),
        "n_instrumental": n_instrumental,
        "n_faltam": len(missing),
        "completo": not missing,
        "custo_total_usd": round(usage.cost(args.model), 4),
        "escala": "prompt 1-9 -> to_unit -> [-1, 1] (V_raw, A_raw)",
    }
    (args.out.parent / "lf2_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n{TAG} {args.out}: {len(values)} com V/A, {len(missing)} faltam, "
        f"custo ${usage.cost(args.model):.4f}"
    )
    if over_budget or missing:
        print(f"{TAG} rode de novo (mesmo comando) para retomar de onde parou.")


if __name__ == "__main__":
    main()
