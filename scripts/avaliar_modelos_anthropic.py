#!/usr/bin/env python
"""Avalia qual modelo Claude (Anthropic) é o melhor custo-benefício para a LF3,
rodando o MESMO prompt versionado (``prompts/lf3_zero_shot.md``) contra o gold
humano real (``data/annotations/gold_wide.csv``, coluna ``V_letra``/``A_letra``)
e medindo CCC — a métrica primária do projeto — em vez de "achismo" sobre qual
modelo parece melhor.

**Freio de orçamento (duas camadas, pensado pra você poder testar à vontade):**

1. **Pré-voo (antes de gastar 1 centavo):** usa ``client.messages.count_tokens``
   pra estimar o custo de TODA a rodada planejada (todos os modelos x todas as
   faixas) e recusa rodar se a estimativa passar ``--budget``.
2. **Em tempo real:** acumula o custo de verdade (``response.usage``) chamada a
   chamada; se ultrapassar ``--budget`` NO MEIO da rodada, para ali (não some
   nada — o que já rodou já vira resultado parcial).

Preços em USD/1M tokens (conferir contra a doc oficial da Anthropic se for
reusar depois de muito tempo — modelo e preço podem mudar):
    claude-haiku-4-5   $1.00 / $5.00
    claude-sonnet-5    $2.00 / $10.00
    claude-opus-5      $5.00 / $25.00

**Nota de API importante:** Claude Opus 5 e Claude Sonnet 5 NÃO aceitam mais
``temperature`` (parâmetro removido, erro 400) — o controle de determinismo/
custo nesses dois é ``output_config.effort`` (usamos "low", que é o recomendado
pra tarefas de classificação simples como esta). Só o Haiku 4.5 ainda aceita
``temperature`` (mandamos 0.0 nele para reprodutibilidade).

Exemplos::

    # só a estimativa de custo, sem gastar nada
    python scripts/avaliar_modelos_anthropic.py --n 15 --dry-run

    # roda de verdade, teto de USD 2 (bem abaixo dos USD 20 de crédito)
    python scripts/avaliar_modelos_anthropic.py --n 15 --budget 2.0

    # só um modelo, mais faixas
    python scripts/avaliar_modelos_anthropic.py --models claude-sonnet-5 --n 30
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.lf3_llm import build_prompt, estimate_va_from_lyrics_llm  # noqa: E402
from src.metrics import ccc  # noqa: E402

# preço USD por 1M tokens: (input, output) — ver docstring do módulo
PRECOS = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
MODELOS_COM_TEMPERATURA = {"claude-haiku-4-5"}  # os outros usam effort, não temperature
EST_TOKENS_SAIDA = 30  # {"valence": 7.2, "arousal": 6.5} — chute generoso p/ pré-voo


def carregar_tracks_gold(n: int, seed: int,
                        lyrics_dir: Path) -> list[tuple[str, str, float, float]]:
    """-> [(track_id, letra, V_gold, A_gold)] amostrado do gold real com letra."""
    gold = pd.read_csv(ROOT / "data" / "annotations" / "gold_wide.csv")
    usaveis = []
    for _, row in gold[gold["V_letra"].notna()].iterrows():
        tid = row["trackId"]
        f = lyrics_dir / f"{tid}.txt"
        if not f.exists():
            continue
        texto = f.read_text(encoding="utf-8", errors="replace").strip()
        if not texto:
            continue
        usaveis.append((tid, texto, float(row["V_letra"]), float(row["A_letra"])))
    if not usaveis:
        sys.exit(f"[avaliar] nenhuma faixa com gold de letra + .txt em {lyrics_dir}.")
    rng = random.Random(seed)
    rng.shuffle(usaveis)
    return usaveis[:n]


def estimar_custo_preflight(client, model: str,
                            tracks: list[tuple[str, str, float, float]]) -> float:
    """Estima o custo TOTAL (todas as faixas) para um modelo, sem gastar nada."""
    preco_in, preco_out = PRECOS[model]
    total_in = 0
    for _, letra, _, _ in tracks:
        prompt = build_prompt(letra)
        resp = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": prompt}])
        total_in += resp.input_tokens
    return (total_in * preco_in + len(tracks) * EST_TOKENS_SAIDA * preco_out) / 1e6


def make_anthropic_client(sdk_client, model: str, effort: str):
    """-> (call, uso) — ``call`` é o ``LLMClient`` (str->str) que src/lf3_llm.py
    espera; ``uso`` acumula tokens reais de cada chamada (pra custo em tempo real).
    """
    uso = {"input": 0, "output": 0, "chamadas": 0}

    def call(prompt: str) -> str:
        kwargs = dict(model=model, max_tokens=256,
                     messages=[{"role": "user", "content": prompt}])
        if model in MODELOS_COM_TEMPERATURA:
            # SDK 1.x removeu `temperature` da assinatura tipada de create()
            # (não é restrição do modelo/API, é só o wrapper Python) — passa
            # via extra_body, que vai direto no corpo HTTP sem checagem tipada.
            kwargs["extra_body"] = {"temperature": 0.0}
        else:
            kwargs["output_config"] = {"effort": effort}
        resp = sdk_client.messages.create(**kwargs)
        uso["input"] += resp.usage.input_tokens
        uso["output"] += resp.usage.output_tokens
        uso["chamadas"] += 1
        return "".join(b.text for b in resp.content if b.type == "text")

    return call, uso


def custo_atual(uso: dict, model: str) -> float:
    preco_in, preco_out = PRECOS[model]
    return (uso["input"] * preco_in + uso["output"] * preco_out) / 1e6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", default="claude-haiku-4-5,claude-sonnet-5,claude-opus-5")
    p.add_argument("--n", type=int, default=15,
                   help="quantas faixas (com gold de letra) testar por modelo.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--effort", default="low",
                   help="output_config.effort p/ Sonnet/Opus (low recomendado p/ classificação).")
    p.add_argument("--budget", type=float, default=2.0,
                   help="teto em USD (some os modelos) — bem abaixo dos seus 20 USD de crédito.")
    p.add_argument("--lyrics-dir", type=Path, default=ROOT / "data" / "corpus" / "lyrics")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "outputs" / "avaliacao_modelos_lf3.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="só mostra a estimativa de custo, não gasta nada.")
    p.add_argument("--yes", action="store_true",
                   help="pula a confirmação interativa antes de gastar de verdade.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modelos = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in modelos:
        if m not in PRECOS:
            sys.exit(f"[avaliar] modelo desconhecido: {m!r} (conhecidos: {list(PRECOS)})")

    import anthropic
    sdk_client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente/.env

    tracks = carregar_tracks_gold(args.n, args.seed, args.lyrics_dir)
    print(f"[avaliar] {len(tracks)} faixa(s) com gold de letra real "
          f"(seed={args.seed}): {', '.join(t[0] for t in tracks)}")

    print("[avaliar] estimando custo (pré-voo, count_tokens — nada gasto ainda)...")
    estimativas = {m: estimar_custo_preflight(sdk_client, m, tracks) for m in modelos}
    total_estimado = sum(estimativas.values())
    for m, custo in estimativas.items():
        print(f"    {m}: ~${custo:.4f}")
    print(f"[avaliar] total estimado: ~${total_estimado:.4f}  "
          f"(orçamento definido: ${args.budget:.2f})")

    if total_estimado > args.budget:
        sys.exit(f"[avaliar] ABORTADO: estimativa (~${total_estimado:.4f}) passa do "
                 f"--budget (${args.budget:.2f}). Reduza --n/--models ou suba --budget.")

    if args.dry_run:
        print("[avaliar] --dry-run: nada gasto, nenhuma chamada de fato feita.")
        return

    if not args.yes:
        resp = input(f"Confirma gastar de verdade até ~${total_estimado:.4f}? [s/N] ")
        if resp.strip().lower() != "s":
            sys.exit("[avaliar] abortado por você.")

    resultados = []
    for model in modelos:
        print(f"\n[avaliar] === {model} (effort={args.effort}) ===")
        client_fn, uso = make_anthropic_client(sdk_client, model, args.effort)
        preds_v, preds_a, gold_v, gold_a = [], [], [], []
        t0 = time.monotonic()
        parou_por_orcamento = False
        for tid, letra, vg, ag in tracks:
            try:
                res = estimate_va_from_lyrics_llm(letra, client_fn, tem_letra=True,
                                                  model_name=model)
            except Exception as e:  # noqa: BLE001
                print(f"    {tid}: ERRO ({e}) — pulando")
                continue
            preds_v.append(res.v_raw)
            preds_a.append(res.a_raw)
            gold_v.append(vg)
            gold_a.append(ag)
            c = custo_atual(uso, model)
            print(f"    {tid}: V={res.v_raw:+.2f} A={res.a_raw:+.2f} "
                  f"(gold V={vg:+.2f} A={ag:+.2f})  custo acumulado ${c:.4f}")
            if c > args.budget:
                print(f"    [PARADA DE SEGURANÇA] custo real (${c:.4f}) passou o "
                      f"--budget (${args.budget:.2f}) no meio da rodada — parando aqui.")
                parou_por_orcamento = True
                break
        tempo = time.monotonic() - t0
        custo_final = custo_atual(uso, model)
        resultados.append({
            "model": model,
            "n_faixas": len(preds_v),
            "ccc_valencia": ccc(np.array(preds_v), np.array(gold_v)),
            "ccc_arousal": ccc(np.array(preds_a), np.array(gold_a)),
            "custo_total_usd": round(custo_final, 4),
            "custo_por_musica_usd": round(custo_final / max(len(preds_v), 1), 5),
            "tempo_total_s": round(tempo, 1),
            "parou_por_orcamento": parou_por_orcamento,
        })

    print("\n[avaliar] === RESULTADO ===")
    print(f"{'modelo':<20} {'n':>4} {'CCC_V':>8} {'CCC_A':>8} "
          f"{'custo_total':>12} {'custo/música':>13} {'tempo_s':>9}")
    for r in resultados:
        print(f"{r['model']:<20} {r['n_faixas']:>4} {r['ccc_valencia']:>8.3f} "
              f"{r['ccc_arousal']:>8.3f} ${r['custo_total_usd']:>10.4f} "
              f"${r['custo_por_musica_usd']:>11.5f} {r['tempo_total_s']:>9.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        w.writeheader()
        w.writerows(resultados)
    print(f"\n[avaliar] resultado salvo em {args.out}")
    print("[avaliar] leitura: maior CCC_V/CCC_A contra o gold real = melhor qualidade; "
          "escolha o mais barato que não perca CCC de forma relevante.")


if __name__ == "__main__":
    main()
