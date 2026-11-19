#!/usr/bin/env python
"""Runner de PRODUÇÃO da LF3 (LLM zero-shot) via Anthropic — estima V/A das
letras usando Claude (padrão: Sonnet 5, decidido em `avaliar_modelos_anthropic.py`
contra o gold real: CCC praticamente empatado com Opus 5 em valência/arousal,
~2,5x mais barato — ver `data/outputs/avaliacao_modelos_lf3_n40.csv`).

Mesmo contrato do `gerar_lf3.py` (Gemini): lê a LETRA COMPLETA de cada faixa em
``data/corpus/lyrics/<id>.txt``, manda ao LLM com o MESMO prompt versionado
(``prompts/lf3_zero_shot.md``), e escreve:

- ``data/outputs/lf3_raw.csv``   (id, V_raw, A_raw em [-1, 1])
- ``data/outputs/lf3_meta.json`` (modelo, provider, versão do prompt, contagens)

Instrumentais (``.txt`` vazio) NÃO consomem chamada — viram ``NaN`` direto.
RETOMA sozinho: se ``lf3_raw.csv`` já existe, pula ids que já têm V/A (não-NaN).

**ATENÇÃO METODOLÓGICA:** por padrão este script escreve no MESMO
``lf3_raw.csv`` que o `gerar_lf3.py` (Gemini) já preencheu para as 53 faixas
originais. Como ele só recalcula o que está faltando, isso deixaria o canal LF3
com **dois instrumentos diferentes misturados na mesma coluna** (Gemini nas 53
antigas, Claude nas faixas novas) — o mesmo problema que descartamos ao decidir
NUNCA trocar de instrumento de medição no meio do estudo (ex.: não usar
"valence" do Spotify como se fosse a LF1). Use ``--force-todas`` para
recalcular TUDO com Claude de forma uniforme antes de fechar qualquer
resultado comparativo entre corpus antigo e novo.

**A chave da API entra por variável de ambiente ``ANTHROPIC_API_KEY``** (lida
de um ``.env`` na raiz se existir) — nunca como argumento nem no código.

**Freio de orçamento** (mesma lógica do `avaliar_modelos_anthropic.py`):
pré-voo com ``count_tokens`` antes de gastar, e parada real se o custo
acumulado passar ``--budget`` no meio da rodada.

Exemplos::

    python scripts/gerar_lf3_anthropic.py --dry-run           # só estimativa
    python scripts/gerar_lf3_anthropic.py                     # roda o que falta
    python scripts/gerar_lf3_anthropic.py --force-todas       # recalcula tudo
    python scripts/gerar_lf3_anthropic.py --model claude-opus-5 --budget 5.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.lf3_llm import PROMPT_VERSION, build_prompt, estimate_va_from_lyrics_llm  # noqa: E402
from src.lf_common import LFRawOutput, write_lf_raw  # noqa: E402

# preço USD por 1M tokens: (input, output) — conferir contra a doc oficial se
# for reusar depois de muito tempo (modelo e preço podem mudar).
PRECOS = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
MODELOS_COM_TEMPERATURA = {"claude-haiku-4-5"}  # os demais usam output_config.effort
EST_TOKENS_SAIDA = 30  # {"valence": 7.2, "arousal": 6.5} — chute generoso p/ pré-voo


# ----------------------------------------------------------------------------- #
# Adaptador Anthropic (única parte específica de provedor)
# ----------------------------------------------------------------------------- #
def make_anthropic_client(sdk_client, model: str, effort: str):
    """-> (call, uso). ``call`` é o ``LLMClient`` (str->str) que src/lf3_llm.py
    espera; ``uso`` acumula tokens reais (input/output/chamadas) desta sessão.
    """
    uso = {"input": 0, "output": 0, "chamadas": 0}

    def call(prompt: str) -> str:
        kwargs = dict(model=model, max_tokens=256,
                     messages=[{"role": "user", "content": prompt}])
        if model in MODELOS_COM_TEMPERATURA:
            # SDK 1.x removeu `temperature` da assinatura tipada de create()
            # (não é restrição do modelo, é só o wrapper Python) — vai via
            # extra_body, direto no corpo HTTP, sem checagem tipada.
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


def estimar_custo_preflight(sdk_client, model: str,
                            tracks: list[tuple[str, str, bool]]) -> float:
    """Estima custo TOTAL das faixas com letra ainda pendentes, sem gastar nada."""
    preco_in, preco_out = PRECOS[model]
    total_in = 0
    n_com_letra = 0
    for _, texto, tem_letra in tracks:
        if not tem_letra:
            continue
        n_com_letra += 1
        prompt = build_prompt(texto)
        resp = sdk_client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": prompt}])
        total_in += resp.input_tokens
    return (total_in * preco_in + n_com_letra * EST_TOKENS_SAIDA * preco_out) / 1e6


# ----------------------------------------------------------------------------- #
# Runner
# ----------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera lf3_raw.csv via LLM (Anthropic/Claude).")
    p.add_argument("--lyrics-dir", type=Path,
                   default=ROOT / "data" / "corpus" / "lyrics")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "outputs" / "lf3_raw.csv")
    p.add_argument("--model", default="claude-sonnet-5",
                   help="modelo Claude (decidido: claude-sonnet-5, ver docstring).")
    p.add_argument("--effort", default="low",
                   help="output_config.effort (low recomendado p/ esta tarefa).")
    p.add_argument("--budget", type=float, default=2.0,
                   help="teto em USD; para a rodada se passar disso.")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="pausa (s) entre chamadas, por etiqueta com a API.")
    p.add_argument("--force-todas", action="store_true",
                   help="recalcula TODAS as faixas com Claude, mesmo as que já "
                        "têm valor de outro provedor (evita misturar instrumentos "
                        "— ver ATENÇÃO METODOLÓGICA no topo do arquivo).")
    p.add_argument("--dry-run", action="store_true",
                   help="só mostra a estimativa de custo, não gasta nada.")
    p.add_argument("--yes", action="store_true",
                   help="pula a confirmação interativa antes de gastar de verdade.")
    return p.parse_args()


def descobrir_letras(lyrics_dir: Path) -> list[tuple[str, str, bool]]:
    if not lyrics_dir.exists():
        sys.exit(f"[lf3-anthropic] pasta de letras não existe: {lyrics_dir}")
    tracks = []
    for f in sorted(lyrics_dir.glob("*.txt")):
        texto = f.read_text(encoding="utf-8", errors="replace").strip()
        tracks.append((f.stem, texto, bool(texto)))
    return tracks


def carregar_feitos(out_path: Path, force_todas: bool) -> dict[str, tuple[float, float]]:
    """Lê V/A já calculados (não-NaN) de um lf3_raw.csv anterior -> {id:(v,a)}.

    Com ``force_todas=True``, ignora o que já existe (recalcula tudo, mesmo o
    que veio de outro provedor) — ver ATENÇÃO METODOLÓGICA no topo do arquivo.
    """
    if force_todas or not out_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(out_path)
    return {str(r["id"]): (float(r["V_raw"]), float(r["A_raw"]))
            for _, r in df.iterrows() if pd.notna(r["V_raw"])}


def escrever_csv(out_path: Path, tracks: list[tuple[str, str, bool]],
                 va: dict[str, tuple[float, float]]) -> None:
    nan = float("nan")
    outs = [LFRawOutput(tid, *va.get(tid, (nan, nan))) for tid, _, _ in tracks]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lf_raw(outs, out_path)


def main() -> None:
    args = parse_args()
    if args.model not in PRECOS:
        sys.exit(f"[lf3-anthropic] modelo desconhecido: {args.model!r} "
                 f"(conhecidos: {list(PRECOS)})")

    import anthropic
    sdk_client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente/.env

    tracks = descobrir_letras(args.lyrics_dir)
    if not tracks:
        sys.exit(f"[lf3-anthropic] nenhuma letra .txt em {args.lyrics_dir}.")

    va = carregar_feitos(args.out, args.force_todas)
    if va:
        print(f"[lf3-anthropic] retomando: {len(va)} faixa(s) já feitas serão puladas "
              f"(use --force-todas para recalcular tudo com Claude).")

    pendentes = [t for t in tracks if t[2] and t[0] not in va]
    n_instr = sum(1 for _, _, tl in tracks if not tl)
    print(f"[lf3-anthropic] {len(pendentes)} faixa(s) a processar com {args.model} "
          f"(de {len(tracks)} no total; {n_instr} instrumental(is); "
          f"{len(tracks) - len(pendentes) - n_instr} já prontas).")

    if not pendentes:
        print("[lf3-anthropic] nada a fazer — todas as faixas com letra já têm V/A.")
        escrever_csv(args.out, tracks, va)  # garante NaN explícito nos instrumentais
        return

    print("[lf3-anthropic] estimando custo (pré-voo, count_tokens — nada gasto ainda)...")
    custo_estimado = estimar_custo_preflight(sdk_client, args.model, pendentes)
    print(f"[lf3-anthropic] custo estimado: ~${custo_estimado:.4f} "
          f"(orçamento: ${args.budget:.2f})")
    if custo_estimado > args.budget:
        sys.exit(f"[lf3-anthropic] ABORTADO: estimativa (~${custo_estimado:.4f}) passa "
                 f"do --budget (${args.budget:.2f}). Suba --budget ou rode em partes.")

    if args.dry_run:
        print("[lf3-anthropic] --dry-run: nada gasto, nenhuma chamada de fato feita.")
        return

    if not args.yes:
        resp = input(f"Confirma gastar de verdade até ~${custo_estimado:.4f}? [s/N] ")
        if resp.strip().lower() != "s":
            sys.exit("[lf3-anthropic] abortado por você.")

    client_fn, uso = make_anthropic_client(sdk_client, args.model, args.effort)

    parou_por_orcamento = False
    for tid, texto, tem_letra in pendentes:
        try:
            res = estimate_va_from_lyrics_llm(
                texto, client_fn, tem_letra=True, model_name=args.model)
        except Exception as e:  # noqa: BLE001
            escrever_csv(args.out, tracks, va)  # salva antes de propagar
            raise
        if res.v_raw != res.v_raw:  # NaN check sem depender de math/numpy aqui
            print(f"  {tid}: ERRO ({res.meta.get('error', '?')[:60]}) — tentar depois")
            time.sleep(args.sleep)
            continue
        va[tid] = (res.v_raw, res.a_raw)
        c = custo_atual(uso, args.model)
        print(f"  {tid}: V={res.v_raw:+.2f} A={res.a_raw:+.2f}  "
              f"({len(va)} ok, custo acumulado ${c:.4f})")
        escrever_csv(args.out, tracks, va)  # INCREMENTAL: sobrevive a crash
        if c > args.budget:
            print(f"  [PARADA DE SEGURANÇA] custo real (${c:.4f}) passou o "
                  f"--budget (${args.budget:.2f}) — parando aqui.")
            parou_por_orcamento = True
            break
        time.sleep(args.sleep)

    escrever_csv(args.out, tracks, va)

    n_ok = len(va)
    faltam = [t[0] for t in tracks if t[2] and t[0] not in va]
    meta = {
        "provider": "anthropic",
        "modelo": args.model,
        "effort": args.effort,
        "prompt_version": PROMPT_VERSION,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "n_faixas": len(tracks),
        "n_com_saida": n_ok,
        "n_instrumental": n_instr,
        "n_faltam": len(faltam),
        "completo": not faltam,
        "custo_total_usd": round(custo_atual(uso, args.model), 4),
        "escala": "prompt 1-9 -> to_unit -> [-1, 1] (V_raw, A_raw)",
    }
    (args.out.parent / "lf3_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[lf3-anthropic] {args.out}  ({n_ok}/{len(tracks) - n_instr} com letra "
          f"têm V/A; faltam {len(faltam)}; custo total ${custo_atual(uso, args.model):.4f})")
    if parou_por_orcamento or faltam:
        print("[lf3-anthropic] rode de novo (mesmo comando) para retomar de onde parou.")


if __name__ == "__main__":
    main()
