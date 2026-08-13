#!/usr/bin/env python
"""Runner da LF3 (LLM zero-shot) — estima V/A das letras via Gemini.

Roda no Python normal. Lê a LETRA COMPLETA de cada faixa em
``data/stimuli/lyrics/<id>.txt`` (mesmo id do áudio), manda ao LLM com o prompt
versionado (``prompts/lf3_zero_shot.md``), e escreve:

- ``data/lf_outputs/lf3_raw.csv``   (id, V_raw, A_raw em [-1, 1])
- ``data/lf_outputs/lf3_meta.json`` (modelo, versão do prompt, contagens)

Instrumentais (``.txt`` vazio) NÃO consomem chamada — viram ``NaN`` direto.

**A chave da API entra por variável de ambiente ``GEMINI_API_KEY``** (ou
``GOOGLE_API_KEY``) — nunca como argumento nem no código.

O adaptador do Gemini é a única parte específica de provedor; o núcleo da LF3
(``src/lf3_llm.py``) é agnóstico. Para trocar de provedor, basta outra
``make_*_client``.

Exemplos::

    # define a chave uma vez (Windows PowerShell): setx GEMINI_API_KEY "sua-chave"
    python scripts/gerar_lf3.py                 # --model auto (descobre um flash)
    python scripts/gerar_lf3.py --list-models   # ver modelos da sua chave e sair
    python scripts/gerar_lf3.py --model gemini-3.6-flash --rpm 15
    python scripts/gerar_lf3.py --fake          # testa o encanamento sem gastar chamada
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lf3_llm import PROMPT_VERSION, estimate_va_from_lyrics_llm  # noqa: E402
from src.lf_common import LFRawOutput, write_lf_raw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- #
# Adaptador do Gemini (única parte específica de provedor)
# ----------------------------------------------------------------------------- #
def _new_genai_client():
    """Instancia o SDK do Gemini (import preguiçoso; chave via env). -> (client, types)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover - depende de ambiente externo
        raise ImportError(
            "SDK do Gemini ausente. Rode `pip install google-genai`."
        ) from e
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("[lf3] defina a chave: setx GEMINI_API_KEY \"sua-chave\" "
                 "(reabra o terminal depois).")
    return genai.Client(api_key=api_key), types


def listar_modelos_generate(client) -> list[str]:
    """Nomes dos modelos que suportam generateContent (o que o LF3 usa)."""
    nomes = []
    for m in client.models.list():
        acts = (getattr(m, "supported_actions", None)
                or getattr(m, "supported_generation_methods", None) or [])
        if not acts or "generateContent" in acts:
            nomes.append(m.name)  # ex.: "models/gemini-3.6-flash"
    return nomes


def escolher_flash(nomes: list[str]) -> str | None:
    """Escolhe o melhor modelo 'flash' disponível (maior versão; evita variantes).

    Preferência: flash não-lite -> flash-lite -> qualquer generateContent. Ignora
    previews/experimentais e modelos não-texto.
    """
    ruins = ("preview", "exp", "thinking", "image", "tts", "audio", "live",
             "vision", "embedding", "learnlm", "gemma", "aqa")

    def ver(n: str) -> tuple[int, int]:
        m = re.search(r"(\d+)\.(\d+)", n)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    def limpo(n: str) -> bool:
        return not any(b in n.lower() for b in ruins)

    flash = [n for n in nomes if "flash" in n.lower() and "lite" not in n.lower()
             and limpo(n)]
    if flash:
        return max(flash, key=ver)
    lite = [n for n in nomes if "flash" in n.lower() and limpo(n)]
    if lite:
        return max(lite, key=ver)
    gen = [n for n in nomes if limpo(n)]
    return max(gen, key=ver) if gen else (nomes[0] if nomes else None)


def make_gemini_client(model: str, temperature: float, rpm: int,
                       max_retries: int = 5):
    """Cria um ``client`` (prompt -> texto) para a LF3 usando o Gemini.

    Se ``model == "auto"``, descobre um modelo 'flash' disponível para a sua chave
    (os IDs mudam e dependem da conta). Throttle p/ respeitar ``rpm`` (free tier:
    ~10 RPM), retry com backoff em 429/503, saída ``application/json``.

    Returns:
        ``(call, modelo_usado)`` — a função ``prompt -> texto`` e o id resolvido.
    """
    client, types = _new_genai_client()

    if model == "auto":
        disponiveis = listar_modelos_generate(client)
        model = escolher_flash(disponiveis)
        if not model:
            sys.exit("[lf3] nenhum modelo generateContent disponível para esta chave.")
        print(f"[lf3] modelo auto-selecionado: {model}")

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )
    min_interval = 60.0 / rpm if rpm > 0 else 0.0
    state = {"last": 0.0}

    def call(prompt: str) -> str:
        wait = min_interval - (time.monotonic() - state["last"])
        if wait > 0:
            time.sleep(wait)
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=cfg)
                state["last"] = time.monotonic()
                return resp.text or ""
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                daily = "PerDay" in msg  # cota DIÁRIA: não adianta esperar segundos
                server = "503" in msg or "UNAVAILABLE" in msg
                rate = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                transient = server or (rate and not daily)
                if attempt < max_retries - 1 and transient:
                    back = min(5 * 2 ** attempt, 60)
                    print(f"[lf3] limite/minuto; aguardando {back}s...")
                    time.sleep(back)
                    continue
                raise  # cota diária ou erro definitivo -> sobe para o main parar limpo
        return ""

    return call, model


def make_fake_client():
    """Cliente falso (sem rede) para testar o encanamento — V/A fixos."""
    def call(prompt: str) -> str:
        return '{"valence": 6.0, "arousal": 5.0}'
    return call


# ----------------------------------------------------------------------------- #
# Runner
# ----------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera lf3_raw.csv via LLM (Gemini).")
    p.add_argument("--lyrics-dir", type=Path,
                   default=ROOT / "data" / "stimuli" / "lyrics")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "lf_outputs" / "lf3_raw.csv")
    p.add_argument("--model", default="auto",
                   help="modelo Gemini ('auto' descobre um flash disponível).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="temperatura baixa p/ reprodutibilidade (padrão 0.0).")
    p.add_argument("--rpm", type=int, default=10,
                   help="teto de requests/min (free: ~10 Flash, ~15 Flash-Lite).")
    p.add_argument("--fake", action="store_true",
                   help="usa cliente falso (sem rede) para testar o pipeline.")
    p.add_argument("--list-models", action="store_true",
                   help="lista os modelos disponíveis para a sua chave e sai.")
    return p.parse_args()


def descobrir_letras(lyrics_dir: Path) -> list[tuple[str, str, bool]]:
    if not lyrics_dir.exists():
        sys.exit(f"[lf3] pasta de letras não existe: {lyrics_dir}")
    tracks = []
    for f in sorted(lyrics_dir.glob("*.txt")):
        texto = f.read_text(encoding="utf-8", errors="replace").strip()
        tracks.append((f.stem, texto, bool(texto)))
    return tracks


def carregar_feitos(out_path: Path) -> dict[str, tuple[float, float]]:
    """Lê V/A já calculados (não-NaN) de um lf3_raw.csv anterior -> {id:(v,a)}.

    Permite RETOMAR: com a cota diária pequena (~20/dia p/ o flash), a LF3 roda em
    vários dias/modelos; o que já foi feito não é refeito.
    """
    if not out_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(out_path)
    return {str(r["id"]): (float(r["V_raw"]), float(r["A_raw"]))
            for _, r in df.iterrows() if pd.notna(r["V_raw"])}


def escrever_csv(out_path: Path, tracks: list[tuple[str, str, bool]],
                 va: dict[str, tuple[float, float]]) -> None:
    """Escreve o CSV completo a partir de ``va``; faltantes/instrumentais viram NaN."""
    nan = float("nan")
    outs = [LFRawOutput(tid, *va.get(tid, (nan, nan))) for tid, _, _ in tracks]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lf_raw(outs, out_path)


def main() -> None:
    args = parse_args()

    if args.list_models:
        client, _ = _new_genai_client()
        nomes = listar_modelos_generate(client)
        print(f"[lf3] {len(nomes)} modelo(s) disponível(is) p/ generateContent:")
        for n in sorted(nomes):
            print("   ", n)
        escolhido = escolher_flash(nomes)
        print(f"[lf3] 'auto' escolheria: {escolhido}")
        return

    tracks = descobrir_letras(args.lyrics_dir)
    if not tracks:
        sys.exit(f"[lf3] nenhuma letra .txt em {args.lyrics_dir}.")
    n_vocal = sum(1 for _, _, tl in tracks if tl)
    n_instr = len(tracks) - n_vocal

    if args.fake:
        client, modelo, va = make_fake_client(), "fake", {}
        print("[lf3] MODO FAKE (sem rede) — só valida o encanamento.")
    else:
        client, modelo = make_gemini_client(args.model, args.temperature, args.rpm)
        va = carregar_feitos(args.out)  # RETOMA o que já foi feito
        if va:
            print(f"[lf3] retomando: {len(va)} faixa(s) já feitas serão puladas.")

    pendentes = [t for t in tracks if t[2] and t[0] not in va]
    print(f"[lf3] {len(pendentes)} faixa(s) a processar "
          f"(modelo {modelo}, {args.rpm} RPM, ~{len(pendentes)*60/max(args.rpm,1)/60:.0f} min); "
          f"{n_instr} instrumental(is).")

    parou_cota = False
    for tid, texto, tem_letra in tracks:
        if not tem_letra or tid in va:
            continue
        try:
            res = estimate_va_from_lyrics_llm(
                texto, client, tem_letra=True, model_name=modelo)
        except Exception as e:  # noqa: BLE001
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"[lf3] cota esgotada em {tid} — salvando e parando.")
                parou_cota = True
                break
            escrever_csv(args.out, tracks, va)  # salva antes de propagar
            raise
        if math.isnan(res.v_raw):
            print(f"  {tid}: ERRO ({res.meta.get('error', '?')[:60]}) — tentar depois")
            continue  # não grava; será retentado numa próxima rodada
        va[tid] = (res.v_raw, res.a_raw)
        print(f"  {tid}: V={res.v_raw:+.2f} A={res.a_raw:+.2f}  ({len(va)} ok)")
        escrever_csv(args.out, tracks, va)  # INCREMENTAL: sobrevive a crash/cota

    escrever_csv(args.out, tracks, va)

    n_ok = len(va)
    faltam = [t[0] for t in tracks if t[2] and t[0] not in va]
    meta = {
        "modelo": modelo,
        "prompt_version": PROMPT_VERSION,
        "temperature": args.temperature if not args.fake else None,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "n_faixas": len(tracks),
        "n_com_saida": n_ok,
        "n_instrumental": n_instr,
        "n_faltam": len(faltam),
        "completo": not faltam,
        "escala": "prompt 1-9 -> to_unit -> [-1, 1] (V_raw, A_raw)",
    }
    (args.out.parent / "lf3_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lf3] {args.out}  ({n_ok}/{n_vocal} vocais com V/A; faltam {len(faltam)})")
    if parou_cota or faltam:
        print("[lf3] Para completar as faltantes, faça UMA destas:")
        print("      • rode de novo AMANHÃ (retoma sozinho); ou")
        print(f"      • troque de modelo agora (cota é POR modelo): "
              f"--model gemini-3.5-flash-lite  /  --model gemini-flash-lite-latest; ou")
        print("      • ative billing (Tier 1) — ~centavos p/ as 53, remove os limites.")


if __name__ == "__main__":
    main()
