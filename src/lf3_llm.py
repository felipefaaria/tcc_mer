"""LF3 — estimador de Valência-Arousal da letra via LLM zero-shot.

Pede a um LLM que estime (V, A) da letra **sem exemplos de treino**, só com
instruções. Complementa o léxico (LF2) porque entende contexto, negação e ironia
— exatamente onde o ANEW-Br cega. Riscos: o LLM tem viés próprio (anglocêntrico),
pode alucinar e é sensível ao prompt. Por isso é **uma** LF, ancorada pelo gold
via calibração afim — nunca a verdade.

Contrato de reprodutibilidade (do roteiro):
- Prompt **fixo e versionado** em ``prompts/lf3_zero_shot.md``.
- Temperatura **baixa**, saída **JSON estruturada** com dois floats numa escala
  explícita (1-9), documentada no prompt.
- **Registrar modelo e versão** usados em cada chamada.

**Isolamento da dependência:** nenhum SDK de LLM é importado aqui. O acesso ao
modelo entra como um ``client`` injetável — uma função ``str (prompt) -> str
(resposta bruta)``. Isso mantém o módulo provider-agnóstico e testável com um
cliente falso; o adaptador real (Anthropic, OpenAI, etc.) é escrito na borda pelo
chamador, com temperatura baixa configurada lá.

Mesma regra de instrumentais da LF2: ``tem_letra=não`` -> ``NaN`` explícito.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterable

from .lf_common import LFRawOutput
from .scale import to_unit

__all__ = [
    "LLMClient",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "load_prompt_template",
    "build_prompt",
    "parse_response",
    "estimate_va_from_lyrics_llm",
    "run_batch",
    "LLM_LO",
    "LLM_HI",
]

# Escala pedida no prompt (SAM 1-9), convertida na borda para [-1, 1].
LLM_LO, LLM_HI = 1.0, 9.0

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "lf3_zero_shot.md"
PROMPT_VERSION = "v1"  # incrementar junto com o texto do prompt

# Um client recebe o prompt final e devolve a resposta bruta do LLM (texto).
LLMClient = Callable[[str], str]


def load_prompt_template(path: str | Path = PROMPT_PATH) -> str:
    """Lê o texto do prompt versionado do disco."""
    return Path(path).read_text(encoding="utf-8")


def build_prompt(lyrics: str, template: str | None = None) -> str:
    """Insere a letra no template do prompt (placeholder ``{lyrics}``)."""
    tmpl = template if template is not None else load_prompt_template()
    return tmpl.replace("{lyrics}", lyrics)


def parse_response(raw: str) -> tuple[float, float]:
    """Extrai ``(valence, arousal)`` (escala 1-9) da resposta bruta do LLM.

    Tolera texto ao redor do JSON (ex.: cercas ```json ... ```), localizando o
    primeiro objeto ``{...}`` na resposta.

    Raises:
        ValueError: se não houver JSON válido, faltarem chaves, os valores não
            forem numéricos ou caírem fora de [1, 9].
    """
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"nenhum objeto JSON encontrado na resposta: {raw!r}")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido na resposta: {raw!r}") from e

    if "valence" not in obj or "arousal" not in obj:
        raise ValueError(f"chaves 'valence'/'arousal' ausentes: {obj!r}")
    try:
        v = float(obj["valence"])
        a = float(obj["arousal"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"valores não numéricos: {obj!r}") from e

    for name, val in (("valence", v), ("arousal", a)):
        if not (LLM_LO <= val <= LLM_HI):
            raise ValueError(
                f"{name}={val} fora da escala [{LLM_LO}, {LLM_HI}]"
            )
    return v, a


def estimate_va_from_lyrics_llm(
    lyrics: str,
    client: LLMClient,
    *,
    tem_letra: bool = True,
    model_name: str = "",
    template: str | None = None,
) -> LFRawOutput:
    """Estima (V_raw, A_raw) em [-1, 1] da letra via LLM zero-shot.

    Args:
        lyrics: texto completo da letra.
        client: função ``str -> str`` que envia o prompt ao LLM (temperatura baixa
            configurada pelo chamador) e devolve a resposta bruta.
        tem_letra: se ``False`` (instrumental), retorna ``NaN`` explícito.
        model_name: identificador do modelo/versão, registrado em ``meta`` para
            reprodutibilidade.
        template: template do prompt (usa o versionado em disco se ``None``).

    Returns:
        :class:`LFRawOutput`; ``meta`` registra modelo, versão do prompt e a
        resposta bruta. Se o parsing falhar, V/A saem ``NaN`` e o erro fica em
        ``meta['error']`` (não interrompe o lote).
    """
    if not tem_letra:
        return LFRawOutput("", float("nan"), float("nan"),
                           {"reason": "instrumental",
                            "model": model_name, "prompt_version": PROMPT_VERSION})

    prompt = build_prompt(lyrics, template=template)
    raw = client(prompt)
    meta = {
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "raw_response": raw,
    }
    try:
        v_1_9, a_1_9 = parse_response(raw)
    except ValueError as e:
        meta["error"] = str(e)
        return LFRawOutput("", float("nan"), float("nan"), meta)

    return LFRawOutput(
        "",
        to_unit(v_1_9, LLM_LO, LLM_HI),
        to_unit(a_1_9, LLM_LO, LLM_HI),
        meta,
    )


def run_batch(
    tracks: Iterable[tuple[str, str, bool]],
    client: LLMClient,
    *,
    model_name: str = "",
    template: str | None = None,
) -> list[LFRawOutput]:
    """Roda a LF3 sobre várias faixas.

    Args:
        tracks: iterável de ``(track_id, lyrics, tem_letra)``.
        client: cliente de LLM injetável.
        model_name: identificador do modelo/versão (reprodutibilidade).
        template: template do prompt (opcional).

    Returns:
        Lista de :class:`LFRawOutput` (uma por faixa, ``track_id`` preenchido).
        Persista com :func:`src.lf_common.write_lf_raw` em ``lf3_raw.csv``.
    """
    out: list[LFRawOutput] = []
    for track_id, lyrics, tem_letra in tracks:
        res = estimate_va_from_lyrics_llm(
            lyrics, client, tem_letra=tem_letra,
            model_name=model_name, template=template,
        )
        res.track_id = track_id
        out.append(res)
    return out
