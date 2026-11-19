"""LF2: valência e arousal da letra estimados por um LLM em regime zero-shot.

O modelo recebe a letra e uma instrução, sem exemplos, e devolve ``(V, A)`` em
JSON na escala 1-9. Contrato de reprodutibilidade:

* prompt fixo e versionado em ``prompts/lf2_zero_shot.md`` (``PROMPT_VERSION``);
* saída JSON estruturada com dois números na escala 1-9;
* modelo e versão do prompt registrados em ``meta`` a cada chamada.

Nenhum SDK é importado aqui. O acesso ao modelo é um ``client`` injetável
(``prompt -> resposta bruta``), o que mantém o módulo agnóstico de provedor e
testável com um cliente falso. O adaptador real fica em
``scripts/rotulagem/gerar_lf2_llm.py``.

Faixa instrumental (``has_lyrics=False``) devolve ``NaN`` explícito.
"""

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from .lf_common import LFRawOutput
from .paths import PROMPTS_DIR
from .scale import to_unit

__all__ = [
    "LLMClient",
    "LLM_LO",
    "LLM_HI",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "load_prompt_template",
    "build_prompt",
    "parse_response",
    "estimate_va_from_lyrics_llm",
    "run_batch",
]

# Escala pedida no prompt (SAM 1-9), convertida na borda para [-1, 1].
LLM_LO, LLM_HI = 1.0, 9.0

PROMPT_PATH = PROMPTS_DIR / "lf2_zero_shot.md"
PROMPT_VERSION = "v1"  # incrementar junto com o texto do prompt

LLMClient = Callable[[str], str]


def load_prompt_template(path: str | Path = PROMPT_PATH) -> str:
    """Lê o texto do prompt versionado."""
    return Path(path).read_text(encoding="utf-8")


def build_prompt(lyrics: str, template: str | None = None) -> str:
    """Insere a letra no placeholder ``{lyrics}`` do template."""
    template = load_prompt_template() if template is None else template
    return template.replace("{lyrics}", lyrics)


def parse_response(raw: str) -> tuple[float, float]:
    """Extrai ``(valence, arousal)`` (escala 1-9) da resposta bruta do LLM.

    Tolera texto ao redor do JSON (como cercas ```json) e lê o primeiro objeto
    ``{...}``.

    Raises:
        ValueError: sem JSON válido, sem as chaves, com valor não numérico ou fora
            de [1, 9].
    """
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"nenhum objeto JSON encontrado na resposta: {raw!r}")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON inválido na resposta: {raw!r}") from error

    if "valence" not in obj or "arousal" not in obj:
        raise ValueError(f"chaves 'valence'/'arousal' ausentes: {obj!r}")
    try:
        valence, arousal = float(obj["valence"]), float(obj["arousal"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"valores não numéricos: {obj!r}") from error

    for name, value in (("valence", valence), ("arousal", arousal)):
        if not LLM_LO <= value <= LLM_HI:
            raise ValueError(f"{name}={value} fora da escala [{LLM_LO}, {LLM_HI}]")
    return valence, arousal


def estimate_va_from_lyrics_llm(
    lyrics: str,
    client: LLMClient,
    *,
    has_lyrics: bool = True,
    model_name: str = "",
    template: str | None = None,
) -> LFRawOutput:
    """Estima ``(V_raw, A_raw)`` em [-1, 1] da letra, com um LLM zero-shot.

    Args:
        lyrics: texto completo da letra.
        client: função ``prompt -> resposta bruta``. A temperatura baixa (ou o
            esforço de raciocínio) é configurada por quem cria o cliente.
        has_lyrics: ``False`` para faixa instrumental, que devolve ``NaN``.
        model_name: modelo e versão, registrados em ``meta``.
        template: template do prompt; ``None`` usa o versionado em disco.

    Returns:
        :class:`LFRawOutput`. ``meta`` traz modelo, versão do prompt e a resposta
        bruta. Se o parsing falhar, V e A saem ``NaN`` com o erro em
        ``meta["error"]``, sem interromper o lote.
    """
    meta = {"model": model_name, "prompt_version": PROMPT_VERSION}
    if not has_lyrics:
        return LFRawOutput("", float("nan"), float("nan"), {**meta, "reason": "instrumental"})

    raw = client(build_prompt(lyrics, template=template))
    meta["raw_response"] = raw
    try:
        valence, arousal = parse_response(raw)
    except ValueError as error:
        return LFRawOutput("", float("nan"), float("nan"), {**meta, "error": str(error)})

    return LFRawOutput("", to_unit(valence, LLM_LO, LLM_HI), to_unit(arousal, LLM_LO, LLM_HI), meta)


def run_batch(
    tracks: Iterable[tuple[str, str, bool]],
    client: LLMClient,
    *,
    model_name: str = "",
    template: str | None = None,
) -> list[LFRawOutput]:
    """Roda a LF2 em várias faixas.

    Args:
        tracks: iterável de ``(track_id, lyrics, has_lyrics)``.
        client: cliente de LLM injetável.
        model_name: modelo e versão, para reprodutibilidade.
        template: template do prompt (opcional).
    """
    results = []
    for track_id, lyrics, has_lyrics in tracks:
        result = estimate_va_from_lyrics_llm(
            lyrics, client, has_lyrics=has_lyrics, model_name=model_name, template=template
        )
        result.track_id = track_id
        results.append(result)
    return results
