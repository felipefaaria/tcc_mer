"""LF1 — estimador de Valência-Arousal **do áudio** (Music2Emo / MERT).

Estima (V, A) sem letra, substituindo o par MusiCNN+DEAM do TCC1. É forte em
Arousal e fraca (e enviesada) em Valência — por design, já que não vê letra.

**Isolamento da dependência (crítico).** O Music2Emo não está no PyPI: é
clone-do-repo com dependências pinadas (torch==2.3.1, numpy==1.26.4, transformers,
librosa, nnAudio, music21, pytorch_lightning...) que conflitam com o ambiente de
análise. Por isso o modelo NÃO é importado no topo deste módulo. Ele entra como um
``predictor`` injetável — uma função ``str (caminho do áudio) -> dict`` com as
chaves ``valence``/``arousal`` (escala 1-9) e ``predicted_moods``.

Consequências práticas:
- A lógica (conversão de escala, persistência, smoke test) é testável **sem** o
  Music2Emo instalado.
- O modelo real deve viver num **venv isolado** (ver ``docker/requirements-lf1.txt``);
  ``make_music2emo_predictor()`` faz o import preguiçoso lá dentro.
- Se o Music2Emo não subir, troca-se só o ``predictor`` (Plano B) sem tocar no
  resto do pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from .lf_common import LFRawOutput
from .scale import to_unit

__all__ = [
    "AudioPredictor",
    "make_music2emo_predictor",
    "estimate_va_from_audio",
    "run_batch",
    "M2E_LO",
    "M2E_HI",
]

# Escala nativa do Music2Emo (valência/arousal), convertida na borda para [-1, 1].
M2E_LO, M2E_HI = 1.0, 9.0

# Um predictor recebe o caminho do áudio e devolve o dict do Music2Emo.
AudioPredictor = Callable[[str], dict]


def make_music2emo_predictor(**model_kwargs) -> AudioPredictor:
    """Cria um predictor que embrulha ``music2emo.Music2emo().predict``.

    Import **preguiçoso**: só toca no pacote pesado quando esta função é chamada,
    permitindo importar :mod:`src.lf1_audio` e rodar os testes sem o Music2Emo.
    Instancie o modelo **uma vez** e reutilize o predictor em todas as faixas.

    Raises:
        ImportError: se o pacote ``music2emo`` não estiver disponível no ambiente
            atual (rode no venv isolado de ``docker/requirements-lf1.txt``).
    """
    try:
        from music2emo import Music2emo  # type: ignore
    except ImportError as e:  # pragma: no cover - depende de ambiente externo
        raise ImportError(
            "Pacote 'music2emo' não encontrado. Instale num venv isolado "
            "(docker/requirements-lf1.txt) — ele conflita com o numpy/torch do ambiente "
            "de análise. Ou injete outro predictor (Plano B)."
        ) from e

    model = Music2emo(**model_kwargs)

    def _predict(audio_path: str) -> dict:
        return model.predict(audio_path)

    return _predict


def estimate_va_from_audio(
    audio_path: str,
    predictor: AudioPredictor,
) -> LFRawOutput:
    """Estima (V_raw, A_raw) em [-1, 1] a partir de um excerto de áudio.

    Args:
        audio_path: caminho do excerto (~30-45s centrado no refrão).
        predictor: função ``str -> dict`` (ver :func:`make_music2emo_predictor`
            ou um fake nos testes). Deve devolver ``valence``/``arousal`` (1-9) e,
            opcionalmente, ``predicted_moods``.

    Returns:
        :class:`LFRawOutput` com V/A em [-1, 1]; os moods vão em ``meta``.
    """
    out = predictor(audio_path)
    valence_1_9 = float(out["valence"])
    arousal_1_9 = float(out["arousal"])
    moods = out.get("predicted_moods", [])
    return LFRawOutput(
        "",
        to_unit(valence_1_9, M2E_LO, M2E_HI),
        to_unit(arousal_1_9, M2E_LO, M2E_HI),
        {"predicted_moods": moods, "audio_path": audio_path},
    )


def run_batch(
    tracks: Iterable[tuple[str, str]],
    predictor: AudioPredictor,
    *,
    on_error: str = "nan",
) -> list[LFRawOutput]:
    """Roda a LF1 sobre várias faixas (use um subconjunto de 5-10 como smoke test).

    Args:
        tracks: iterável de ``(track_id, audio_path)``.
        predictor: predictor de áudio (injetável).
        on_error: ``"nan"`` registra a faixa com V/A = ``NaN`` e o erro em ``meta``
            sem interromper o lote; ``"raise"`` propaga a exceção.

    Returns:
        Lista de :class:`LFRawOutput` (uma por faixa, com ``track_id`` preenchido).
        Persista com :func:`src.lf_common.write_lf_raw` em ``lf1_raw.csv``.
    """
    results: list[LFRawOutput] = []
    for track_id, audio_path in tracks:
        try:
            res = estimate_va_from_audio(audio_path, predictor)
            res.track_id = track_id
        except Exception as e:  # noqa: BLE001 - lote não pode parar por 1 faixa
            if on_error == "raise":
                raise
            res = LFRawOutput(track_id, float("nan"), float("nan"),
                              {"error": repr(e), "audio_path": audio_path})
        results.append(res)
    return results
