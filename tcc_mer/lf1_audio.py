"""LF1: valência e arousal estimados só do áudio, com o Music2Emo (MERT).

O Music2Emo não está no PyPI. É um repositório com dependências pinadas
(``torch==2.3.1``, ``numpy==1.26.4``, ``transformers``, ``librosa``...) que
conflitam com o ambiente de análise, então ele roda num container (ver ``docker/``)
e este módulo não o importa no topo. O modelo entra como um ``predictor``
injetável: uma função ``caminho_do_áudio -> dict`` com as chaves ``valence`` e
``arousal`` (escala 1-9) e, opcionalmente, ``predicted_moods``.
"""

from collections.abc import Callable, Iterable

from .lf_common import LFRawOutput
from .scale import to_unit

__all__ = [
    "AudioPredictor",
    "M2E_LO",
    "M2E_HI",
    "make_music2emo_predictor",
    "estimate_va_from_audio",
    "run_batch",
]

# Escala nativa do Music2Emo, convertida na borda para [-1, 1].
M2E_LO, M2E_HI = 1.0, 9.0

AudioPredictor = Callable[[str], dict]


def make_music2emo_predictor(**model_kwargs) -> AudioPredictor:
    """Cria um predictor que embrulha ``music2emo.Music2emo().predict``.

    O import é preguiçoso, então este módulo e seus testes funcionam sem o pacote.
    Instancie o predictor uma vez e reutilize em todas as faixas.

    Raises:
        ImportError: se ``music2emo`` não estiver disponível (use o container).
    """
    try:
        from music2emo import Music2emo  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            "Pacote 'music2emo' não encontrado. Rode dentro do container de docker/ "
            "ou injete outro predictor."
        ) from error

    model = Music2emo(**model_kwargs)
    return model.predict


def estimate_va_from_audio(audio_path: str, predictor: AudioPredictor) -> LFRawOutput:
    """Estima ``(V_raw, A_raw)`` em [-1, 1] de um trecho de áudio.

    Os moods previstos vão em ``meta``.
    """
    output = predictor(audio_path)
    return LFRawOutput(
        "",
        to_unit(float(output["valence"]), M2E_LO, M2E_HI),
        to_unit(float(output["arousal"]), M2E_LO, M2E_HI),
        {"predicted_moods": output.get("predicted_moods", []), "audio_path": audio_path},
    )


def run_batch(
    tracks: Iterable[tuple[str, str]],
    predictor: AudioPredictor,
    *,
    on_error: str = "nan",
) -> list[LFRawOutput]:
    """Roda a LF1 em várias faixas.

    Args:
        tracks: iterável de ``(track_id, audio_path)``.
        predictor: predictor de áudio (injetável).
        on_error: ``"nan"`` grava ``NaN`` com o erro em ``meta`` e segue o lote;
            ``"raise"`` propaga a exceção.
    """
    results: list[LFRawOutput] = []
    for track_id, audio_path in tracks:
        try:
            result = estimate_va_from_audio(audio_path, predictor)
            result.track_id = track_id
        except Exception as error:
            if on_error == "raise":
                raise
            result = LFRawOutput(
                track_id,
                float("nan"),
                float("nan"),
                {"error": repr(error), "audio_path": audio_path},
            )
        results.append(result)
    return results
