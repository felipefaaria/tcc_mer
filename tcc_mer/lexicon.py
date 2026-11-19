"""Estimador léxico de valência e arousal da letra (ANEW-Br).

Fonte de rótulo **descartada** do pipeline principal: concorda pouco com os
leitores humanos e inverte o sinal em letras de protesto (Seção 4.1 do artigo).
O código fica para reproduzir essa comparação.

Casa os lemas da letra com as normas afetivas do ANEW-Br (Kristensen et al.,
cerca de 1.050 palavras com valência e arousal no SAM 1-9) e tira a média. O
léxico ignora negação, ironia, gíria e contexto.

O léxico entra por :class:`AnewBrLexicon` e o lematizador é injetável. Sem
lematizador, os tokens são usados como estão e ``meta["lemmatized"]`` registra a
limitação. O ANEW-Br tem licença própria e **não é versionado**: aponte para o
arquivo local.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .lf_common import LFRawOutput
from .scale import to_unit

__all__ = [
    "AnewBrLexicon",
    "SAM_LO",
    "SAM_HI",
    "tokenize",
    "make_spacy_lemmatizer",
    "estimate_va_from_lyrics",
    "run_batch",
]

# Escala nativa das normas ANEW-Br (SAM), convertida na borda para [-1, 1].
SAM_LO, SAM_HI = 1.0, 9.0

# Sequências de letras (com acentos), sem dígitos nem pontuação.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

Lemmatizer = Callable[[list[str]], list[str]]


@dataclass
class AnewBrLexicon:
    """Mapa lema -> ``(valência, arousal)`` na escala SAM 1-9.

    Use :meth:`from_csv` ou :meth:`from_anew_br_txt` para o léxico real. O
    construtor aceita um dict, o que facilita testes com um mini-léxico.
    """

    norms: dict[str, tuple[float, float]]

    def __post_init__(self) -> None:
        self.norms = {word.lower(): norm for word, norm in self.norms.items()}

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        word_col: str = "word",
        valence_col: str = "valence_mean",
        arousal_col: str = "arousal_mean",
        sep: str = ",",
        decimal: str = ".",
        encoding: str = "utf-8",
    ) -> "AnewBrLexicon":
        """Carrega o léxico de um arquivo tabular (palavra, valência, arousal).

        Args:
            path: arquivo do léxico.
            word_col: coluna da palavra ou lema.
            valence_col: coluna da valência média (SAM 1-9).
            arousal_col: coluna do arousal médio (SAM 1-9).
            sep: separador de colunas (``r"\\s+"`` para espaços).
            decimal: caractere decimal.
            encoding: codificação do arquivo.
        """
        df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding, engine="python")
        for column in (word_col, valence_col, arousal_col):
            if column not in df.columns:
                raise ValueError(f"coluna '{column}' ausente em {path}: {list(df.columns)}")
        norms = {
            str(word).lower(): (float(valence), float(arousal))
            for word, valence, arousal in zip(df[word_col], df[valence_col], df[arousal_col])
        }
        return cls(norms=norms)

    @classmethod
    def from_anew_br_txt(cls, path: str | Path) -> "AnewBrLexicon":
        """Carrega o ``anew-br.txt`` oficial (github.com/affective-computing/anew-br).

        Formato: UTF-8, separado por espaços, decimal com vírgula e colunas
        ``Palavra Valencia-Media Valencia-DP Alerta-Media Alerta-DP``.
        """
        return cls.from_csv(
            path,
            word_col="Palavra",
            valence_col="Valencia-Media",
            arousal_col="Alerta-Media",
            sep=r"\s+",
            decimal=",",
        )

    def lookup(self, lemma: str) -> tuple[float, float] | None:
        """Normas ``(valência, arousal)`` do lema, ou ``None`` se ele não estiver no léxico."""
        return self.norms.get(lemma.lower())


def tokenize(text: str) -> list[str]:
    """Divide a letra em palavras minúsculas, mantendo acentos e removendo dígitos."""
    return _TOKEN_RE.findall(text.lower()) if text else []


def make_spacy_lemmatizer(model: str = "pt_core_news_sm") -> Lemmatizer:
    """Cria um lematizador PT-BR com spaCy (import preguiçoso).

    Casar lemas ("chorando" e "chorei" viram "chorar") aumenta a cobertura do
    ANEW-Br, que lista formas de dicionário. Setup, uma vez::

        pip install spacy
        python -m spacy download pt_core_news_sm

    Raises:
        ImportError: se o spaCy ou o modelo não estiverem instalados.
    """
    try:
        import spacy  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("spaCy não instalado: `pip install spacy`.") from error
    try:
        nlp = spacy.load(model, disable=["parser", "ner"])
    except OSError as error:
        raise ImportError(
            f"modelo spaCy '{model}' não encontrado: `python -m spacy download {model}`."
        ) from error

    def lemmatize(tokens: list[str]) -> list[str]:
        return [token.lemma_.lower() for token in nlp(" ".join(tokens)) if not token.is_space]

    return lemmatize


def estimate_va_from_lyrics(
    lyrics: str,
    lexicon: AnewBrLexicon,
    *,
    has_lyrics: bool = True,
    lemmatizer: Lemmatizer | None = None,
    weight_by_freq: bool = True,
) -> LFRawOutput:
    """Estima ``(V_raw, A_raw)`` em [-1, 1] da letra, pela média das normas do ANEW-Br.

    Args:
        lyrics: texto completo da letra.
        lexicon: normas ANEW-Br.
        has_lyrics: ``False`` para faixa instrumental, que devolve ``NaN`` (nunca zero).
        lemmatizer: função ``list[str] -> list[str]`` opcional (ex.: spaCy).
        weight_by_freq: se ``True``, média sobre todas as ocorrências casadas (palavras
            frequentes pesam mais); se ``False``, média sobre lemas únicos.

    Returns:
        :class:`LFRawOutput`; ``meta`` traz ``n_tokens``, ``n_matched``, ``coverage``
        e ``lemmatized``. Sem nenhuma palavra casada, V e A saem ``NaN``.
    """
    if not has_lyrics:
        return LFRawOutput(
            "", float("nan"), float("nan"), {"reason": "instrumental", "n_matched": 0}
        )

    tokens = tokenize(lyrics)
    lemmas = lemmatizer(tokens) if lemmatizer is not None else tokens
    if not weight_by_freq:
        lemmas = list(dict.fromkeys(lemmas))

    hits = [norm for lemma in lemmas if (norm := lexicon.lookup(lemma)) is not None]
    meta = {
        "n_tokens": len(tokens),
        "n_matched": len(hits),
        "coverage": len(hits) / len(tokens) if tokens else 0.0,
        "lemmatized": lemmatizer is not None,
        "weight_by_freq": weight_by_freq,
    }
    if not hits:
        return LFRawOutput("", float("nan"), float("nan"), meta)

    valence = float(np.mean([norm[0] for norm in hits]))
    arousal = float(np.mean([norm[1] for norm in hits]))
    return LFRawOutput("", to_unit(valence, SAM_LO, SAM_HI), to_unit(arousal, SAM_LO, SAM_HI), meta)


def run_batch(
    tracks: Iterable[tuple[str, str, bool]],
    lexicon: AnewBrLexicon,
    *,
    lemmatizer: Lemmatizer | None = None,
    weight_by_freq: bool = True,
) -> list[LFRawOutput]:
    """Roda o estimador léxico em várias faixas.

    Args:
        tracks: iterável de ``(track_id, lyrics, has_lyrics)``.
        lexicon: normas ANEW-Br.
        lemmatizer: lematizador opcional.
        weight_by_freq: estratégia de agregação (ver :func:`estimate_va_from_lyrics`).
    """
    results = []
    for track_id, lyrics, has_lyrics in tracks:
        result = estimate_va_from_lyrics(
            lyrics,
            lexicon,
            has_lyrics=has_lyrics,
            lemmatizer=lemmatizer,
            weight_by_freq=weight_by_freq,
        )
        result.track_id = track_id
        results.append(result)
    return results
