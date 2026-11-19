"""LF2 — estimador léxico de Valência-Arousal da letra (ANEW-Br).

Estima (V, A) **da letra** casando os lemas do texto com as normas afetivas do
ANEW-Br (Kristensen et al.: ~1.046 palavras em PT-BR com normas V/A coletadas via
SAM 1-9). É a contrapartida modal exata da LF1 (áudio) — a "pinça" que abre a
dissociação. Forte em Valência, fraca em Arousal (texto carrega mal energia).

Limitações conhecidas (documentar na monografia): o léxico ignora negação,
ironia, gíria e contexto — por isso não é a única fonte textual (a LF3/LLM
cobre o que ele não pega).

Isolamento de dependência: o léxico entra por :class:`AnewBrLexicon` (carregável
de csv) e o lematizador é **injetável** (``lemmatizer``). Sem lematizador PT-BR
disponível, usa-se normalização simples (identidade sobre tokens) e a limitação
fica registrada — exatamente o plano da Fase B.

O ANEW-Br é material de terceiros com licença própria: **não versionar** o csv no
repositório; apontar ``ANEW_BR_PATH`` para o arquivo local.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .lf_common import LFRawOutput
from .scale import to_unit

__all__ = [
    "AnewBrLexicon",
    "tokenize",
    "make_spacy_lemmatizer",
    "estimate_va_from_lyrics",
    "run_batch",
    "SAM_LO",
    "SAM_HI",
]

# Escala nativa das normas ANEW-Br (SAM), convertida na borda para [-1, 1].
SAM_LO, SAM_HI = 1.0, 9.0

# Tokens de letras (unicode-aware): sequências de letras, sem dígitos/pontuação.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

Lemmatizer = Callable[[list[str]], list[str]]


@dataclass
class AnewBrLexicon:
    """Mapa lema -> (valência, arousal) na escala SAM 1-9.

    Use :meth:`from_csv` para carregar o léxico real; o construtor direto aceita
    um dict (conveniente para testes com um mini-léxico sintético).
    """

    norms: dict[str, tuple[float, float]]

    def __post_init__(self) -> None:
        # Normaliza as chaves para minúsculas para casar com os tokens.
        self.norms = {k.lower(): v for k, v in self.norms.items()}

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

        Os parâmetros ``sep``/``decimal``/``encoding`` cobrem formatos além do CSV
        padrão — em particular o ``anew-br.txt`` oficial (separado por espaços,
        decimal com **vírgula**, colunas ``Palavra``/``Valencia-Media``/
        ``Alerta-Media``). Ver :meth:`from_anew_br_txt` para o atalho.

        Args:
            path: caminho do arquivo do léxico (não versionado no repo).
            word_col: nome da coluna de palavra/lema.
            valence_col: nome da coluna de valência média (SAM 1-9).
            arousal_col: nome da coluna de arousal médio (SAM 1-9).
            sep: separador de colunas (``","`` CSV; ``r"\\s+"`` p/ espaços).
            decimal: caractere decimal (``"."`` ou ``","``).
            encoding: codificação do arquivo.
        """
        df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding,
                         engine="python")
        for col in (word_col, valence_col, arousal_col):
            if col not in df.columns:
                raise ValueError(f"coluna '{col}' ausente em {path}: {list(df.columns)}")
        norms = {
            str(w).lower(): (float(v), float(a))
            for w, v, a in zip(df[word_col], df[valence_col], df[arousal_col])
        }
        return cls(norms=norms)

    @classmethod
    def from_anew_br_txt(cls, path: str | Path) -> "AnewBrLexicon":
        """Atalho para o ``anew-br.txt`` oficial (github.com/affective-computing/anew-br).

        Formato: UTF-8, separado por espaços, decimal com vírgula, colunas
        ``Palavra Valencia-Media Valencia-DP Alerta-Media Alerta-DP`` (SAM 1-9).
        """
        return cls.from_csv(
            path,
            word_col="Palavra",
            valence_col="Valencia-Media",
            arousal_col="Alerta-Media",
            sep=r"\s+",
            decimal=",",
            encoding="utf-8",
        )

    def lookup(self, lemma: str) -> tuple[float, float] | None:
        return self.norms.get(lemma.lower())


def tokenize(text: str) -> list[str]:
    """Tokeniza uma letra em palavras minúsculas (mantém acentos, remove dígitos)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def make_spacy_lemmatizer(model: str = "pt_core_news_sm") -> Lemmatizer:
    """Cria um lematizador PT-BR baseado em spaCy (import preguiçoso).

    Casar lemas (ex.: 'chorando'/'chorei' -> 'chorar') aumenta muito a cobertura
    do ANEW-Br, que lista formas de dicionário. É **opcional**: sem spaCy, a LF2
    usa normalização simples (identidade sobre os tokens) e a limitação fica
    registrada em ``meta['lemmatized'] == False`` — como o plano da Fase B prevê.

    Setup (uma vez)::

        pip install spacy
        python -m spacy download pt_core_news_sm

    Args:
        model: modelo spaCy PT-BR a carregar.

    Returns:
        Função ``list[str] -> list[str]`` que devolve o lema de cada token.

    Raises:
        ImportError: se o spaCy ou o modelo não estiverem instalados (rode o
            setup acima, ou siga sem lematizador — a LF2 degrada com elegância).
    """
    try:
        import spacy  # type: ignore
    except ImportError as e:  # pragma: no cover - depende de ambiente externo
        raise ImportError(
            "spaCy não instalado. `pip install spacy` e "
            "`python -m spacy download pt_core_news_sm`, ou rode a LF2 sem "
            "lematizador (normalização simples)."
        ) from e
    try:
        nlp = spacy.load(model, disable=["parser", "ner"])
    except OSError as e:  # pragma: no cover - depende de ambiente externo
        raise ImportError(
            f"modelo spaCy '{model}' não encontrado. Rode "
            f"`python -m spacy download {model}`."
        ) from e

    def lemmatize(tokens: list[str]) -> list[str]:
        doc = nlp(" ".join(tokens))
        return [t.lemma_.lower() for t in doc if not t.is_space]

    return lemmatize


def estimate_va_from_lyrics(
    lyrics: str,
    lexicon: AnewBrLexicon,
    *,
    tem_letra: bool = True,
    lemmatizer: Lemmatizer | None = None,
    weight_by_freq: bool = True,
) -> LFRawOutput:
    """Estima (V_raw, A_raw) em [-1, 1] a partir da letra via ANEW-Br.

    Args:
        lyrics: texto completo da letra.
        lexicon: normas ANEW-Br.
        tem_letra: se ``False`` (faixa instrumental), retorna ``NaN`` explícito —
            LF de letra não se aplica; nunca zero.
        lemmatizer: função opcional ``list[str] -> list[str]`` (ex.: spaCy PT-BR).
            Sem ela, usa normalização simples (os tokens como estão); a limitação
            fica anotada em ``meta['lemmatized']``.
        weight_by_freq: se ``True`` (padrão), agrega por média sobre **todas** as
            ocorrências casadas (palavras frequentes pesam mais); se ``False``,
            média sobre lemas únicos.

    Returns:
        :class:`LFRawOutput` com ``v_raw``/``a_raw`` em [-1, 1] e ``meta`` de
        diagnóstico (n_tokens, n_matched, cobertura, lemmatized).
    """
    if not tem_letra:
        return LFRawOutput("", float("nan"), float("nan"),
                           {"reason": "instrumental", "n_matched": 0})

    tokens = tokenize(lyrics)
    lemmatized = lemmatizer is not None
    lemmas = lemmatizer(tokens) if lemmatized else tokens

    if not weight_by_freq:
        lemmas = list(dict.fromkeys(lemmas))  # únicos, preservando ordem

    vs: list[float] = []
    as_: list[float] = []
    for lemma in lemmas:
        hit = lexicon.lookup(lemma)
        if hit is not None:
            vs.append(hit[0])
            as_.append(hit[1])

    n_tokens = len(tokens)
    n_matched = len(vs)
    meta = {
        "n_tokens": n_tokens,
        "n_matched": n_matched,
        "coverage": (n_matched / n_tokens) if n_tokens else 0.0,
        "lemmatized": lemmatized,
        "weight_by_freq": weight_by_freq,
    }

    if n_matched == 0:
        # Sem palavras no léxico: não há como estimar -> NaN explícito.
        return LFRawOutput("", float("nan"), float("nan"), meta)

    v_mean_sam = float(np.mean(vs))
    a_mean_sam = float(np.mean(as_))
    return LFRawOutput(
        "",
        to_unit(v_mean_sam, SAM_LO, SAM_HI),
        to_unit(a_mean_sam, SAM_LO, SAM_HI),
        meta,
    )


def run_batch(
    tracks: Iterable[tuple[str, str, bool]],
    lexicon: AnewBrLexicon,
    *,
    lemmatizer: Lemmatizer | None = None,
    weight_by_freq: bool = True,
) -> list[LFRawOutput]:
    """Roda a LF2 sobre uma coleção de faixas.

    Args:
        tracks: iterável de ``(track_id, lyrics, tem_letra)``.
        lexicon: normas ANEW-Br.
        lemmatizer: lematizador opcional (ver :func:`estimate_va_from_lyrics`).
        weight_by_freq: estratégia de agregação.

    Returns:
        Lista de :class:`LFRawOutput`, uma por faixa (com ``track_id`` preenchido).
        Persista com :func:`src.lf_common.write_lf_raw` em ``lf2_raw.csv``.
    """
    out: list[LFRawOutput] = []
    for track_id, lyrics, tem_letra in tracks:
        res = estimate_va_from_lyrics(
            lyrics, lexicon, tem_letra=tem_letra,
            lemmatizer=lemmatizer, weight_by_freq=weight_by_freq,
        )
        res.track_id = track_id
        out.append(res)
    return out
