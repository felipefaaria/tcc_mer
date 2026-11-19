#!/usr/bin/env python
"""Testa a LF2 (LLM) e o léxico ANEW-Br em UMA letra, para rodar na mão.

Mesmo espírito do ``docker/testar.ps1`` para o áudio: mostra a saída de cada estimador, com
o detalhe palavra por palavra do léxico e o prompt exato enviado ao LLM.

    python scripts/rotulagem/testar_letra.py MPB02                   # letra do corpus (id)
    python scripts/rotulagem/testar_letra.py --arquivo carta.txt     # um .txt qualquer
    python scripts/rotulagem/testar_letra.py --texto "sua letra aqui"

    python scripts/rotulagem/testar_letra.py MPB02 --so-lexico       # só o léxico (sem API)
    python scripts/rotulagem/testar_letra.py MPB02 --so-llm          # só o LLM
    python scripts/rotulagem/testar_letra.py MPB02 --fake-llm        # LLM sem rede
"""

import argparse
import sys
from pathlib import Path

from tcc_mer.anthropic_adapter import PRICES, make_client
from tcc_mer.env import load_env
from tcc_mer.lexicon import (
    AnewBrLexicon,
    estimate_va_from_lyrics,
    make_spacy_lemmatizer,
    tokenize,
)
from tcc_mer.lf2_llm import PROMPT_PATH, PROMPT_VERSION, build_prompt, estimate_va_from_lyrics_llm
from tcc_mer.paths import LEXICONS_DIR, LYRICS_DIR

TAG = "[testar_letra]"
MAX_WORDS_SHOWN = 40
FAKE_RESPONSE = '{"valence": 3.5, "arousal": 6.0}'


def load_lyrics(args: argparse.Namespace) -> tuple[str, str]:
    """``(rótulo, texto)`` a partir de ``--texto``, ``--arquivo`` ou do id do corpus."""
    if args.texto:
        return "(texto direto)", args.texto.strip()
    if args.arquivo:
        if not args.arquivo.exists():
            sys.exit(f"{TAG} arquivo não encontrado: {args.arquivo}")
        return args.arquivo.name, args.arquivo.read_text(encoding="utf-8", errors="replace").strip()
    if args.id:
        path = LYRICS_DIR / f"{args.id}.txt"
        if not path.exists():
            sys.exit(f"{TAG} letra não encontrada: {path} (use --arquivo ou --texto)")
        return args.id, path.read_text(encoding="utf-8", errors="replace").strip()
    sys.exit(f"{TAG} informe um id do corpus, --arquivo ou --texto.")


def run_lexicon(label: str, lyrics: str, lexicon_path: Path, lemmatize: bool) -> None:
    print("=" * 64)
    print(f"Léxico ANEW-Br (fonte descartada)  ::  {label}")
    print("=" * 64)
    if not lexicon_path.exists():
        sys.exit(f"{TAG} léxico não encontrado: {lexicon_path}")
    lexicon = AnewBrLexicon.from_anew_br_txt(lexicon_path)

    lemmatizer = None
    if lemmatize:
        try:
            lemmatizer = make_spacy_lemmatizer()
        except ImportError as error:
            print(f"[aviso] lematizador indisponível ({error}); seguindo sem ele.")

    tokens = tokenize(lyrics)
    lemmas = lemmatizer(tokens) if lemmatizer else tokens
    print(f"\n1) Tokenização: {len(tokens)} palavras.")
    print("2) Lematização: " + ("ativa (spaCy pt_core_news_sm)" if lemmatizer else "desativada"))
    print("3) Palavras que bateram no léxico:\n")

    hits = []
    for token, lemma in zip(tokens, lemmas):
        norm = lexicon.lookup(lemma)
        if norm is None:
            continue
        hits.append(norm)
        if len(hits) <= MAX_WORDS_SHOWN:
            word = f"'{token}'" if token == lemma else f"'{token}' (lema: '{lemma}')"
            print(f"   {word:34s} -> V={norm[0]:.2f}  A={norm[1]:.2f}  (escala 1-9)")
    if len(hits) > MAX_WORDS_SHOWN:
        print(f"   ... e mais {len(hits) - MAX_WORDS_SHOWN} palavra(s) casada(s).")

    coverage = len(hits) / len(tokens) if tokens else 0.0
    print(f"\n4) Cobertura: {len(hits)}/{len(tokens)} palavras ({coverage:.1%}).")
    if not hits:
        print("\n   Nenhuma palavra bateu: o léxico devolve NaN (sem sinal, nunca zero).")
        return
    print(
        f"5) Média SAM (1-9): V={sum(h[0] for h in hits) / len(hits):.2f}  "
        f"A={sum(h[1] for h in hits) / len(hits):.2f}"
    )
    final = estimate_va_from_lyrics(lyrics, lexicon, lemmatizer=lemmatizer)
    print(f"6) Em [-1, 1]:  V = {final.v_raw:+.2f}   A = {final.a_raw:+.2f}\n")


def run_llm(label: str, lyrics: str, model: str, fake: bool) -> None:
    print("=" * 64)
    print(f"LF2 (LLM zero-shot)  ::  {label}")
    print("=" * 64)
    print(f"\n1) Prompt enviado (versão {PROMPT_VERSION}, template em {PROMPT_PATH.name}):")
    print("-" * 64)
    print(build_prompt(lyrics))
    print("-" * 64)

    if fake:
        client, used_model = (lambda prompt: FAKE_RESPONSE), "fake"
    else:
        load_env()
        import anthropic

        client, _ = make_client(anthropic.Anthropic(), model)
        used_model = model
    print(f"\n2) Enviando ao modelo '{used_model}'...")
    result = estimate_va_from_lyrics_llm(lyrics, client, model_name=used_model)

    print("\n3) Resposta bruta:")
    print("   " + result.meta.get("raw_response", "").strip().replace("\n", "\n   "))
    if result.meta.get("error"):
        print(f"\n   Erro ao interpretar a resposta: {result.meta['error']}")
        return
    print(f"\n   LF2 final:  V = {result.v_raw:+.2f}   A = {result.a_raw:+.2f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("id", nargs="?", help="id de uma letra em data/corpus/lyrics/<id>.txt")
    parser.add_argument("--arquivo", type=Path, help="caminho de um .txt com a letra")
    parser.add_argument("--texto", help="letra digitada na linha de comando")
    parser.add_argument("--lexico", type=Path, default=LEXICONS_DIR / "anew-br.txt")
    parser.add_argument("--so-lexico", action="store_true", help="roda só o léxico")
    parser.add_argument("--so-llm", action="store_true", help="roda só o LLM")
    parser.add_argument("--sem-lematizador", action="store_true", help="léxico sem lematizar")
    parser.add_argument("--model", default="claude-sonnet-5", choices=list(PRICES))
    parser.add_argument(
        "--fake-llm", action="store_true", help="LLM sem rede (só testa o encanamento)"
    )
    args = parser.parse_args()
    if args.so_lexico and args.so_llm:
        sys.exit(f"{TAG} --so-lexico e --so-llm são mutuamente exclusivos.")

    label, lyrics = load_lyrics(args)
    if not lyrics:
        sys.exit(f"{TAG} '{label}' está vazia (instrumental): nada para testar.")
    print(f"\n### Testando: {label}  ({len(lyrics)} caracteres) ###\n")

    if not args.so_llm:
        run_lexicon(label, lyrics, args.lexico, lemmatize=not args.sem_lematizador)
        print()
    if not args.so_lexico:
        run_llm(label, lyrics, args.model, args.fake_llm)


if __name__ == "__main__":
    main()
