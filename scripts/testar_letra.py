#!/usr/bin/env python
"""Testa a LF2 (ANEW-Br) e a LF3 (LLM) em UMA letra — para rodar na mão.

Mesmo espírito do ``docker/testar.ps1`` para o áudio: dá UMA letra e mostra a
saída de cada LF — com detalhe **palavra por palavra** na LF2 (para ver
exatamente como o léxico decide) e o **prompt exato** enviado ao LLM na LF3.

Roda no Python normal do projeto — sem Docker; LF2 e LF3 são leves.

Uso (a partir da raiz do repo):

    python scripts/testar_letra.py MPB02                  # letra do corpus (id)
    python scripts/testar_letra.py --arquivo carta.txt     # arquivo .txt qualquer
    python scripts/testar_letra.py --texto "sua letra aqui, entre aspas"

    python scripts/testar_letra.py MPB02 --so-lf2          # só o léxico (grátis, sem API)
    python scripts/testar_letra.py MPB02 --so-lf3          # só o LLM
    python scripts/testar_letra.py MPB02 --sem-lematizador # ANEW-Br sem lematizar (compara cobertura)
    python scripts/testar_letra.py MPB02 --fake-llm        # LF3 sem rede (testa o encanamento)

Para a LF3 de verdade, defina a chave uma vez:
    setx GEMINI_API_KEY "sua-chave"     (reabra o terminal depois)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # para `import src.*`
sys.path.insert(0, str(SCRIPTS_DIR))   # para `import gerar_lf3` (script irmão)

from src.lf2_lexicon import (  # noqa: E402
    AnewBrLexicon,
    estimate_va_from_lyrics,
    make_spacy_lemmatizer,
    tokenize,
)
from src.lf3_llm import (  # noqa: E402
    PROMPT_VERSION,
    build_prompt,
    estimate_va_from_lyrics_llm,
)

LEXICON_PATH = ROOT / "data" / "lexicons" / "anew-br.txt"
LYRICS_DIR = ROOT / "data" / "corpus" / "lyrics"


# --------------------------------------------------------------------------- #
# Carregar a letra (id do corpus, arquivo, ou texto direto)
# --------------------------------------------------------------------------- #
def carregar_letra(args: argparse.Namespace) -> tuple[str, str]:
    """Devolve ``(rotulo, texto)`` a partir de --texto / --arquivo / id posicional."""
    if args.texto:
        return "(texto direto)", args.texto.strip()
    if args.arquivo:
        p = Path(args.arquivo)
        if not p.exists():
            sys.exit(f"[testar_letra] arquivo não encontrado: {p}")
        return p.name, p.read_text(encoding="utf-8", errors="replace").strip()
    if args.id:
        p = LYRICS_DIR / f"{args.id}.txt"
        if not p.exists():
            sys.exit(f"[testar_letra] letra não encontrada: {p}\n"
                     f"                use --arquivo ou --texto para outra fonte.")
        return args.id, p.read_text(encoding="utf-8", errors="replace").strip()
    sys.exit("[testar_letra] informe um id do corpus, --arquivo ou --texto.")


# --------------------------------------------------------------------------- #
# LF2 — ANEW-Br, com o passo a passo à mostra
# --------------------------------------------------------------------------- #
def rodar_lf2(rotulo: str, letra: str, *, lexico_path: Path, usar_lematizador: bool) -> None:
    print("=" * 64)
    print(f"LF2 — ANEW-Br (léxico)  ::  {rotulo}")
    print("=" * 64)

    if not lexico_path.exists():
        sys.exit(f"[testar_letra] léxico não encontrado: {lexico_path}")
    lexicon = AnewBrLexicon.from_anew_br_txt(lexico_path)

    lemmatizer = None
    if usar_lematizador:
        try:
            lemmatizer = make_spacy_lemmatizer()
        except ImportError as e:
            print(f"[aviso] lematizador indisponível ({e}); seguindo sem ele.")

    # 1) tokenização — quebra a letra em palavras (minúsculas, sem pontuação/dígitos)
    tokens = tokenize(letra)
    # 2) lematização — reduz cada palavra à forma de dicionário ('chorando' -> 'chorar')
    lemas = lemmatizer(tokens) if lemmatizer else tokens

    print(f"\n1) Tokenização: {len(tokens)} palavras na letra.")
    print("2) Lematização: " + ("ATIVA (spaCy pt_core_news_sm)" if lemmatizer
                                else "DESATIVADA (usa a palavra como está)"))
    print("3) Casamento com o léxico — cada palavra que BATEU:\n")

    # 3) casamento — cada lema é procurado no dicionário ANEW-Br (V,A em SAM 1-9)
    vs: list[float] = []
    as_: list[float] = []
    for token, lema in zip(tokens, lemas):
        hit = lexicon.lookup(lema)
        if hit is None:
            continue
        vs.append(hit[0])
        as_.append(hit[1])
        if len(vs) <= 40:
            rotulo_palavra = f"'{token}'" if token == lema else f"'{token}' (lema: '{lema}')"
            print(f"   {rotulo_palavra:34s} -> V={hit[0]:.2f}  A={hit[1]:.2f}  (escala 1-9)")
    if len(vs) > 40:
        print(f"   ... e mais {len(vs) - 40} palavra(s) casada(s) não mostradas.")

    n_tokens = len(tokens)
    n_matched = len(vs)
    cobertura = n_matched / n_tokens if n_tokens else 0.0
    print(f"\n4) Cobertura: {n_matched}/{n_tokens} palavras bateram no léxico ({cobertura:.1%}).")

    if n_matched == 0:
        print("\n   NENHUMA palavra bateu -> LF2 devolve NaN (sem sinal, nunca zero).")
        return

    # 4) agregação — MÉDIA das notas SAM (1-9) de todas as palavras casadas
    print(f"5) Média SAM (1-9) das palavras casadas: "
          f"V={sum(vs)/len(vs):.2f}  A={sum(as_)/len(as_):.2f}")

    # 5) conversão de escala — SAM 1-9 -> projeto [-1,1] (via to_unit, na borda)
    #    reusa a função oficial para garantir que a demo bate com a produção
    oficial = estimate_va_from_lyrics(letra, lexicon, lemmatizer=lemmatizer)
    print("6) Conversão SAM(1-9) -> [-1,1] (to_unit):")
    print(f"\n   >>> LF2 final:  V = {oficial.v_raw:+.2f}   A = {oficial.a_raw:+.2f}\n")


# --------------------------------------------------------------------------- #
# LF3 — LLM zero-shot, com o prompt exato à mostra
# --------------------------------------------------------------------------- #
def rodar_lf3(rotulo: str, letra: str, *, modelo: str, fake: bool) -> None:
    print("=" * 64)
    print(f"LF3 — LLM zero-shot  ::  {rotulo}")
    print("=" * 64)

    prompt = build_prompt(letra)
    print(f"\n1) Prompt enviado (versão {PROMPT_VERSION}, "
          f"template em prompts/lf3_zero_shot.md):")
    print("-" * 64)
    print(prompt)
    print("-" * 64)

    import gerar_lf3  # script irmão — reusa o MESMO adaptador do lote (não duplica)

    if fake:
        client, modelo_usado = gerar_lf3.make_fake_client(), "fake"
    else:
        client, modelo_usado = gerar_lf3.make_gemini_client(
            modelo, temperature=0.0, rpm=10)

    print(f"\n2) Enviando ao modelo '{modelo_usado}'...")
    res = estimate_va_from_lyrics_llm(letra, client, model_name=modelo_usado)

    print("\n3) Resposta bruta do modelo:")
    bruta = res.meta.get("raw_response", "").strip()
    print("   " + bruta.replace("\n", "\n   "))

    if res.meta.get("error"):
        print(f"\n   ERRO ao interpretar a resposta: {res.meta['error']}")
        return
    print(f"\n   >>> LF3 final:  V = {res.v_raw:+.2f}   A = {res.a_raw:+.2f}\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("id", nargs="?", help="id de uma letra em data/corpus/lyrics/<id>.txt")
    p.add_argument("--arquivo", help="caminho de um .txt com a letra")
    p.add_argument("--texto", help="letra digitada direto na linha de comando")
    p.add_argument("--lexico", type=Path, default=LEXICON_PATH,
                   help=f"caminho do léxico ANEW-Br (padrão: {LEXICON_PATH}).")
    p.add_argument("--so-lf2", action="store_true", help="roda só o léxico (ANEW-Br)")
    p.add_argument("--so-lf3", action="store_true", help="roda só o LLM")
    p.add_argument("--sem-lematizador", action="store_true",
                   help="LF2 sem lematizar (compara a cobertura com/sem)")
    p.add_argument("--model", default="auto", help="modelo Gemini (padrão: auto)")
    p.add_argument("--fake-llm", action="store_true",
                   help="LF3 sem rede/API — só testa o encanamento")
    args = p.parse_args()

    if args.so_lf2 and args.so_lf3:
        sys.exit("[testar_letra] --so-lf2 e --so-lf3 são mutuamente exclusivos.")

    rotulo, letra = carregar_letra(args)
    if not letra:
        sys.exit(f"[testar_letra] '{rotulo}' está vazia (instrumental) — "
                 f"nada para testar na letra.")

    print(f"\n### Testando: {rotulo}  ({len(letra)} caracteres) ###\n")

    if not args.so_lf3:
        rodar_lf2(rotulo, letra, lexico_path=args.lexico,
                  usar_lematizador=not args.sem_lematizador)
        print()
    if not args.so_lf2:
        rodar_lf3(rotulo, letra, modelo=args.model, fake=args.fake_llm)


if __name__ == "__main__":
    main()
