#!/usr/bin/env python
"""Extrai letra (Genius) e preview de 30s de áudio (Deezer/iTunes) para o corpus
expandido descrito em ``docs/corpus_expandido_candidatos.csv``.

Lê cada linha (intérprete, faixa, gênero/ritmo, década, autoria já verificada) e:

- Busca a letra completa via Genius (API oficial p/ achar a página + scraping da
  página, já que a API não devolve o texto da letra em si) -> grava
  ``data/corpus/lyrics/<id>.txt``. Gêneros tipicamente instrumentais (frevo,
  vanerão/milonga, guitarrada) NÃO buscam letra — grava ``.txt`` vazio direto,
  mesma convenção de instrumental usada no resto do projeto.
- Busca um preview oficial de 30s via Deezer (sem autenticação); se não achar,
  tenta o fallback iTunes Search API (também sem autenticação) -> grava
  ``data/corpus/audio/<id>.mp3``.
- Grava/atualiza ``data/corpus/manifest_expandido.csv`` (id -> metadados
  verificados: gênero, região, década, tier, intérprete, compositor, autoral,
  faixa, ano, e status de cada fonte) — não existia um mapa id->metadado no
  projeto; este arquivo passa a ser essa referência para o corpus novo.

Linhas sem uma faixa concreta (marcadas "a confirmar", "a pesquisar", "(variado)"
ou "NÃO CONFIRMADO" na planilha) são puladas e reportadas — não recebem ID.

IDs novos usam o mesmo padrão de 3 letras + número de 2 dígitos do corpus atual
(ex. ``MPB10``), continuando a numeração a partir do maior número já existente
em ``data/corpus/audio/`` para aquele prefixo (nunca reaproveita ID de música já
existente). Para os ritmos regionais (sem prefixo prévio no corpus), foram
definidos códigos novos, checados para não colidir com nenhum prefixo já usado
pelas 53 faixas atuais (ver ``CODIGO_POR_GENERO`` abaixo).

**As chaves de API entram por variável de ambiente** (``GENIUS_ACCESS_TOKEN``),
lidas de um ``.env`` na raiz do projeto se existir (``python-dotenv``) — nunca
como argumento nem no código.

Exemplos::

    python scripts/extrair_corpus.py --dry-run              # só mostra o plano
    python scripts/extrair_corpus.py --limit 3               # testa em 3 músicas
    python scripts/extrair_corpus.py --limit 3 --only-lyrics # só letra, sem áudio
    python scripts/extrair_corpus.py                          # roda tudo, retomável
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# console do Windows costuma vir em cp1252, que não sabe imprimir todo Unicode
# (ex.: um título estrangeiro raro num "match_suspeito") — força UTF-8 com
# substituição em vez de deixar o print derrubar o processo no meio da rodada.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # .env é conveniência; sem python-dotenv, exporte as variáveis manualmente

import requests  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuração fixa
# --------------------------------------------------------------------------- #

# genero_macro_ou_ritmo (exatamente como está no CSV) -> prefixo de 3 letras.
# Gêneros já existentes no corpus atual (MPB, SAM, SER, FOR, FNK, RAP, ROC, AXE,
# BOS, PAG) reaproveitam o MESMO prefixo — são o mesmo gênero, só músicas novas.
# Ritmos regionais são todos novos; os prefixos abaixo foram checados contra as
# 53 faixas existentes (ARR, AXE, BOS, FNK, FOR, FRE, INS, MET, MPB, PAG, PIS,
# POP, RAP, ROC, ROM, SAM, SER, SOF, SOU, TRA, TRO) para não colidir — em
# especial "FRE" já existe, por isso frevo usa "FRV".
CODIGO_POR_GENERO = {
    "MPB": "MPB",
    "Samba": "SAM",
    "Sertanejo": "SER",
    "Forró": "FOR",
    "Funk carioca": "FNK",
    "Rap nacional": "RAP",
    "Rock brasileiro": "ROC",
    "Axé": "AXE",
    "Bossa nova": "BOS",
    "Pagode": "PAG",
    "Frevo": "FRV",
    "Maracatu / manguebeat": "MAN",
    "Coco / embolada": "COC",
    "Repente / cordel": "REP",
    "Música gaúcha tradicionalista": "GAU",
    "Vanerão / milonga campeira": "VAN",
    "Fandango caiçara": "FAN",
    "Carimbó": "CAR",
    "Guitarrada paraense": "GUI",
    "Toada de boi-bumbá": "BOI",
    "Toada / rasqueado pantaneiro": "PAN",
    "Catira": "CAT",
}

# Ritmos tipicamente instrumentais: não buscar letra, gravar .txt vazio direto
# (mesma convenção de instrumental já usada no resto do projeto).
RITMOS_INSTRUMENTAIS = {
    "Frevo",
    "Vanerão / milonga campeira",
    "Guitarrada paraense",
}

PLACEHOLDERS_FAIXA = (
    "a confirmar", "a pesquisar", "(variado)", "não confirmado", "nao confirmado",
)

GENIUS_API = "https://api.genius.com"
DEEZER_API = "https://api.deezer.com/search"
ITUNES_API = "https://itunes.apple.com/search"


@dataclass
class Candidato:
    linha: int
    eixo: str
    regiao: str
    genero: str
    decada: str
    tier: str
    interprete: str
    compositor: str
    autoral: str
    faixa: str
    ano: str
    id: str = ""
    instrumental: bool = False
    genius_status: str = "pendente"
    deezer_status: str = "pendente"
    aviso: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Leitura do CSV + atribuição de ID
# --------------------------------------------------------------------------- #
def faixa_e_concreta(faixa: str) -> bool:
    f = (faixa or "").strip().lower()
    if not f:
        return False
    return not any(p in f for p in PLACEHOLDERS_FAIXA)


def ler_candidatos(csv_path: Path) -> tuple[list[Candidato], list[Candidato]]:
    """-> (processáveis, pulados) a partir do corpus_expandido_candidatos.csv."""
    processaveis, pulados = [], []
    with csv_path.open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):  # linha 1 = header
            c = Candidato(
                linha=i, eixo=row["eixo"], regiao=row.get("regiao", ""),
                genero=row["genero_macro_ou_ritmo"], decada=row["decada"],
                tier=row["popularidade_tier"], interprete=row["interprete"],
                compositor=row["compositor"], autoral=row["autoral"],
                faixa=row["faixa"], ano=row["ano_aprox"],
            )
            status_csv = (row.get("status") or "").strip()
            if status_csv.startswith("DUPLICATA"):
                c.aviso.append("duplicata de faixa já existente no corpus original — pulada")
                pulados.append(c)
                continue
            if row.get("autoral") == "coletivo" or not faixa_e_concreta(c.faixa):
                c.aviso.append("faixa não concreta o suficiente para buscar — pulada")
                pulados.append(c)
                continue
            if c.genero not in CODIGO_POR_GENERO:
                c.aviso.append(f"gênero/ritmo sem código mapeado: {c.genero!r}")
                pulados.append(c)
                continue
            c.instrumental = c.genero in RITMOS_INSTRUMENTAIS
            processaveis.append(c)
    return processaveis, pulados


def proximo_numero(prefixo: str, audio_dir: Path) -> int:
    maior = 0
    for f in audio_dir.glob(f"{prefixo}*.mp3"):
        m = re.match(rf"^{prefixo}(\d+)$", f.stem)
        if m:
            maior = max(maior, int(m.group(1)))
    return maior + 1


def atribuir_ids(candidatos: list[Candidato], audio_dir: Path) -> None:
    """Atribui IDs novos, continuando a numeração já existente por prefixo."""
    contador = {}
    for c in candidatos:
        prefixo = CODIGO_POR_GENERO[c.genero]
        if prefixo not in contador:
            contador[prefixo] = proximo_numero(prefixo, audio_dir)
        c.id = f"{prefixo}{contador[prefixo]:02d}"
        contador[prefixo] += 1


# --------------------------------------------------------------------------- #
# Genius (letra)
# --------------------------------------------------------------------------- #
def _primeiro_artista(interprete: str) -> str:
    """'Chico Buarque e Gilberto Gil' -> 'Chico Buarque' (melhora a busca).

    Remove parênteses ANTES de separar (evita cortar no meio de anotações como
    '(intérprete/melodia)' ou '(frevo-de-bloco; gravado por ...)').
    """
    sem_parenteses = re.sub(r"\([^)]*\)", "", interprete).strip()
    base = sem_parenteses or interprete.strip()
    for sep in (" e ", " E ", ",", "/"):
        if sep in base:
            return base.split(sep)[0].strip()
    return base


def _titulo_bate(consultado: str, achado: str, limiar: float = 0.45) -> bool:
    """Confere se o título devolvido pelo Genius realmente parece o que pedimos.

    Sem isso, uma busca sem bom resultado (ex.: peça instrumental rara) pode
    casar por acaso com uma página do Genius sem relação nenhuma (já visto na
    prática: buscar "Favorito" devolveu o syllabus de um curso universitário
    chamado "Popular Music of Latin America", que o código aceitava cegamente
    como se fosse a letra). Usa razão de similaridade simples (difflib), não
    precisa ser exata — só rejeitar o caso claramente errado.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    a, b = norm(consultado), norm(achado)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= limiar


def buscar_letra_genius(interprete: str, faixa: str, token: str,
                        timeout: float = 15.0) -> tuple[str, str]:
    """-> (texto_da_letra, status).

    status em {'ok','vazio','sem_resultado','match_suspeito','erro busca: ...',
    'erro página: ...','sem_container_letra'}.
    """
    headers = {"Authorization": f"Bearer {token}"}
    query = f"{_primeiro_artista(interprete)} {faixa}"
    try:
        r = requests.get(f"{GENIUS_API}/search", params={"q": query},
                         headers=headers, timeout=timeout)
        r.raise_for_status()
        hits = r.json().get("response", {}).get("hits", [])
    except Exception as e:  # noqa: BLE001
        return "", f"erro busca: {e}"
    if not hits:
        return "", "sem_resultado"
    achado = hits[0]["result"]
    if not _titulo_bate(faixa, achado.get("title", "")):
        return "", (f"match_suspeito (buscou {faixa!r}, Genius achou "
                    f"{achado.get('title', '')!r} de "
                    f"{achado.get('primary_artist', {}).get('name', '?')!r})")
    url = achado["url"]
    try:
        # Genius bloqueia (403) User-Agents que não parecem navegador real.
        page_headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/128.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        }
        page = requests.get(url, timeout=timeout, headers=page_headers)
        page.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return "", f"erro página: {e}"
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page.text, "html.parser")
    blocos = soup.select('div[data-lyrics-container="true"]')
    if not blocos:
        return "", "sem_container_letra"
    linhas = []
    for bloco in blocos:
        for br in bloco.find_all("br"):
            br.replace_with("\n")
        linhas.append(bloco.get_text())
    texto = limpar_letra_genius("\n".join(linhas).strip())
    return texto, "ok" if texto else "vazio"


def limpar_letra_genius(texto: str) -> str:
    """Remove ruído estrutural que a página do Genius injeta no container de
    letra — mantém só o texto cantado (mesma convenção do resto do corpus:
    sem tags de seção, sem créditos de contribuidor).

    Remove:
    - O blurb/descrição antes da letra, sempre terminado pelo marcador padrão
      da Genius ``[Letra de "<título>" ...]`` (contadores de contribuidor,
      links de tradução, resumo do "Read More").
    - Linhas inteiras entre colchetes (``[Refrão: ...]``, ``[Verso 1: ...]``,
      ``[Intro]``) — são rótulos de estrutura, não letra cantada.

    Preserva anotações INLINE dentro de uma linha cantada (ex.: "(Cale-se!)",
    "(pai)") — essas fazem parte do que é de fato cantado/gravado.
    """
    # A Genius sempre imprime "<Título> Lyrics" logo no início do container,
    # antes da letra de verdade — às vezes só isso, às vezes com uma bio antes
    # (contadores de contribuidor, links de tradução). Corta pelo primeiro
    # "Lyrics" dentro de uma janela inicial (nunca no meio da letra de verdade,
    # que é em português — "Lyrics" não deveria aparecer ali por acaso).
    janela = texto[:400]
    idx = janela.find("Lyrics")
    if idx != -1:
        texto = texto[idx + len("Lyrics"):]
    # Se tinha bio antes da letra, ela sempre termina em "Read More" seguido do
    # primeiro marcador de estrutura da letra colado na mesma linha — varia
    # (`[Letra de "..."]`, `[Verso 1]`, `[Refrão]` etc.), então ancora em
    # "Read More" (constante) + qualquer colchete logo depois, não no texto
    # exato do colchete.
    m = re.search(r"Read More\s*\[[^\]]*\]", texto[:600])
    if m:
        texto = texto[m.end():]
    linhas = [l for l in texto.split("\n")
              if not re.fullmatch(r"\s*\[[^\]]*\]\s*", l)]
    texto = "\n".join(linhas).strip()
    return re.sub(r"\n{3,}", "\n\n", texto)


# --------------------------------------------------------------------------- #
# Deezer / iTunes (preview de áudio)
# --------------------------------------------------------------------------- #
def buscar_preview_deezer(interprete: str, faixa: str,
                          timeout: float = 15.0) -> tuple[bytes | None, str]:
    # Busca em texto livre (não o filtro artist:"..." track:"...") — testado
    # manualmente: o filtro por campo retorna 0 resultados para vários casos
    # reais do corpus (acentos/diacríticos), enquanto a busca livre acha o
    # resultado certo como 1º hit de forma consistente.
    query = f"{_primeiro_artista(interprete)} {faixa}"
    try:
        r = requests.get(DEEZER_API, params={"q": query}, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:  # noqa: BLE001
        return None, f"erro busca deezer: {e}"
    if not data or not data[0].get("preview"):
        return None, "sem_resultado_deezer"
    if not _titulo_bate(faixa, data[0].get("title", "")):
        return None, (f"match_suspeito_deezer (buscou {faixa!r}, achou "
                      f"{data[0].get('title', '')!r})")
    try:
        audio = requests.get(data[0]["preview"], timeout=timeout)
        audio.raise_for_status()
        return audio.content, "ok_deezer"
    except Exception as e:  # noqa: BLE001
        return None, f"erro download deezer: {e}"


def buscar_preview_itunes(interprete: str, faixa: str,
                          timeout: float = 15.0) -> tuple[bytes | None, str]:
    termo = f"{_primeiro_artista(interprete)} {faixa}"
    try:
        r = requests.get(ITUNES_API, params={"term": termo, "media": "music",
                                              "limit": 1}, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:  # noqa: BLE001
        return None, f"erro busca itunes: {e}"
    if not results or not results[0].get("previewUrl"):
        return None, "sem_resultado_itunes"
    if not _titulo_bate(faixa, results[0].get("trackName", "")):
        return None, (f"match_suspeito_itunes (buscou {faixa!r}, achou "
                      f"{results[0].get('trackName', '')!r})")
    try:
        audio = requests.get(results[0]["previewUrl"], timeout=timeout)
        audio.raise_for_status()
        return audio.content, "ok_itunes"
    except Exception as e:  # noqa: BLE001
        return None, f"erro download itunes: {e}"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
MANIFEST_CAMPOS = [
    "id", "eixo", "regiao", "genero_macro_ou_ritmo", "decada", "popularidade_tier",
    "interprete", "compositor", "autoral", "faixa", "ano_aprox", "instrumental",
    "genius_status", "deezer_status",
]


def gravar_manifest(candidatos: list[Candidato], manifest_path: Path) -> None:
    existentes: dict[str, dict] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existentes[row["id"]] = row
    for c in candidatos:
        existentes[c.id] = {
            "id": c.id, "eixo": c.eixo, "regiao": c.regiao,
            "genero_macro_ou_ritmo": c.genero, "decada": c.decada,
            "popularidade_tier": c.tier, "interprete": c.interprete,
            "compositor": c.compositor, "autoral": c.autoral, "faixa": c.faixa,
            "ano_aprox": c.ano, "instrumental": c.instrumental,
            "genius_status": c.genius_status, "deezer_status": c.deezer_status,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_CAMPOS)
        w.writeheader()
        for row in existentes.values():
            w.writerow(row)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus-csv", type=Path,
                   default=ROOT / "docs" / "corpus_expandido_candidatos.csv")
    p.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "corpus" / "audio")
    p.add_argument("--lyrics-dir", type=Path, default=ROOT / "data" / "corpus" / "lyrics")
    p.add_argument("--manifest-out", type=Path,
                   default=ROOT / "data" / "corpus" / "manifest_expandido.csv")
    p.add_argument("--limit", type=int, default=None,
                   help="processa só as N primeiras músicas processáveis (teste).")
    p.add_argument("--only-lyrics", action="store_true")
    p.add_argument("--only-audio", action="store_true")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="pausa (s) entre músicas, por etiqueta com Genius/Deezer.")
    p.add_argument("--dry-run", action="store_true",
                   help="só mostra o plano (IDs atribuídos), sem rede.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processaveis, pulados = ler_candidatos(args.corpus_csv)
    atribuir_ids(processaveis, args.audio_dir)

    print(f"[extrair_corpus] {len(processaveis)} processável(is), "
          f"{len(pulados)} pulada(s) (faixa não concreta ou gênero sem código).")
    for c in pulados:
        print(f"  PULADA (linha {c.linha}): {c.interprete} — {c.faixa!r} "
              f"[{'; '.join(c.aviso)}]")

    if args.limit:
        processaveis = processaveis[:args.limit]
        print(f"[extrair_corpus] --limit {args.limit}: processando só essas.")

    if args.dry_run:
        for c in processaveis:
            tipo = "INSTRUMENTAL (sem letra)" if c.instrumental else "com letra"
            print(f"  {c.id}  {c.interprete} — {c.faixa} ({c.ano})  [{tipo}]")
        print("[extrair_corpus] --dry-run: nenhuma rede/arquivo tocado.")
        return

    token = None
    if not args.only_audio:
        import os
        token = os.environ.get("GENIUS_ACCESS_TOKEN")
        if not token:
            sys.exit("[extrair_corpus] defina GENIUS_ACCESS_TOKEN no .env "
                     "(ou use --only-audio para pular a letra).")

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.lyrics_dir.mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(processaveis, start=1):
        print(f"[{i}/{len(processaveis)}] {c.id}  {c.interprete} — {c.faixa}")

        lyrics_path = args.lyrics_dir / f"{c.id}.txt"
        if args.only_audio:
            c.genius_status = "pulado (--only-audio)"
        elif c.instrumental:
            lyrics_path.write_text("", encoding="utf-8")
            c.genius_status = "instrumental (sem busca)"
            print("    letra: instrumental, .txt vazio gravado")
        elif lyrics_path.exists() and lyrics_path.read_text(encoding="utf-8").strip():
            c.genius_status = "já existia"
            print("    letra: já existia, mantida")
        else:
            texto, status = buscar_letra_genius(c.interprete, c.faixa, token)
            c.genius_status = status
            lyrics_path.write_text(texto, encoding="utf-8")
            print(f"    letra: {status} ({len(texto)} chars)")

        audio_path = args.audio_dir / f"{c.id}.mp3"
        if args.only_lyrics:
            c.deezer_status = "pulado (--only-lyrics)"
        elif audio_path.exists():
            c.deezer_status = "já existia"
            print("    áudio: já existia, mantido")
        else:
            conteudo, status = buscar_preview_deezer(c.interprete, c.faixa)
            if conteudo is None:
                conteudo, status = buscar_preview_itunes(c.interprete, c.faixa)
            c.deezer_status = status
            if conteudo is not None:
                audio_path.write_bytes(conteudo)
            print(f"    áudio: {status}"
                  f"{'' if conteudo is None else f' ({len(conteudo)} bytes)'}")

        gravar_manifest(processaveis[:i], args.manifest_out)  # incremental
        time.sleep(args.sleep)

    ok_letra = sum(1 for c in processaveis if c.genius_status in
                   ("ok", "instrumental (sem busca)", "já existia"))
    ok_audio = sum(1 for c in processaveis if c.deezer_status in
                   ("ok_deezer", "ok_itunes", "já existia"))
    print(f"\n[extrair_corpus] concluído: {ok_letra}/{len(processaveis)} letras, "
          f"{ok_audio}/{len(processaveis)} áudios. Manifest em {args.manifest_out}")


if __name__ == "__main__":
    main()
