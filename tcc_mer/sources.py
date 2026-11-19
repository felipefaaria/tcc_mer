"""Fontes externas do corpus: letra no Genius e prévia de 30 s de áudio no Deezer e no iTunes.

* **Letra.** A API oficial do Genius acha a página, mas não devolve o texto da letra, que é
  extraído da página.
* **Áudio.** O Deezer e o iTunes oferecem uma prévia de 30 s sem autenticação; o iTunes é a
  alternativa quando o Deezer não acha.

Toda busca valida o título devolvido contra o pedido (:func:`title_matches`). Sem isso, uma
busca sem bom resultado casa por acaso com uma página sem relação: buscar "Favorito" já
devolveu o programa de um curso universitário, que seria aceito como letra.

As funções de busca devolvem ``(conteúdo, status)`` e nunca levantam erro de rede: o
status registra o que aconteceu (``ok``, ``sem_resultado``, ``match_suspeito...``,
``erro ...``) e vai para o manifesto de cada coleta. As chaves de API vêm do ambiente.
"""

import difflib
import re
from pathlib import Path

import requests

__all__ = [
    "first_artist",
    "title_matches",
    "is_concrete_track",
    "genius_hits",
    "deezer_hits",
    "fetch_genius_lyrics",
    "clean_genius_lyrics",
    "fetch_deezer_preview",
    "fetch_itunes_preview",
    "ensure_lyrics",
    "ensure_preview",
]

GENIUS_API = "https://api.genius.com"
DEEZER_API = "https://api.deezer.com/search"
ITUNES_API = "https://itunes.apple.com/search"

REQUEST_TIMEOUT = 15.0
TITLE_SIMILARITY = 0.45
PLACEHOLDER_TITLES = ("a confirmar", "a pesquisar", "(variado)", "não confirmado", "nao confirmado")

# O Genius bloqueia (403) User-Agents que não parecem um navegador.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

NETWORK_ERRORS = (requests.RequestException, ValueError)


def first_artist(performers: str) -> str:
    """Primeiro artista de uma lista, para melhorar a busca.

    ``"Chico Buarque e Gilberto Gil"`` vira ``"Chico Buarque"``. Os parênteses saem antes de
    separar, para não cortar no meio de anotações como ``"(intérprete/melodia)"``.
    """
    base = re.sub(r"\([^)]*\)", "", performers).strip() or performers.strip()
    for separator in (" e ", " E ", ",", "/"):
        if separator in base:
            return base.split(separator)[0].strip()
    return base


def title_matches(searched: str, found: str, threshold: float = TITLE_SIMILARITY) -> bool:
    """Se o título devolvido parece o pedido (razão de similaridade do ``difflib``).

    Não precisa ser exata: o objetivo é rejeitar o caso claramente errado.
    """

    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    a, b = normalize(searched), normalize(found)
    return bool(a and b) and difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def is_concrete_track(title: str | None) -> bool:
    """Se o título é de uma faixa de fato, e não um espaço reservado do planejamento."""
    title = (title or "").strip().lower()
    return bool(title) and not any(p in title for p in PLACEHOLDER_TITLES)


def genius_hits(query: str, token: str, timeout: float = REQUEST_TIMEOUT) -> list[dict]:
    """Resultados da busca do Genius para ``query``.

    Raises:
        requests.RequestException, ValueError: falha de rede ou resposta inválida.
    """
    response = requests.get(
        f"{GENIUS_API}/search",
        params={"q": query},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", {}).get("hits", [])


def deezer_hits(query: str, timeout: float = REQUEST_TIMEOUT) -> list[dict]:
    """Resultados da busca do Deezer para ``query``.

    A busca é em texto livre, e não pelo filtro ``artist:"..." track:"..."``: o filtro devolve
    zero resultados para vários casos reais (acentos), enquanto a busca livre acha a faixa
    certa como primeiro resultado de forma consistente.

    Raises:
        requests.RequestException, ValueError: falha de rede ou resposta inválida.
    """
    response = requests.get(DEEZER_API, params={"q": query}, timeout=timeout)
    response.raise_for_status()
    return response.json().get("data", [])


def fetch_genius_lyrics(
    performer: str, title: str, token: str, timeout: float = REQUEST_TIMEOUT
) -> tuple[str, str]:
    """Letra completa de uma faixa, extraída da página do Genius.

    Returns:
        ``(texto, status)``, com status ``ok``, ``vazio``, ``sem_resultado``,
        ``match_suspeito (...)``, ``sem_container_letra``, ``erro busca: ...`` ou
        ``erro página: ...``.
    """
    try:
        hits = genius_hits(f"{first_artist(performer)} {title}", token, timeout)
    except NETWORK_ERRORS as error:
        return "", f"erro busca: {error}"
    if not hits:
        return "", "sem_resultado"
    found = hits[0]["result"]
    if not title_matches(title, found.get("title", "")):
        artist = found.get("primary_artist", {}).get("name", "?")
        return "", (
            f"match_suspeito (buscou {title!r}, Genius achou "
            f"{found.get('title', '')!r} de {artist!r})"
        )
    try:
        page = requests.get(found["url"], timeout=timeout, headers=BROWSER_HEADERS)
        page.raise_for_status()
    except NETWORK_ERRORS as error:
        return "", f"erro página: {error}"

    from bs4 import BeautifulSoup

    blocks = BeautifulSoup(page.text, "html.parser").select('div[data-lyrics-container="true"]')
    if not blocks:
        return "", "sem_container_letra"
    lines = []
    for block in blocks:
        for line_break in block.find_all("br"):
            line_break.replace_with("\n")
        lines.append(block.get_text())
    text = clean_genius_lyrics("\n".join(lines).strip())
    return text, "ok" if text else "vazio"


def clean_genius_lyrics(text: str) -> str:
    """Remove o ruído estrutural que a página do Genius injeta no container da letra.

    Fica só o texto cantado: sem o resumo antes da letra (contadores de contribuidor, links de
    tradução, "Read More"), sem linhas inteiras entre colchetes (``[Refrão]``, ``[Verso 1]``) e
    sem créditos. Anotações dentro de uma linha cantada, como ``(Cale-se!)``, são mantidas
    porque fazem parte do que é cantado.
    """
    # O container começa com "<Título> Lyrics", às vezes precedido de um resumo. O corte é só
    # na janela inicial, onde "Lyrics" não aparece por acaso numa letra em português.
    head = text[:400]
    index = head.find("Lyrics")
    if index != -1:
        text = text[index + len("Lyrics") :]
    # O resumo termina em "Read More" com o primeiro marcador de estrutura colado na mesma
    # linha (``[Letra de "..."]``, ``[Verso 1]``...); ancora-se em "Read More" e num colchete
    # qualquer logo depois, não no texto exato do colchete.
    summary = re.search(r"Read More\s*\[[^\]]*\]", text[:600])
    if summary:
        text = text[summary.end() :]
    lines = [line for line in text.split("\n") if not re.fullmatch(r"\s*\[[^\]]*\]\s*", line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


def _download_preview(url: str, timeout: float, source: str) -> tuple[bytes | None, str]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except NETWORK_ERRORS as error:
        return None, f"erro download {source}: {error}"
    return response.content, f"ok_{source}"


def fetch_deezer_preview(
    performer: str, title: str, timeout: float = REQUEST_TIMEOUT
) -> tuple[bytes | None, str]:
    """Prévia de 30 s do Deezer, como ``(bytes, status)``; ``bytes`` é ``None`` se falhar."""
    try:
        hits = deezer_hits(f"{first_artist(performer)} {title}", timeout)
    except NETWORK_ERRORS as error:
        return None, f"erro busca deezer: {error}"
    if not hits or not hits[0].get("preview"):
        return None, "sem_resultado_deezer"
    if not title_matches(title, hits[0].get("title", "")):
        return None, f"match_suspeito_deezer (buscou {title!r}, achou {hits[0].get('title', '')!r})"
    return _download_preview(hits[0]["preview"], timeout, "deezer")


def fetch_itunes_preview(
    performer: str, title: str, timeout: float = REQUEST_TIMEOUT
) -> tuple[bytes | None, str]:
    """Prévia de 30 s do iTunes, como ``(bytes, status)``; ``bytes`` é ``None`` se falhar."""
    try:
        response = requests.get(
            ITUNES_API,
            params={"term": f"{first_artist(performer)} {title}", "media": "music", "limit": 1},
            timeout=timeout,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except NETWORK_ERRORS as error:
        return None, f"erro busca itunes: {error}"
    if not results or not results[0].get("previewUrl"):
        return None, "sem_resultado_itunes"
    if not title_matches(title, results[0].get("trackName", "")):
        return (
            None,
            f"match_suspeito_itunes (buscou {title!r}, achou {results[0].get('trackName', '')!r})",
        )
    return _download_preview(results[0]["previewUrl"], timeout, "itunes")


def ensure_lyrics(path: Path, performer: str, title: str, token: str) -> str:
    """Garante a letra em ``path`` e devolve o status; uma letra já salva é mantida."""
    if path.exists() and path.read_text(encoding="utf-8").strip():
        return "já existia"
    text, status = fetch_genius_lyrics(performer, title, token)
    path.write_text(text, encoding="utf-8")
    return status


def ensure_preview(path: Path, performer: str, title: str) -> str:
    """Garante a prévia de áudio em ``path`` (Deezer, depois iTunes) e devolve o status."""
    if path.exists():
        return "já existia"
    content, status = fetch_deezer_preview(performer, title)
    if content is None:
        content, status = fetch_itunes_preview(performer, title)
    if content is not None:
        path.write_bytes(content)
    return status
