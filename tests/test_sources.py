"""Testes de tcc_mer/sources.py, sem rede: ``requests.get`` é substituído por um falso."""

import pytest

requests = pytest.importorskip("requests")

from tcc_mer import sources  # noqa: E402


class FakeResponse:
    def __init__(self, payload=None, content=b"", text="", status=200):
        self._payload, self.content, self.text, self.status = payload, content, text, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def json(self):
        return self._payload


def fake_get(routes):
    """Cria um ``requests.get`` falso: ``routes`` mapeia um trecho da URL à resposta."""

    def get(url, **kwargs):
        for fragment, response in routes.items():
            if fragment in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"URL inesperada: {url}")

    return get


@pytest.mark.parametrize(
    ("performers", "expected"),
    [
        ("Chico Buarque e Gilberto Gil", "Chico Buarque"),
        ("Tom Jobim/Vinícius de Moraes", "Tom Jobim"),
        ("Pixinguinha (intérprete/melodia)", "Pixinguinha"),
        ("Cartola", "Cartola"),
        ("(anônimo)", "(anônimo)"),
    ],
)
def test_first_artist(performers, expected):
    assert sources.first_artist(performers) == expected


def test_titulo_parecido_e_aceito_e_sem_relacao_e_rejeitado():
    assert sources.title_matches("Águas de Março", "Aguas de Marco")
    assert not sources.title_matches("Favorito", "Popular Music of Latin America")
    assert not sources.title_matches("", "qualquer")


def test_faixa_concreta():
    assert sources.is_concrete_track("Asa Branca")
    assert not sources.is_concrete_track("a confirmar")
    assert not sources.is_concrete_track("  ")
    assert not sources.is_concrete_track(None)


def test_limpeza_da_letra_do_genius():
    raw = (
        'Asa Branca Lyrics\nResumo da música Read More [Letra de "Asa Branca"]\n'
        "[Verso 1]\nQuando olhei a terra ardendo\n(pai)\n\n\n\n[Refrão]\nAi, ai\n"
    )
    assert sources.clean_genius_lyrics(raw) == "Quando olhei a terra ardendo\n(pai)\n\nAi, ai"


def test_deezer_devolve_a_previa(monkeypatch):
    hits = {"data": [{"title": "Aguas de Marco", "preview": "https://cdn/preview.mp3"}]}
    monkeypatch.setattr(
        requests,
        "get",
        fake_get(
            {
                "api.deezer.com": FakeResponse(hits),
                "cdn/preview": FakeResponse(content=b"audio"),
            }
        ),
    )
    assert sources.fetch_deezer_preview("Elis Regina", "Águas de Março") == (b"audio", "ok_deezer")


def test_deezer_rejeita_titulo_sem_relacao(monkeypatch):
    hits = {"data": [{"title": "Completamente Outra Coisa", "preview": "https://cdn/p.mp3"}]}
    monkeypatch.setattr(requests, "get", fake_get({"api.deezer.com": FakeResponse(hits)}))
    content, status = sources.fetch_deezer_preview("Elis Regina", "Águas de Março")
    assert content is None and status.startswith("match_suspeito_deezer")


def test_deezer_sem_resultado_e_erro_de_rede(monkeypatch):
    monkeypatch.setattr(requests, "get", fake_get({"api.deezer.com": FakeResponse({"data": []})}))
    assert sources.fetch_deezer_preview("x", "y") == (None, "sem_resultado_deezer")

    monkeypatch.setattr(
        requests, "get", fake_get({"api.deezer.com": requests.ConnectionError("fora")})
    )
    content, status = sources.fetch_deezer_preview("x", "y")
    assert content is None and status.startswith("erro busca deezer")


def test_garantir_previa_tenta_o_itunes_quando_o_deezer_falha(monkeypatch, tmp_path):
    itunes = {"results": [{"trackName": "Aguas de Marco", "previewUrl": "https://cdn/it.m4a"}]}
    monkeypatch.setattr(
        requests,
        "get",
        fake_get(
            {
                "api.deezer.com": FakeResponse({"data": []}),
                "itunes.apple.com": FakeResponse(itunes),
                "cdn/it": FakeResponse(content=b"it"),
            }
        ),
    )
    path = tmp_path / "a.mp3"
    assert sources.ensure_preview(path, "Elis Regina", "Águas de Março") == "ok_itunes"
    assert path.read_bytes() == b"it"
    assert sources.ensure_preview(path, "Elis Regina", "Águas de Março") == "já existia"


def test_garantir_letra_mantem_arquivo_existente(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", fake_get({}))  # qualquer chamada falharia
    path = tmp_path / "l.txt"
    path.write_text("letra salva", encoding="utf-8")
    assert sources.ensure_lyrics(path, "x", "y", token="t") == "já existia"
    assert path.read_text(encoding="utf-8") == "letra salva"


def test_letra_do_genius_com_titulo_suspeito(monkeypatch):
    hits = {
        "response": {
            "hits": [
                {
                    "result": {
                        "title": "Outra Música",
                        "url": "https://g/x",
                        "primary_artist": {"name": "Alguém"},
                    }
                }
            ]
        }
    }
    monkeypatch.setattr(requests, "get", fake_get({"api.genius.com": FakeResponse(hits)}))
    text, status = sources.fetch_genius_lyrics("Cartola", "O Mundo é um Moinho", token="t")
    assert text == "" and status.startswith("match_suspeito")
