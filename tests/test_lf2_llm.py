"""Testes de tcc_mer/lf2_llm.py — LLM zero-shot com client injetado (fake)."""

import numpy as np
import pytest

from tcc_mer.lf2_llm import (
    PROMPT_VERSION,
    build_prompt,
    estimate_va_from_lyrics_llm,
    load_prompt_template,
    parse_response,
    run_batch,
)


def test_prompt_versionado_existe_e_tem_placeholder():
    tmpl = load_prompt_template()
    assert "{lyrics}" in tmpl
    assert "valence" in tmpl and "arousal" in tmpl


def test_build_prompt_insere_letra():
    p = build_prompt("minha letra", template="X {lyrics} Y")
    assert p == "X minha letra Y"


def test_parse_json_limpo():
    v, a = parse_response('{"valence": 7.5, "arousal": 3.0}')
    assert v == 7.5 and a == 3.0


def test_parse_com_cerca_de_codigo():
    raw = 'Claro!\n```json\n{"valence": 2, "arousal": 8}\n```\n'
    v, a = parse_response(raw)
    assert v == 2.0 and a == 8.0


def test_parse_sem_json_levanta():
    with pytest.raises(ValueError):
        parse_response("não sei responder")


def test_parse_fora_da_escala_levanta():
    with pytest.raises(ValueError):
        parse_response('{"valence": 12, "arousal": 5}')


def test_parse_chave_faltando_levanta():
    with pytest.raises(ValueError):
        parse_response('{"valence": 5}')


def test_estimativa_converte_para_unit():
    def client(prompt):
        return '{"valence": 9, "arousal": 1}'

    res = estimate_va_from_lyrics_llm("letra alegre", client, model_name="fake-1")
    assert res.v_raw == pytest.approx(1.0)
    assert res.a_raw == pytest.approx(-1.0)
    assert res.meta["model"] == "fake-1"
    assert res.meta["prompt_version"] == PROMPT_VERSION


def test_instrumental_nan():
    def client(prompt):
        return '{"valence": 5, "arousal": 5}'

    res = estimate_va_from_lyrics_llm("x", client, has_lyrics=False)
    assert np.isnan(res.v_raw) and np.isnan(res.a_raw)


def test_resposta_ruim_vira_nan_com_erro_em_meta():
    """LLM que alucina texto não-JSON: V/A viram NaN, erro registrado."""

    def client(prompt):
        return "desculpe, não posso"

    res = estimate_va_from_lyrics_llm("letra", client, model_name="fake")
    assert np.isnan(res.v_raw)
    assert "error" in res.meta
    assert res.meta["raw_response"] == "desculpe, não posso"


def test_client_recebe_prompt_com_a_letra():
    capturado = {}

    def client(prompt):
        capturado["prompt"] = prompt
        return '{"valence": 5, "arousal": 5}'

    estimate_va_from_lyrics_llm("LETRA_UNICA_123", client)
    assert "LETRA_UNICA_123" in capturado["prompt"]


def test_run_batch():
    def client(prompt):
        return '{"valence": 6, "arousal": 4}'

    tracks = [("t1", "letra a", True), ("t2", "sem letra", False)]
    out = run_batch(tracks, client, model_name="m")
    assert out[0].track_id == "t1"
    assert not np.isnan(out[0].v_raw)
    assert np.isnan(out[1].v_raw)


PROMPT_V1_SHA256 = "28d964fb602274f0bbcc2f40105d5952c44908ab4e3fc11a42e8cf9a6b45a612"


def test_prompt_v1_nao_mudou():
    """O arquivo inteiro é enviado ao modelo, cabeçalho incluído. Editá-lo muda o instrumento
    de medição: se precisar, incremente PROMPT_VERSION e atualize este hash."""
    import hashlib

    from tcc_mer.lf2_llm import PROMPT_PATH

    assert hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest() == PROMPT_V1_SHA256
