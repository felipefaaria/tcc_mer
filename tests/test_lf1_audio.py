"""Testes de src/lf1_audio.py — wrapper Music2Emo com predictor injetado (fake).

A dependência pesada (music2emo) NUNCA é importada aqui: injetamos um predictor
falso, provando que a lógica de escala/persistência/robustez é testável sem o
modelo real.
"""
import numpy as np
import pytest

from src.lf1_audio import estimate_va_from_audio, run_batch
from src.lf_common import write_lf_raw, read_lf_raw
from src.scale import to_unit


def fake_predictor(audio_path):
    """Predictor determinístico: devolve 1-9 conforme o nome do arquivo."""
    table = {
        "alegre.mp3": {"valence": 9.0, "arousal": 7.0, "predicted_moods": ["happy"]},
        "triste.mp3": {"valence": 1.0, "arousal": 2.0, "predicted_moods": ["sad"]},
        "neutro.mp3": {"valence": 5.0, "arousal": 5.0, "predicted_moods": []},
    }
    return table[audio_path]


def test_converte_escala_1_9_para_unit():
    res = estimate_va_from_audio("neutro.mp3", fake_predictor)
    assert res.v_raw == pytest.approx(0.0)  # 5 no meio de 1-9 -> 0
    assert res.a_raw == pytest.approx(0.0)


def test_bordas():
    alegre = estimate_va_from_audio("alegre.mp3", fake_predictor)
    triste = estimate_va_from_audio("triste.mp3", fake_predictor)
    assert alegre.v_raw == pytest.approx(1.0)
    assert triste.v_raw == pytest.approx(to_unit(1.0, 1, 9))
    assert triste.v_raw == pytest.approx(-1.0)


def test_moods_em_meta():
    res = estimate_va_from_audio("alegre.mp3", fake_predictor)
    assert res.meta["predicted_moods"] == ["happy"]


def test_run_batch_smoke():
    tracks = [("t1", "alegre.mp3"), ("t2", "triste.mp3"), ("t3", "neutro.mp3")]
    out = run_batch(tracks, fake_predictor)
    assert [o.track_id for o in out] == ["t1", "t2", "t3"]
    assert out[0].v_raw > out[2].v_raw > out[1].v_raw


def test_run_batch_erro_vira_nan_sem_parar():
    """on_error='nan': uma faixa que quebra não interrompe o lote."""
    def flaky(path):
        if path == "quebra.mp3":
            raise RuntimeError("decodificação falhou")
        return {"valence": 5.0, "arousal": 5.0}

    tracks = [("bom", "ok.mp3"), ("ruim", "quebra.mp3")]
    out = run_batch(tracks, flaky, on_error="nan")
    assert not np.isnan(out[0].v_raw)
    assert np.isnan(out[1].v_raw)
    assert "error" in out[1].meta


def test_run_batch_erro_raise():
    def flaky(path):
        raise RuntimeError("falhou")

    with pytest.raises(RuntimeError):
        run_batch([("x", "a.mp3")], flaky, on_error="raise")


def test_persistencia_roundtrip(tmp_path):
    tracks = [("t1", "alegre.mp3"), ("t2", "triste.mp3")]
    out = run_batch(tracks, fake_predictor)
    path = tmp_path / "lf1_raw.csv"
    df = write_lf_raw(out, path)
    assert list(df.columns) == ["id", "V_raw", "A_raw"]
    back = read_lf_raw(path)
    assert list(back["id"]) == ["t1", "t2"]
    assert back["V_raw"].iloc[0] == pytest.approx(1.0)


def test_make_predictor_sem_music2emo_da_import_error():
    """Sem o pacote instalado, o factory falha de forma clara (Plano B).

    Pulado se o music2emo estiver presente (ex.: rodando no venv isolado), pois
    lá o factory tentaria baixar/instanciar o modelo real.
    """
    import importlib.util
    if importlib.util.find_spec("music2emo") is not None:
        pytest.skip("music2emo instalado neste ambiente; teste não se aplica")
    from src.lf1_audio import make_music2emo_predictor
    with pytest.raises(ImportError):
        make_music2emo_predictor()
