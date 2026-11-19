"""Testes dos scripts de coleta que operam sobre arquivos em disco, com uma coleta sintética.

Esses scripts rodam sobre o estado da coleta, anterior à consolidação dos ids, que já não existe
no disco real. Aqui o estado é reconstruído numa pasta temporária.
"""

import csv
import sys

import pandas as pd
import pytest

from tcc_mer import corpus_files
from tests.helpers import load_script

AUDIO_BYTES = b"\xff\xfb" + b"\x00" * 2000  # quadro de sincronia de MP3, acima do tamanho mínimo


@pytest.fixture
def collection(tmp_path, monkeypatch):
    """Coleta sintética: H0001 completa, H0002 instrumental, H0003 sem letra, H0004 sem áudio."""
    audio, lyrics, corpus = tmp_path / "audio", tmp_path / "lyrics", tmp_path / "corpus"
    for folder in (audio, lyrics, corpus):
        folder.mkdir()
    rows = [
        ("H0001", 1990, "Música A", "x", "samba", "ok", "livro", ""),
        ("H0002", 1990, "Música B", "y", "choro", "ok", "livro", "instrumental"),
        ("H0003", 1991, "Música C", "z", "rock", "ok", "rádio", ""),
        ("H0004", 1991, "Música D", "w", "pop", "ok", "rádio", ""),
    ]
    with (corpus / "candidatas_historico.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "id_temp",
                "ano",
                "faixa",
                "interprete_compositor",
                "genero",
                "nota_sucesso",
                "fonte",
                "status",
            ]
        )
        writer.writerows(rows)
    for track_id in ("H0001", "H0002", "H0003"):
        (audio / f"{track_id}.mp3").write_bytes(AUDIO_BYTES)
    (lyrics / "H0001.txt").write_text("uma letra", encoding="utf-8")
    (lyrics / "H0002.txt").write_text("", encoding="utf-8")
    (lyrics / "H0004.txt").write_text("letra sem áudio", encoding="utf-8")
    monkeypatch.setattr(corpus_files, "AUDIO_DIR", audio)
    monkeypatch.setattr(corpus_files, "LYRICS_DIR", lyrics)
    return {"audio": audio, "lyrics": lyrics, "corpus": corpus}


def test_checagem_de_arquivos(collection):
    assert corpus_files.has_audio("H0001") and not corpus_files.has_audio("H0004")
    assert corpus_files.has_lyrics_text("H0001")
    assert not corpus_files.has_lyrics_text("H0002")  # vazio marca instrumental
    assert not corpus_files.has_lyrics_text("H0003")  # sem arquivo


def test_consolidacao_em_ensaio_nao_altera_nada(collection, monkeypatch):
    module = load_script("scripts/coleta/consolidar_corpus_historico.py")
    for name, value in (
        ("SOURCE_CSV", collection["corpus"] / "candidatas_historico.csv"),
        ("TARGET_CSV", collection["corpus"] / "manifest_historico.csv"),
        ("AUDIO_DIR", collection["audio"]),
        ("LYRICS_DIR", collection["lyrics"]),
    ):
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(sys, "argv", ["consolidar"])
    module.main()
    assert not (collection["corpus"] / "manifest_historico.csv").exists()
    assert (collection["audio"] / "H0001.mp3").exists()


def test_consolidacao_renomeia_apaga_parciais_e_grava_o_csv(collection, monkeypatch):
    module = load_script("scripts/coleta/consolidar_corpus_historico.py")
    target = collection["corpus"] / "manifest_historico.csv"
    for name, value in (
        ("SOURCE_CSV", collection["corpus"] / "candidatas_historico.csv"),
        ("TARGET_CSV", target),
        ("AUDIO_DIR", collection["audio"]),
        ("LYRICS_DIR", collection["lyrics"]),
    ):
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(sys, "argv", ["consolidar", "--apply"])

    module.main()

    final = pd.read_csv(target).set_index("id")
    assert list(final.index) == ["1990-01", "1990-02"]  # só H0001 e H0002 estão completas
    assert final.loc["1990-01", "interprete"] == "x"
    assert final.loc["1990-02", "instrumental"] == "sim"
    assert final.loc["1990-01", "arquivo_audio"] == "data/corpus/audio/1990-01.mp3"
    assert sorted(p.name for p in collection["audio"].iterdir()) == ["1990-01.mp3", "1990-02.mp3"]
    # o áudio parcial de H0003 e a letra parcial de H0004 foram apagados
    assert sorted(p.name for p in collection["lyrics"].iterdir()) == ["1990-01.txt", "1990-02.txt"]


def test_auditoria_conta_audio_letra_e_instrumentais(collection):
    module = load_script("scripts/coleta/auditar_coleta.py")
    with (collection["corpus"] / "candidatas_historico.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    result = module.audit(rows)
    lines, totals = module.build_report(len(rows), result)

    assert totals == {"audio": 3, "lyrics": 3, "both": 2, "below_target": 2}
    assert result["n_instrumental"] == 1  # H0002: instrumental confirmado, de letra vazia
    assert [entry[0] for entry in result["missing_audio"]] == ["H0004"]
    assert [entry[0] for entry in result["missing_lyrics"]] == ["H0003"]
    report = "\n".join(lines)
    assert "Áudio confirmado: **3/4**" in report
    assert "- H0004 (1991)" in report


def test_planilha_resume_por_ano_com_status(collection):
    module = load_script("scripts/coleta/gerar_planilha_candidatas.py")
    corpus = pd.read_csv(collection["corpus"] / "candidatas_historico.csv")

    summary = module.summarize_by_year(module.add_status_columns(corpus)).set_index("ano")

    assert summary.loc[1990, "ambos_ok"] == 2
    assert summary.loc[1991, "ambos_ok"] == 0
    assert summary.loc[1990, "status"] == f"faltam {corpus_files.TARGET_PER_YEAR - 2} prontas"


@pytest.mark.parametrize(
    ("head", "expected"),
    [(b"ID3\x04", True), (b"\xff\xfb\x90\x00", True), (b"\x00\x00\x00\x20ftyp", False)],
)
def test_assinatura_de_mp3(tmp_path, head, expected):
    pytest.importorskip("imageio_ffmpeg")
    module = load_script("scripts/coleta/corrigir_audio_m4a.py")
    path = tmp_path / "a.mp3"
    path.write_bytes(head + b"\x00" * 10)
    assert module.is_real_mp3(path) is expected
