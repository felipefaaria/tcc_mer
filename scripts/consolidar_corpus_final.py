"""
Consolida o corpus histórico (590 candidatos) no corpus FINAL: apenas as
músicas com áudio + letra (ou instrumental confirmado) prontos, com IDs
reorganizados no formato <ano>-<seq> e colunas enxutas para o dataset MER.

Roda em modo --dry-run por padrão (só mostra o que faria). Use --apply para
de fato renomear os arquivos de áudio/letra e gravar o CSV final.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_ORIGEM = ROOT / "docs" / "corpus_historico_final.csv"
AUDIO_DIR = ROOT / "data" / "corpus" / "audio"
LYRICS_DIR = ROOT / "data" / "corpus" / "lyrics"
CSV_DESTINO = ROOT / "docs" / "corpus_final.csv"

FONTE_FIX = {
    "Can�o da Hist�ria (letra manual: MPB Cifrantiga/IMS)":
        "Canção da História (letra manual: MPB Cifrantiga/IMS)",
    "Can�o da Hist�ria": "Canção da História",
}

COLUNAS_FINAIS = [
    "id", "ano", "faixa", "interprete", "genero", "contexto", "fonte",
    "instrumental", "arquivo_audio", "arquivo_letra", "valence", "arousal",
]


def audio_ok(tid: str) -> bool:
    p = AUDIO_DIR / f"{tid}.mp3"
    return p.exists() and p.stat().st_size > 1000


def letra_texto_ok(tid: str) -> bool:
    p = LYRICS_DIR / f"{tid}.txt"
    if not p.exists():
        return False
    return bool(p.read_text(encoding="utf-8", errors="ignore").strip())


def ler_candidatos() -> list[dict]:
    with open(CSV_ORIGEM, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def selecionar_completos(rows: list[dict]) -> list[dict]:
    completos = []
    for r in rows:
        tid = r["id_temp"]
        instrumental = "instrumental" in r["status"].lower()
        letra_ok = instrumental or letra_texto_ok(tid)
        if audio_ok(tid) and letra_ok:
            r["_instrumental"] = instrumental
            completos.append(r)
    return completos


def montar_linha_final(r: dict, novo_id: str) -> dict:
    fonte = r["fonte"].strip()
    fonte = FONTE_FIX.get(fonte, fonte)
    return {
        "id": novo_id,
        "ano": r["ano"],
        "faixa": r["faixa"],
        "interprete": r["interprete_compositor"],
        "genero": r["genero"],
        "contexto": r["nota_sucesso"].strip(),
        "fonte": fonte,
        "instrumental": "sim" if r["_instrumental"] else "não",
        "arquivo_audio": f"data/corpus/audio/{novo_id}.mp3",
        "arquivo_letra": f"data/corpus/lyrics/{novo_id}.txt",
        "valence": "",
        "arousal": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                     help="Executa de fato (renomeia arquivos, grava CSV). Sem essa flag, é dry-run.")
    args = ap.parse_args()

    candidatos = ler_candidatos()
    completos = selecionar_completos(candidatos)
    completos.sort(key=lambda r: (int(r["ano"]), r["faixa"]))

    contagem_ano: dict[int, int] = {}
    linhas_finais = []
    renomeacoes = []  # (old_id, new_id)
    for r in completos:
        ano = int(r["ano"])
        contagem_ano[ano] = contagem_ano.get(ano, 0) + 1
        novo_id = f"{ano}-{contagem_ano[ano]:02d}"
        renomeacoes.append((r["id_temp"], novo_id))
        linhas_finais.append(montar_linha_final(r, novo_id))

    print(f"[consolidar] {len(linhas_finais)} músicas completas selecionadas "
          f"(de {len(candidatos)} candidatas).")
    print(f"[consolidar] anos cobertos: {min(contagem_ano)}–{max(contagem_ano)}")

    if not args.apply:
        print("\n--- DRY RUN (nada foi alterado) ---")
        for old, new in renomeacoes[:10]:
            print(f"  {old} -> {new}")
        print("  ...")
        return

    # 1. renomear arquivos de áudio/letra para os novos IDs (as 516 completas)
    ids_completos = {old for old, _new in renomeacoes}
    for old_id, new_id in renomeacoes:
        old_audio = AUDIO_DIR / f"{old_id}.mp3"
        old_lyrics = LYRICS_DIR / f"{old_id}.txt"
        new_audio = AUDIO_DIR / f"{new_id}.mp3"
        new_lyrics = LYRICS_DIR / f"{new_id}.txt"
        if old_audio.exists():
            shutil.move(str(old_audio), str(new_audio))
        if old_lyrics.exists():
            shutil.move(str(old_lyrics), str(new_lyrics))

    # 2. apagar os arquivos parciais das músicas H* que ficaram incompletas
    todos_ids_candidatos = {r["id_temp"] for r in candidatos}
    incompletos = todos_ids_candidatos - ids_completos
    apagados = 0
    for tid in incompletos:
        for pasta, ext in ((AUDIO_DIR, ".mp3"), (LYRICS_DIR, ".txt")):
            p = pasta / f"{tid}{ext}"
            if p.exists():
                p.unlink()
                apagados += 1
    print(f"[consolidar] {apagados} arquivo(s) parcial(is) de músicas incompletas removido(s).")

    # 3. gravar o CSV final
    with open(CSV_DESTINO, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_FINAIS)
        w.writeheader()
        w.writerows(linhas_finais)

    print(f"[consolidar] arquivos renomeados e CSV final gravado em {CSV_DESTINO}")


if __name__ == "__main__":
    main()
