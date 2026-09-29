#!/usr/bin/env python
"""Converte ``docs/corpus_historico_final.csv`` para uma planilha .xlsx pronta
para abrir no Google Sheets, com uma aba extra de resumo por ano que destaca
quais anos ainda não têm as ~5 músicas-alvo (ou não têm nenhuma).

Exemplo::

    python scripts/gerar_planilha_corpus.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "corpus" / "audio"
LYRICS_DIR = ROOT / "data" / "corpus" / "lyrics"
META_ALVO = 5  # "pelo menos 5 músicas de cada ano" — meta original do corpus


def _audio_ok(tid: str) -> bool:
    p = AUDIO_DIR / f"{tid}.mp3"
    return p.exists() and p.stat().st_size > 1000


def _letra_ok(tid: str, status: str) -> bool:
    p = LYRICS_DIR / f"{tid}.txt"
    texto_ok = p.exists() and len(p.read_text(encoding="utf-8").strip()) > 0
    return texto_ok or "instrumental" in status.lower()


def adicionar_colunas_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["audio_ok"] = df["id_temp"].apply(_audio_ok)
    df["letra_ok"] = [_letra_ok(tid, status) for tid, status in zip(df["id_temp"], df["status"])]
    df["ambos_ok"] = df["audio_ok"] & df["letra_ok"]
    return df


def construir_resumo(df: pd.DataFrame) -> pd.DataFrame:
    ano_min, ano_max = int(df["ano"].min()), int(df["ano"].max())
    contagem = df.groupby("ano").size()
    audio_ok = df.groupby("ano")["audio_ok"].sum()
    letra_ok = df.groupby("ano")["letra_ok"].sum()
    ambos_ok = df.groupby("ano")["ambos_ok"].sum()
    linhas = []
    for ano in range(ano_min, ano_max + 1):
        n = int(contagem.get(ano, 0))
        n_audio = int(audio_ok.get(ano, 0))
        n_letra = int(letra_ok.get(ano, 0))
        n_ambos = int(ambos_ok.get(ano, 0))
        if n == 0:
            status = "SEM NENHUMA MUSICA"
        elif n_ambos < META_ALVO:
            status = f"faltam {META_ALVO - n_ambos} prontas"
        else:
            status = "ok"
        linhas.append({"ano": ano, "quantidade_corpus": n, "audio_ok": n_audio,
                       "letra_ok": n_letra, "ambos_ok": n_ambos, "status": status})
    return pd.DataFrame(linhas)


def formatar_aba(ws, df: pd.DataFrame, congelar: str = "A2") -> None:
    ws.freeze_panes = congelar
    ws.auto_filter.ref = ws.dimensions
    for col in ws[1]:
        col.font = Font(bold=True)
    for i, coluna in enumerate(df.columns, start=1):
        maior = max([len(str(coluna))] + [len(str(v)) for v in df[coluna].astype(str)])
        ws.column_dimensions[get_column_letter(i)].width = min(maior + 2, 60)


def destacar_resumo(ws, df_resumo: pd.DataFrame) -> None:
    vermelho = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid")
    amarelo = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    verde = PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid")
    for i, status in enumerate(df_resumo["status"], start=2):  # linha 1 = header
        if status == "SEM NENHUMA MUSICA":
            fill = vermelho
        elif status.startswith("faltam"):
            fill = amarelo
        else:
            fill = verde
        for col in range(1, len(df_resumo.columns) + 1):
            ws.cell(row=i, column=col).fill = fill


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, default=ROOT / "docs" / "corpus_historico_final.csv")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "corpus_historico_final.xlsx")
    args = p.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8")
    df = adicionar_colunas_status(df)
    df_resumo = construir_resumo(df)

    faltantes = df_resumo[df_resumo["status"] != "ok"]
    print(f"[planilha] {len(df)} músicas, {df['ano'].nunique()} anos com pelo menos 1 música.")
    print(f"[planilha] áudio OK: {df['audio_ok'].sum()}/{len(df)} | letra OK: {df['letra_ok'].sum()}/{len(df)} "
          f"| ambos OK: {df['ambos_ok'].sum()}/{len(df)}")
    print(f"[planilha] {len(faltantes)} ano(s) abaixo da meta de {META_ALVO} musicas PRONTAS/ano "
          f"({(df_resumo['status'] == 'SEM NENHUMA MUSICA').sum()} sem nenhuma no corpus).")

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="corpus", index=False)
        df_resumo.to_excel(writer, sheet_name="resumo_por_ano", index=False)
        formatar_aba(writer.sheets["corpus"], df)
        formatar_aba(writer.sheets["resumo_por_ano"], df_resumo)
        destacar_resumo(writer.sheets["resumo_por_ano"], df_resumo)

    print(f"[planilha] gravado em {args.out}")


if __name__ == "__main__":
    main()
