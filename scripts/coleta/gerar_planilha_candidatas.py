#!/usr/bin/env python
"""Converte ``data/corpus/candidatas_historico.csv`` numa planilha .xlsx para o Google Sheets.

A planilha tem a aba do corpus e uma aba de resumo por ano que destaca os anos sem as cinco faixas
prontas (ou sem nenhuma).

    python scripts/coleta/gerar_planilha_candidatas.py
"""

import argparse
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from tcc_mer.corpus_files import TARGET_PER_YEAR, has_audio, has_lyrics_text
from tcc_mer.paths import CORPUS_DIR, OUTPUTS_DIR

NO_TRACKS = "SEM NENHUMA MUSICA"
FILLS = {
    "missing": PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid"),
    "short": PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid"),
    "ok": PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid"),
}
MAX_COLUMN_WIDTH = 60


def add_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ``audio_ok``, ``letra_ok`` (letra ou instrumental) e ``ambos_ok``."""
    df = df.copy()
    df["audio_ok"] = df["id_temp"].map(has_audio)
    df["letra_ok"] = [
        has_lyrics_text(track_id) or "instrumental" in str(status).lower()
        for track_id, status in zip(df["id_temp"], df["status"])
    ]
    df["ambos_ok"] = df["audio_ok"] & df["letra_ok"]
    return df


def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por ano, de ``min`` a ``max``, com as contagens e o status frente à meta."""
    by_year = df.groupby("ano")
    totals = by_year.size()
    audio, lyrics, both = (by_year[column].sum() for column in ("audio_ok", "letra_ok", "ambos_ok"))
    rows = []
    for year in range(int(df["ano"].min()), int(df["ano"].max()) + 1):
        n, n_ready = int(totals.get(year, 0)), int(both.get(year, 0))
        if n == 0:
            status = NO_TRACKS
        elif n_ready < TARGET_PER_YEAR:
            status = f"faltam {TARGET_PER_YEAR - n_ready} prontas"
        else:
            status = "ok"
        rows.append(
            {
                "ano": year,
                "quantidade_corpus": n,
                "audio_ok": int(audio.get(year, 0)),
                "letra_ok": int(lyrics.get(year, 0)),
                "ambos_ok": n_ready,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def format_sheet(sheet, df: pd.DataFrame) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for index, column in enumerate(df.columns, start=1):
        widest = max([len(str(column))] + [len(str(value)) for value in df[column]])
        sheet.column_dimensions[get_column_letter(index)].width = min(widest + 2, MAX_COLUMN_WIDTH)


def highlight_summary(sheet, summary: pd.DataFrame) -> None:
    for row, status in enumerate(summary["status"], start=2):  # a linha 1 é o cabeçalho
        kind = (
            "missing" if status == NO_TRACKS else "short" if status.startswith("faltam") else "ok"
        )
        for column in range(1, len(summary.columns) + 1):
            sheet.cell(row=row, column=column).fill = FILLS[kind]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=CORPUS_DIR / "candidatas_historico.csv")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "candidatas_historico.xlsx")
    args = parser.parse_args()

    corpus = add_status_columns(pd.read_csv(args.csv, encoding="utf-8"))
    summary = summarize_by_year(corpus)

    below_target = summary[summary["status"] != "ok"]
    print(f"[planilha] {len(corpus)} músicas, {corpus['ano'].nunique()} anos com pelo menos uma.")
    print(
        f"[planilha] áudio OK: {corpus['audio_ok'].sum()}/{len(corpus)} | letra OK: "
        f"{corpus['letra_ok'].sum()}/{len(corpus)} | "
        f"ambos OK: {corpus['ambos_ok'].sum()}/{len(corpus)}"
    )
    print(
        f"[planilha] {len(below_target)} ano(s) abaixo da meta de {TARGET_PER_YEAR} prontas "
        f"({(summary['status'] == NO_TRACKS).sum()} sem nenhuma)."
    )

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        corpus.to_excel(writer, sheet_name="corpus", index=False)
        summary.to_excel(writer, sheet_name="resumo_por_ano", index=False)
        format_sheet(writer.sheets["corpus"], corpus)
        format_sheet(writer.sheets["resumo_por_ano"], summary)
        highlight_summary(writer.sheets["resumo_por_ano"], summary)
    print(f"[planilha] gravado em {args.out}")


if __name__ == "__main__":
    main()
