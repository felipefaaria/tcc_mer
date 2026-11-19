#!/usr/bin/env python
"""Auditoria do corpus histórico: o que de fato existe em disco.

Para cada música de ``data/corpus/candidatas_historico.csv``, confere se o arquivo de áudio e o
de letra existem (não confia só no manifesto) e trata a letra vazia como "instrumental
confirmado" quando o status da linha diz isso. Grava ``data/corpus/auditoria_coleta.md`` com o
resumo geral, os anos abaixo da meta de cinco faixas prontas e as faixas sem áudio ou sem letra.

Opera sobre os ids da fase de coleta (``H####``), antes da consolidação em ``<ano>-<seq>``.
"""

import csv

from tcc_mer.corpus_files import TARGET_PER_YEAR, has_audio, has_lyrics_text
from tcc_mer.env import use_utf8_console
from tcc_mer.paths import CORPUS_DIR


def audit(rows: list[dict]) -> dict:
    """Situação de cada faixa e totais por ano."""
    missing_audio, missing_lyrics, instrumental_ids = [], [], []
    per_year: dict[int, dict] = {}
    for row in rows:
        track_id, year = row["id_temp"], int(row["ano"])
        instrumental = "instrumental" in row["status"].lower()
        audio_ok = has_audio(track_id)
        lyrics_text = has_lyrics_text(track_id)
        lyrics_ok = lyrics_text or instrumental

        if instrumental and not lyrics_text:
            instrumental_ids.append(track_id)
        entry = (track_id, year, row["faixa"], row["interprete_compositor"])
        if not audio_ok:
            missing_audio.append(entry)
        if not lyrics_ok:
            missing_lyrics.append(entry)

        counts = per_year.setdefault(
            year, {"total": 0, "audio_ok": 0, "letra_ok": 0, "ambos_ok": 0}
        )
        counts["total"] += 1
        counts["audio_ok"] += audio_ok
        counts["letra_ok"] += lyrics_ok
        counts["ambos_ok"] += audio_ok and lyrics_ok
    return {
        "missing_audio": missing_audio,
        "missing_lyrics": missing_lyrics,
        "n_instrumental": len(instrumental_ids),
        "per_year": per_year,
    }


def build_report(n_tracks: int, result: dict) -> tuple[list[str], dict]:
    per_year = result["per_year"]
    n_audio = sum(c["audio_ok"] for c in per_year.values())
    n_lyrics = sum(c["letra_ok"] for c in per_year.values())
    n_both = sum(c["ambos_ok"] for c in per_year.values())

    lines = [
        "# Auditoria do corpus histórico\n",
        f"Total de músicas: **{n_tracks}**\n",
        f"- Áudio confirmado: **{n_audio}/{n_tracks}** ({100 * n_audio / n_tracks:.1f}%)",
        f"- Letra confirmada (inclui {result['n_instrumental']} instrumentais confirmados, que "
        f"não são lacuna): **{n_lyrics}/{n_tracks}** ({100 * n_lyrics / n_tracks:.1f}%)",
        f"- **Ambos OK (áudio + letra/instrumental)**: **{n_both}/{n_tracks}** "
        f"({100 * n_both / n_tracks:.1f}%)\n",
        f"## Anos abaixo da meta ({TARGET_PER_YEAR} músicas/ano), por músicas *prontas*\n",
        "| Ano | Total no corpus | Áudio OK | Letra OK | **Ambos OK** | Status |",
        "|---|---|---|---|---|---|",
    ]
    below_target = 0
    for year in range(min(per_year), max(per_year) + 1):
        counts = per_year.get(year, {"total": 0, "audio_ok": 0, "letra_ok": 0, "ambos_ok": 0})
        if counts["ambos_ok"] >= TARGET_PER_YEAR:
            continue
        below_target += 1
        status = (
            "SEM NENHUMA"
            if counts["total"] == 0
            else f"faltam {TARGET_PER_YEAR - counts['ambos_ok']}"
        )
        lines.append(
            f"| {year} | {counts['total']} | {counts['audio_ok']} | {counts['letra_ok']} "
            f"| {counts['ambos_ok']} | {status} |"
        )
    lines.append(
        f"\n**{below_target} ano(s)** abaixo de {TARGET_PER_YEAR} músicas *prontas* "
        f"(áudio e letra ou instrumental confirmados).\n"
    )

    for title, entries in (
        ("Músicas sem áudio confirmado", result["missing_audio"]),
        ("Músicas sem letra confirmada (excluindo instrumentais)", result["missing_lyrics"]),
    ):
        lines.append(f"\n## {title}\n")
        lines.extend(
            f"- {track_id} ({year}) — {name} — {performer}"
            for track_id, year, name, performer in sorted(entries, key=lambda e: e[1])
        )
    return lines, {
        "audio": n_audio,
        "lyrics": n_lyrics,
        "both": n_both,
        "below_target": below_target,
    }


def main() -> None:
    use_utf8_console()
    with (CORPUS_DIR / "candidatas_historico.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = audit(rows)
    lines, totals = build_report(len(rows), result)

    out = CORPUS_DIR / "auditoria_coleta.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[auditoria] áudio={totals['audio']}/{len(rows)} letra={totals['lyrics']}/{len(rows)} "
        f"ambos={totals['both']}/{len(rows)}"
    )
    print(
        f"[auditoria] {totals['below_target']} ano(s) abaixo de {TARGET_PER_YEAR} músicas prontas"
    )
    print(
        f"[auditoria] {len(result['missing_audio'])} sem áudio, "
        f"{len(result['missing_lyrics'])} sem letra (não instrumental)"
    )
    print(f"[auditoria] gravado em {out}")


if __name__ == "__main__":
    main()
