#!/usr/bin/env python
"""Varredura completa do corpus historico: para cada musica, confere se o
arquivo de audio e o de letra realmente existem em disco (nao confia so no
texto do manifest) e classifica letra vazia como "instrumental confirmado"
(nao e uma lacuna) sempre que o status da linha diz isso explicitamente.

Gera:
- resumo geral (quantas tem audio, letra, ambos)
- lista de musicas com audio faltando
- lista de musicas com letra faltando (excluindo instrumentais confirmados)
- tabela por ano: total, audio_ok, letra_ok, ambos_ok, status vs meta de 5/ano

Grava tudo em docs/auditoria_corpus.md (utf-8) e adiciona uma aba na planilha.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "corpus" / "audio"
LYRICS_DIR = ROOT / "data" / "corpus" / "lyrics"
META_ALVO = 5


def audio_ok(tid: str) -> bool:
    p = AUDIO_DIR / f"{tid}.mp3"
    return p.exists() and p.stat().st_size > 1000


def letra_texto_ok(tid: str) -> bool:
    p = LYRICS_DIR / f"{tid}.txt"
    return p.exists() and len(p.read_text(encoding="utf-8").strip()) > 0


def main() -> None:
    rows = list(csv.DictReader((ROOT / "docs" / "corpus_historico_final.csv").open(encoding="utf-8")))

    sem_audio, sem_letra, instrumentais = [], [], []
    por_ano: dict[int, dict] = {}

    for r in rows:
        tid, ano, faixa = r["id_temp"], int(r["ano"]), r["faixa"]
        status = r["status"].lower()
        instrumental = "instrumental" in status
        a_ok = audio_ok(tid)
        l_txt_ok = letra_texto_ok(tid)
        l_ok = l_txt_ok or instrumental

        if instrumental and not l_txt_ok:
            instrumentais.append((tid, ano, faixa))
        if not a_ok:
            sem_audio.append((tid, ano, faixa, r["interprete_compositor"]))
        if not l_ok:
            sem_letra.append((tid, ano, faixa, r["interprete_compositor"]))

        d = por_ano.setdefault(ano, {"total": 0, "audio_ok": 0, "letra_ok": 0, "ambos_ok": 0})
        d["total"] += 1
        d["audio_ok"] += int(a_ok)
        d["letra_ok"] += int(l_ok)
        d["ambos_ok"] += int(a_ok and l_ok)

    n = len(rows)
    n_audio = sum(1 for r in rows if audio_ok(r["id_temp"]))
    n_letra = sum(1 for r in rows if letra_texto_ok(r["id_temp"]) or "instrumental" in r["status"].lower())
    n_ambos = sum(1 for r in rows if audio_ok(r["id_temp"]) and
                  (letra_texto_ok(r["id_temp"]) or "instrumental" in r["status"].lower()))

    linhas = []
    linhas.append("# Auditoria do corpus histórico\n")
    linhas.append(f"Total de músicas: **{n}**\n")
    linhas.append(f"- Áudio confirmado: **{n_audio}/{n}** ({100*n_audio/n:.1f}%)")
    linhas.append(f"- Letra confirmada (inclui {len(instrumentais)} instrumentais confirmados, "
                  f"que não são lacuna): **{n_letra}/{n}** ({100*n_letra/n:.1f}%)")
    linhas.append(f"- **Ambos OK (áudio + letra/instrumental)**: **{n_ambos}/{n}** ({100*n_ambos/n:.1f}%)\n")

    linhas.append("## Anos abaixo da meta (5 músicas/ano) — por músicas *prontas* (ambos OK)\n")
    linhas.append("| Ano | Total no corpus | Áudio OK | Letra OK | **Ambos OK** | Status |")
    linhas.append("|---|---|---|---|---|---|")
    ano_min, ano_max = min(por_ano), max(por_ano)
    faltantes_prontos = 0
    for ano in range(ano_min, ano_max + 1):
        d = por_ano.get(ano, {"total": 0, "audio_ok": 0, "letra_ok": 0, "ambos_ok": 0})
        if d["ambos_ok"] >= META_ALVO:
            continue
        faltantes_prontos += 1
        falta = META_ALVO - d["ambos_ok"]
        status = "SEM NENHUMA" if d["total"] == 0 else f"faltam {falta}"
        linhas.append(f"| {ano} | {d['total']} | {d['audio_ok']} | {d['letra_ok']} | {d['ambos_ok']} | {status} |")
    linhas.append(f"\n**{faltantes_prontos} ano(s)** abaixo de {META_ALVO} músicas *prontas* (áudio+letra/instrumental confirmados).\n")

    linhas.append("## Músicas sem áudio confirmado\n")
    for tid, ano, faixa, interprete in sorted(sem_audio, key=lambda x: x[1]):
        linhas.append(f"- {tid} ({ano}) — {faixa} — {interprete}")

    linhas.append("\n## Músicas sem letra confirmada (excluindo instrumentais)\n")
    for tid, ano, faixa, interprete in sorted(sem_letra, key=lambda x: x[1]):
        linhas.append(f"- {tid} ({ano}) — {faixa} — {interprete}")

    out = ROOT / "docs" / "auditoria_corpus.md"
    out.write_text("\n".join(linhas), encoding="utf-8")
    print(f"[auditoria] audio={n_audio}/{n} letra={n_letra}/{n} ambos={n_ambos}/{n}")
    print(f"[auditoria] {faltantes_prontos} ano(s) abaixo de {META_ALVO} musicas prontas")
    print(f"[auditoria] {len(sem_audio)} sem audio, {len(sem_letra)} sem letra (nao-instrumental)")
    print(f"[auditoria] gravado em {out}")


if __name__ == "__main__":
    main()
