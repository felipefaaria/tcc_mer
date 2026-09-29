#!/usr/bin/env python
"""Verifica se as músicas de uma lista de candidatos (CSV com colunas
``ano,faixa,compositor_interprete,...``) existem no Genius (letra) e no Deezer
(preview de áudio) — só BUSCA, não baixa nada. Serve pra saber, antes de gastar
esforço de extração/curadoria, quantos candidatos de uma fase são realmente
utilizáveis (músicas antigas podem simplesmente não ter preview digital).

Escreve um relatório com uma linha por música: se achou no Genius (e o título
que casou, pra conferência humana) e se achou no Deezer (idem).

Exemplos::

    python scripts/verificar_disponibilidade.py docs/cancao_no_tempo_top5_por_ano.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import requests  # noqa: E402

GENIUS_API = "https://api.genius.com"
DEEZER_API = "https://api.deezer.com/search"


def _primeiro_artista(interprete: str) -> str:
    sem_parenteses = re.sub(r"\([^)]*\)", "", interprete).strip()
    base = sem_parenteses or interprete.strip()
    for sep in (" e ", " E ", ",", "/"):
        if sep in base:
            return base.split(sep)[0].strip()
    return base


def checar_genius(interprete: str, faixa: str, token: str,
                  timeout: float = 15.0) -> tuple[bool, str]:
    headers = {"Authorization": f"Bearer {token}"}
    query = f"{_primeiro_artista(interprete)} {faixa}"
    try:
        r = requests.get(f"{GENIUS_API}/search", params={"q": query},
                         headers=headers, timeout=timeout)
        r.raise_for_status()
        hits = r.json().get("response", {}).get("hits", [])
    except Exception as e:  # noqa: BLE001
        return False, f"erro: {e}"
    if not hits:
        return False, ""
    top = hits[0]["result"]
    return True, f"{top.get('primary_artist', {}).get('name', '?')} - {top.get('title', '?')}"


def checar_deezer(interprete: str, faixa: str,
                  timeout: float = 15.0) -> tuple[bool, str]:
    query = f"{_primeiro_artista(interprete)} {faixa}"
    try:
        r = requests.get(DEEZER_API, params={"q": query}, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:  # noqa: BLE001
        return False, f"erro: {e}"
    if not data or not data[0].get("preview"):
        return False, ""
    item = data[0]
    return True, f"{item.get('artist', {}).get('name', '?')} - {item.get('title', '?')}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv_in", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    out_path = args.out or args.csv_in.with_name(args.csv_in.stem + "_disponibilidade.csv")

    import os
    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        sys.exit("[verificar] defina GENIUS_ACCESS_TOKEN no .env")

    with args.csv_in.open(encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    if args.limit:
        linhas = linhas[:args.limit]

    campos = list(linhas[0].keys()) + ["genius_achou", "genius_match",
                                       "deezer_achou", "deezer_match"]
    resultados = []
    n_genius = n_deezer = n_ambos = 0
    for i, row in enumerate(linhas, start=1):
        interprete, faixa, ano = row["compositor_interprete"], row["faixa"], row["ano"]
        g_ok, g_match = checar_genius(interprete, faixa, token)
        time.sleep(args.sleep)
        d_ok, d_match = checar_deezer(interprete, faixa)
        time.sleep(args.sleep)
        row = dict(row)
        row["genius_achou"] = g_ok
        row["genius_match"] = g_match
        row["deezer_achou"] = d_ok
        row["deezer_match"] = d_match
        resultados.append(row)
        n_genius += g_ok
        n_deezer += d_ok
        n_ambos += g_ok and d_ok
        print(f"[{i}/{len(linhas)}] {ano} {faixa[:35]:35s} "
              f"genius={'OK' if g_ok else '--'} deezer={'OK' if d_ok else '--'}")
        if i % 20 == 0:
            with out_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=campos)
                w.writeheader()
                w.writerows(resultados)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(resultados)

    print(f"\n[verificar] {len(resultados)} verificadas.")
    print(f"  Genius: {n_genius}/{len(resultados)}  Deezer: {n_deezer}/{len(resultados)}  "
          f"Ambos: {n_ambos}/{len(resultados)}")
    print(f"[verificar] relatório em {out_path}")


if __name__ == "__main__":
    main()
