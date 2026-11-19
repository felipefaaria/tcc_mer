#!/usr/bin/env python
"""Extrai letra (Genius) e prévia de 30 s de áudio (Deezer, depois iTunes) da expansão por gêneros.

Lê ``data/corpus/candidatas_expansao.csv`` (intérprete, faixa, gênero ou ritmo, década e
autoria já verificada) e, para cada candidato:

* busca a letra completa no Genius e grava ``data/corpus/lyrics/<id>.txt``. Os ritmos
  tipicamente instrumentais (frevo, vanerão e milonga, guitarrada) não buscam letra: gravam
  um ``.txt`` vazio, a convenção de instrumental do projeto;
* busca a prévia de áudio e grava ``data/corpus/audio/<id>.mp3``;
* grava ``data/corpus/manifest_expansao.csv`` com os metadados verificados e o status de
  cada fonte, de forma incremental.

Linhas sem faixa concreta ("a confirmar", "a pesquisar", "(variado)"), duplicatas e gêneros
sem código são puladas e reportadas, sem receber id. Os ids seguem o padrão de três letras
mais dois dígitos (``MPB10``), continuando a numeração já existente em ``data/corpus/audio``
para o prefixo. É retomável: o que já tem arquivo salvo é mantido.

A chave do Genius vem de ``GENIUS_ACCESS_TOKEN``, do ambiente ou de um ``.env`` na raiz.

    python scripts/coleta/extrair_corpus_expansao.py --dry-run               # só mostra o plano
    python scripts/coleta/extrair_corpus_expansao.py --limit 3               # testa em 3 faixas
    python scripts/coleta/extrair_corpus_expansao.py --limit 3 --only-lyrics
    python scripts/coleta/extrair_corpus_expansao.py                         # roda tudo
"""

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tcc_mer.env import load_env, use_utf8_console
from tcc_mer.paths import AUDIO_DIR, CORPUS_DIR, LYRICS_DIR
from tcc_mer.sources import ensure_lyrics, ensure_preview, is_concrete_track

# Gênero ou ritmo, como está no csv, para o prefixo de 3 letras do id. Os gêneros que já
# existem no corpus reaproveitam o prefixo; os ritmos regionais têm códigos novos, checados
# contra os prefixos em uso (o "FRE" já existe, por isso o frevo usa "FRV").
GENRE_CODES = {
    "MPB": "MPB",
    "Samba": "SAM",
    "Sertanejo": "SER",
    "Forró": "FOR",
    "Funk carioca": "FNK",
    "Rap nacional": "RAP",
    "Rock brasileiro": "ROC",
    "Axé": "AXE",
    "Bossa nova": "BOS",
    "Pagode": "PAG",
    "Frevo": "FRV",
    "Maracatu / manguebeat": "MAN",
    "Coco / embolada": "COC",
    "Repente / cordel": "REP",
    "Música gaúcha tradicionalista": "GAU",
    "Vanerão / milonga campeira": "VAN",
    "Fandango caiçara": "FAN",
    "Carimbó": "CAR",
    "Guitarrada paraense": "GUI",
    "Toada de boi-bumbá": "BOI",
    "Toada / rasqueado pantaneiro": "PAN",
    "Catira": "CAT",
}

INSTRUMENTAL_GENRES = {"Frevo", "Vanerão / milonga campeira", "Guitarrada paraense"}

MANIFEST_FIELDS = [
    "id",
    "eixo",
    "regiao",
    "genero_macro_ou_ritmo",
    "decada",
    "popularidade_tier",
    "interprete",
    "compositor",
    "autoral",
    "faixa",
    "ano_aprox",
    "instrumental",
    "genius_status",
    "deezer_status",
]


@dataclass
class Candidate:
    """Uma linha do csv de candidatos; os campos espelham as colunas."""

    line: int
    eixo: str
    regiao: str
    genero: str
    decada: str
    tier: str
    interprete: str
    compositor: str
    autoral: str
    faixa: str
    ano: str
    id: str = ""
    instrumental: bool = False
    genius_status: str = "pendente"
    deezer_status: str = "pendente"
    warnings: list[str] = field(default_factory=list)


def read_candidates(csv_path: Path) -> tuple[list[Candidate], list[Candidate]]:
    """``(processáveis, pulados)`` a partir do csv de candidatos."""
    processable, skipped = [], []
    with csv_path.open(encoding="utf-8") as file:
        for line, row in enumerate(csv.DictReader(file), start=2):  # a linha 1 é o cabeçalho
            candidate = Candidate(
                line=line,
                eixo=row["eixo"],
                regiao=row.get("regiao", ""),
                genero=row["genero_macro_ou_ritmo"],
                decada=row["decada"],
                tier=row["popularidade_tier"],
                interprete=row["interprete"],
                compositor=row["compositor"],
                autoral=row["autoral"],
                faixa=row["faixa"],
                ano=row["ano_aprox"],
            )
            if (row.get("status") or "").strip().startswith("DUPLICATA"):
                candidate.warnings.append("duplicata de faixa já existente no corpus original")
            elif row.get("autoral") == "coletivo" or not is_concrete_track(candidate.faixa):
                candidate.warnings.append("faixa não concreta o suficiente para buscar")
            elif candidate.genero not in GENRE_CODES:
                candidate.warnings.append(f"gênero/ritmo sem código mapeado: {candidate.genero!r}")
            if candidate.warnings:
                skipped.append(candidate)
            else:
                candidate.instrumental = candidate.genero in INSTRUMENTAL_GENRES
                processable.append(candidate)
    return processable, skipped


def next_number(prefix: str, audio_dir: Path) -> int:
    """Próximo número livre do prefixo, olhando os áudios já existentes."""
    numbers = [
        int(m.group(1))
        for f in audio_dir.glob(f"{prefix}*.mp3")
        if (m := re.match(rf"^{prefix}(\d+)$", f.stem))
    ]
    return max(numbers, default=0) + 1


def assign_ids(candidates: list[Candidate], audio_dir: Path) -> None:
    """Dá a cada candidato um id novo, continuando a numeração existente do prefixo."""
    counters: dict[str, int] = {}
    for candidate in candidates:
        prefix = GENRE_CODES[candidate.genero]
        counters.setdefault(prefix, next_number(prefix, audio_dir))
        candidate.id = f"{prefix}{counters[prefix]:02d}"
        counters[prefix] += 1


def write_manifest(candidates: list[Candidate], manifest_path: Path) -> None:
    """Grava o manifesto, atualizando as linhas dos ids já existentes."""
    rows: dict[str, dict] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            rows = {row["id"]: row for row in csv.DictReader(file)}
    for c in candidates:
        rows[c.id] = {
            "id": c.id,
            "eixo": c.eixo,
            "regiao": c.regiao,
            "genero_macro_ou_ritmo": c.genero,
            "decada": c.decada,
            "popularidade_tier": c.tier,
            "interprete": c.interprete,
            "compositor": c.compositor,
            "autoral": c.autoral,
            "faixa": c.faixa,
            "ano_aprox": c.ano,
            "instrumental": c.instrumental,
            "genius_status": c.genius_status,
            "deezer_status": c.deezer_status,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-csv", type=Path, default=CORPUS_DIR / "candidatas_expansao.csv")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--lyrics-dir", type=Path, default=LYRICS_DIR)
    parser.add_argument("--manifest-out", type=Path, default=CORPUS_DIR / "manifest_expansao.csv")
    parser.add_argument("--limit", type=int, help="processa só as N primeiras faixas (teste)")
    parser.add_argument("--only-lyrics", action="store_true")
    parser.add_argument("--only-audio", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0, help="pausa (s) entre faixas")
    parser.add_argument("--dry-run", action="store_true", help="só mostra o plano, sem rede")
    return parser.parse_args()


def main() -> None:
    use_utf8_console()
    load_env()
    args = parse_args()
    if not args.corpus_csv.exists():
        sys.exit(f"[extrair_corpus] csv de candidatas não encontrado: {args.corpus_csv}")
    candidates, skipped = read_candidates(args.corpus_csv)
    assign_ids(candidates, args.audio_dir)

    print(f"[extrair_corpus] {len(candidates)} processável(is), {len(skipped)} pulada(s)")
    for c in skipped:
        print(f"  PULADA (linha {c.line}): {c.interprete} — {c.faixa!r} [{'; '.join(c.warnings)}]")
    if args.limit:
        candidates = candidates[: args.limit]
        print(f"[extrair_corpus] --limit {args.limit}: processando só essas")

    if args.dry_run:
        for c in candidates:
            kind = "instrumental (sem letra)" if c.instrumental else "com letra"
            print(f"  {c.id}  {c.interprete} — {c.faixa} ({c.ano})  [{kind}]")
        print("[extrair_corpus] --dry-run: nenhuma rede ou arquivo tocado")
        return

    token = None
    if not args.only_audio:
        token = os.environ.get("GENIUS_ACCESS_TOKEN")
        if not token:
            sys.exit("[extrair_corpus] defina GENIUS_ACCESS_TOKEN no .env (ou use --only-audio)")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.lyrics_dir.mkdir(parents=True, exist_ok=True)

    for position, c in enumerate(candidates, start=1):
        print(f"[{position}/{len(candidates)}] {c.id}  {c.interprete} — {c.faixa}")
        lyrics_path = args.lyrics_dir / f"{c.id}.txt"
        if args.only_audio:
            c.genius_status = "pulado (--only-audio)"
        elif c.instrumental:
            lyrics_path.write_text("", encoding="utf-8")
            c.genius_status = "instrumental (sem busca)"
        else:
            c.genius_status = ensure_lyrics(lyrics_path, c.interprete, c.faixa, token)
        print(f"    letra: {c.genius_status}")

        if args.only_lyrics:
            c.deezer_status = "pulado (--only-lyrics)"
        else:
            c.deezer_status = ensure_preview(args.audio_dir / f"{c.id}.mp3", c.interprete, c.faixa)
        print(f"    áudio: {c.deezer_status}")

        write_manifest(candidates[:position], args.manifest_out)
        time.sleep(args.sleep)

    lyrics_ok = sum(
        c.genius_status in ("ok", "instrumental (sem busca)", "já existia") for c in candidates
    )
    audio_ok = sum(c.deezer_status in ("ok_deezer", "ok_itunes", "já existia") for c in candidates)
    print(
        f"\n[extrair_corpus] concluído: {lyrics_ok}/{len(candidates)} letras, "
        f"{audio_ok}/{len(candidates)} áudios. Manifesto em {args.manifest_out}"
    )


if __name__ == "__main__":
    main()
