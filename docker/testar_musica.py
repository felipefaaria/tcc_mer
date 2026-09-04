#!/usr/bin/env python
"""Testa o Music2Emo em UMA música — para rodar na mão e conferir a saída.

Uso (dentro do container; normalmente chamado pelo atalho ``docker/testar.ps1``)::

    python /app/docker/testar_musica.py /musica/minha_musica.mp3

Ele carrega o modelo, roda na música e imprime valência, arousal e os moods —
tanto na escala nativa do modelo (1 a 9) quanto na escala interna do projeto
([-1, 1]). É o "teste manual": você dá uma música e olha se o retorno faz sentido.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.lf1_audio import make_music2emo_predictor, M2E_LO, M2E_HI
from src.scale import to_unit


def _quadrante(v_unit: float, a_unit: float) -> str:
    """Rótulo do quadrante afetivo a partir de (V, A) em [-1, 1]."""
    if v_unit >= 0 and a_unit >= 0:
        return "Q1 · V+A+ · feliz/animada (ex.: euforia, alegria)"
    if v_unit < 0 and a_unit >= 0:
        return "Q2 · V-A+ · tensa/agitada (ex.: raiva, medo)"
    if v_unit < 0 and a_unit < 0:
        return "Q3 · V-A- · triste/calma (ex.: melancolia, tédio)"
    return "Q4 · V+A- · serena/relaxada (ex.: paz, contentamento)"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("uso: python testar_musica.py <caminho-da-musica>")
    caminho = sys.argv[1]
    if not Path(caminho).exists():
        sys.exit(f"[erro] arquivo não encontrado no container: {caminho}")

    print(f"\n>>> Música: {Path(caminho).name}")
    print(">>> Carregando o Music2Emo (na 1ª vez baixa os pesos; depois é rápido)...\n")
    predictor = make_music2emo_predictor()

    saida = predictor(caminho)
    v9 = float(saida["valence"])          # escala nativa 1-9
    a9 = float(saida["arousal"])          # escala nativa 1-9
    v = to_unit(v9, M2E_LO, M2E_HI)       # escala do projeto -1..1
    a = to_unit(a9, M2E_LO, M2E_HI)
    moods = saida.get("predicted_moods", []) or []

    print("=" * 56)
    print(f"  VALÊNCIA (agradável?)  {v9:.2f}/9   ->  {v:+.2f}  em [-1, 1]")
    print(f"  AROUSAL  (agitada?)    {a9:.2f}/9   ->  {a:+.2f}  em [-1, 1]")
    print(f"  QUADRANTE:             {_quadrante(v, a)}")
    if moods:
        print(f"  MOODS previstos:       {', '.join(moods)}")
    print("=" * 56)
    print("\nDica de leitura: valência = quão POSITIVA/agradável; arousal = quão")
    print("AGITADA/energética. Ex.: uma música animada deve dar V e A altos;")
    print("uma balada triste, V baixo e A baixo.\n")


if __name__ == "__main__":
    main()
