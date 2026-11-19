#!/usr/bin/env python
"""Testa o Music2Emo em UMA música, para rodar na mão e conferir a saída.

Normalmente chamado pelo atalho ``docker/testar.ps1``; dentro do container::

    python /app/docker/testar_musica.py /musica/minha_musica.mp3

Carrega o modelo, roda na música e imprime valência, arousal e os moods, na escala nativa
do modelo (1 a 9) e na escala do projeto ([-1, 1]).
"""

import sys
from pathlib import Path

from tcc_mer.lf1_audio import M2E_HI, M2E_LO, make_music2emo_predictor
from tcc_mer.quadrant import EMOTION_BY_QUADRANT, quadrant_of
from tcc_mer.scale import to_unit


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("uso: python testar_musica.py <caminho-da-musica>")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"[erro] arquivo não encontrado no container: {path}")

    print(f"\n>>> Música: {path.name}")
    print(">>> Carregando o Music2Emo (na 1ª vez baixa os pesos)...\n")
    output = make_music2emo_predictor()(str(path))

    valence_9, arousal_9 = float(output["valence"]), float(output["arousal"])
    valence = to_unit(valence_9, M2E_LO, M2E_HI)
    arousal = to_unit(arousal_9, M2E_LO, M2E_HI)
    quadrant = quadrant_of(valence, arousal)
    moods = output.get("predicted_moods") or []

    print("=" * 56)
    print(f"  VALÊNCIA (agradável?)  {valence_9:.2f}/9   ->  {valence:+.2f}  em [-1, 1]")
    print(f"  AROUSAL  (agitada?)    {arousal_9:.2f}/9   ->  {arousal:+.2f}  em [-1, 1]")
    print(f"  QUADRANTE:             {quadrant} ({EMOTION_BY_QUADRANT[quadrant]})")
    if moods:
        print(f"  MOODS previstos:       {', '.join(moods)}")
    print("=" * 56)
    print("\nValência é o quão agradável, e arousal o quão agitada. Uma música animada deve")
    print("dar valência e arousal altos; uma balada triste, os dois baixos.\n")


if __name__ == "__main__":
    main()
