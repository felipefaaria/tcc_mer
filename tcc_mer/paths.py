"""Caminhos canônicos do repositório.

Os scripts importam daqui em vez de recalcular ``Path(__file__).parent...``. Isso
vale para uma cópia de trabalho com o pacote instalado em modo editável
(``pip install -e .``).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
AUDIO_DIR = CORPUS_DIR / "audio"
LYRICS_DIR = CORPUS_DIR / "lyrics"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
LEXICONS_DIR = DATA_DIR / "lexicons"
OUTPUTS_DIR = DATA_DIR / "outputs"
VALIDATION_DIR = DATA_DIR / "validacao"

DATASET_DIR = ROOT / "dataset"
PROMPTS_DIR = ROOT / "prompts"
