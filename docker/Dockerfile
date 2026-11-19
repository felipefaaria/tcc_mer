# Ambiente da LF1 (áudio, Music2Emo): isolado e pesado.
# Usa Python 3.10, a versão em que o Music2Emo foi desenvolvido. O repositório de análise
# nunca importa isto: o container só produz data/outputs/lf1_raw.csv.

FROM python:3.10-slim

# git (uma dependência vem de git+), ffmpeg (decodifica mp3), libsndfile1 (librosa) e
# build-essential (compila dependências sem wheel).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/AMAAI-Lab/Music2Emotion.git

# torch e torchaudio na variante CPU ANTES do resto: o torch padrão do PyPI puxaria ~2 GB
# de wheels CUDA inúteis num computador sem GPU. Com o torch já instalado, o
# requirements.txt do Music2Emo não o troca. O --extra-index-url cai no PyPI normal para
# dependências que o índice da PyTorch não tem (sem ele, o pip pode tentar compilar
# typing_extensions do código-fonte).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple/ \
        torch==2.3.1 torchaudio==2.3.1

RUN pip install --no-cache-dir -r Music2Emotion/requirements.txt

# Só o nosso código leve: lf1_audio.py, lf_common.py, scale.py e quadrant.py.
COPY tcc_mer/ /app/tcc_mer/
COPY docker/gerar_lf1.py docker/testar_musica.py docker/reprocessar_ids.py /app/docker/

# Torna 'tcc_mer' e 'music2emo' importáveis e define onde ficam os dados montados.
ENV PYTHONPATH=/app:/app/Music2Emotion
ENV DATA_DIR=/app/data

# O cwd é a raiz do repositório do Music2Emo, para os caminhos internos dele funcionarem.
WORKDIR /app/Music2Emotion

ENTRYPOINT ["python", "/app/docker/gerar_lf1.py"]
