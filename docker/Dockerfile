# Ambiente Docker da LF1 (áudio / Music2Emo) — ISOLADO e pesado.
# Usa Python 3.10 (a versão em que o Music2Emo foi desenvolvido), então nada aqui
# depende do Python do seu Windows. Nosso repo de análise nunca importa isto:
# o container só produz o arquivo data/outputs/lf1_raw.csv.

FROM python:3.10-slim

# Libs de sistema: git (uma dependência vem de git+), ffmpeg (decodifica mp3),
# libsndfile1 (librosa), build-essential (compila deps que não tenham wheel).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clona o Music2Emo.
RUN git clone https://github.com/AMAAI-Lab/Music2Emotion.git

# Instala o torch/torchaudio na variante CPU ANTES do resto. Isso evita ~2 GB de
# wheels CUDA (nvidia-*-cu12) que o torch do PyPI padrão puxaria — inúteis num
# laptop CPU-only e onde um download corrompido quebrou o build. Com o torch já
# satisfeito, o requirements.txt abaixo não rebaixará a versão CUDA.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchaudio==2.3.1

# Demais dependências pinadas do Music2Emo (MERT via transformers, librosa, etc.).
RUN pip install --no-cache-dir -r Music2Emotion/requirements.txt

# (Opcional) pré-baixar o checkpoint do modelo aqui, se o repo exigir download
# manual. Veja docker/README.md -> "Solução de problemas: checkpoint". Deixe
# comentado até confirmar o mecanismo:
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('amaai-lab/music2emo')"

# Copia SÓ o nosso código leve (reusa scale.py + lf1_audio.py + lf_common.py).
COPY src/ /app/src/
COPY docker/gerar_lf1.py /app/docker/gerar_lf1.py
COPY docker/testar_musica.py /app/docker/testar_musica.py

# Torna 'src' e 'music2emo' importáveis; define onde ficam os dados montados.
ENV PYTHONPATH=/app:/app/Music2Emotion
ENV DATA_DIR=/app/data

# cwd = raiz do repo do Music2Emo, para que os caminhos internos dele funcionem.
WORKDIR /app/Music2Emotion

ENTRYPOINT ["python", "/app/docker/gerar_lf1.py"]
