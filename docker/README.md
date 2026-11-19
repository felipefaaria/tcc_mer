# LF1 (áudio, Music2Emo) em Docker

Esta pasta roda o **Music2Emo** num container para:

1. gerar `data/outputs/lf1_raw.csv`, a valência e o arousal do **áudio** de todas as faixas;
2. **testar uma música por vez** na mão (`testar.ps1`), para conferir a saída.

O Music2Emo depende de Python 3.10, `torch==2.3.1` e `numpy==1.26.4`, que conflitam com o
ambiente de análise. Por isso ele fica isolado e o resto do projeto só consome o CSV.

## Arquivos

- `Dockerfile`: receita da imagem (Python 3.10, Music2Emo e dependências).
- `gerar_lf1.py`: roda o modelo em **todas** as faixas e grava `lf1_raw.csv`.
- `testar_musica.py`: roda o modelo em **uma** música e imprime V/A e os moods.
- `testar.ps1`: atalho de uma linha para o `testar_musica.py`.
- `reprocessar_ids.py`: refaz só alguns ids que falharam (`NaN`) e atualiza o `lf1_raw.csv`.

## 1. Docker Desktop

Instale o [Docker Desktop](https://www.docker.com/products/docker-desktop/), abra e espere
ficar pronto. Teste com `docker --version`.

## 2. Áudios

Coloque os trechos (cerca de 30 s) em `data/corpus/audio/`. O **nome do arquivo vira o id**
(`BOS01.mp3` dá `id=BOS01`). Aceita `.mp3`, `.wav`, `.flac`, `.m4a` e `.ogg`.

## 3. Construir a imagem

Na **raiz do projeto**, uma vez (baixa alguns GB). Refaça se mudar o `Dockerfile` ou o
código do pacote `tcc_mer/` que roda no lote:

```powershell
docker build -t tcc-lf1 -f docker/Dockerfile .
```

## 4. Testar uma música

```powershell
.\docker\testar.ps1 "data\corpus\audio\BOS01.mp3"
```

Aceita música de qualquer pasta. Imprime valência e arousal (escalas 1 a 9 e -1 a +1), o
quadrante e os moods previstos.

## 5. Lote

Teste rápido, com 5 faixas:

```powershell
docker run --rm `
  -e LF1_LIMIT=5 `
  -v "${PWD}/data:/app/data" `
  -v "lf1-hfcache:/root/.cache/huggingface" `
  tcc-lf1
```

`-v "${PWD}/data:/app/data"` liga a sua pasta `data/` ao container (o áudio entra e o CSV
sai), e `lf1-hfcache` guarda os pesos do MERT para não baixar de novo. O corpus inteiro é o
mesmo comando **sem** `-e LF1_LIMIT=5`. O resultado fica em:

```
data/outputs/lf1_raw.csv     id, V_raw, A_raw   (é o que o resto do projeto usa)
data/outputs/lf1_meta.json   modelo, data e contagens
```

Para refazer só alguns ids:

```powershell
docker run --rm --entrypoint python `
  -v "${PWD}/data:/app/data" -v "lf1-hfcache:/root/.cache/huggingface" `
  tcc-lf1 /app/docker/reprocessar_ids.py ID1 ID2
```

## Solução de problemas

- **"docker: command not found" ou erro de conexão**: o Docker Desktop não está aberto.
- **O `-v` não montou**: use caminho absoluto entre aspas.
- **No Git Bash a pasta "não existe" dentro do container**: o Git Bash converte argumentos
  que começam com `/`, inclusive o lado do container (`/app/data`). Prefixe o comando com
  `MSYS_NO_PATHCONV=1`. No PowerShell isso não acontece.
- **Erro de checkpoint ou `saved_models` ausente**: veja o README do
  [Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion).
- **A imagem sai com arquivos de 0 bytes, ou o container falha ao importar pacotes que
  existem (`cannot import name ...`)**: quase sempre é o **disco cheio**. O Docker precisa de
  cerca de 10 GB livres para construir a imagem e, sem espaço, grava arquivos vazios sem
  avisar. Libere espaço, rode `docker builder prune` e reconstrua. Confira o resultado antes
  de usar:
  ```powershell
  docker run --rm --entrypoint wc tcc-lf1 -c /app/docker/gerar_lf1.py
  ```
  O tamanho precisa ser maior que zero.
- **Lento**: é CPU. Para o corpus inteiro, aumente a RAM do Docker em *Settings → Resources*
  se travar.
- **Recomeçar do zero**: `docker rmi tcc-lf1` e reconstrua (passo 3).

## Observações

- Não versione o áudio: `data/corpus/audio/` está no `.gitignore` (direitos autorais).
- A imagem é **só para a LF1**. O resto do projeto roda no Python normal.
