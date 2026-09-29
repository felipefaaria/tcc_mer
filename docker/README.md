# LF1 de áudio (Music2Emo) em Docker — guia passo a passo

Esta pasta roda o modelo **Music2Emo** dentro de um **container Docker** para:

1. gerar o `data/outputs/lf1_raw.csv` (valência/arousal do **áudio** de todas as faixas); e
2. **testar uma música por vez** na mão (`testar.ps1`), para conferir a saída.

Você não instala Python 3.10, torch nem MERT no Windows — fica tudo no container.
O resto do projeto só consome o CSV.

> **Já tinha a imagem antiga `lf1-usina`?** A pasta foi renomeada de `usina/` para
> `docker/` e a imagem passou a se chamar `tcc-lf1`. **Só re-etiquetar não basta**
> (`docker tag lf1-usina tcc-lf1`) — o `gerar_lf1.py` que já estava DENTRO da
> imagem antiga ainda procura o caminho velho (`data/stimuli/audio`) e vai falhar
> com "pasta de áudio não existe". Descoberto na prática em 2026-09-27: é preciso
> **reconstruir de verdade** (Passo 2) para a imagem passar a usar `data/corpus/audio`.

---

## Arquivos desta pasta

- `Dockerfile` — receita da imagem (Python 3.10 + Music2Emo + deps).
- `gerar_lf1.py` — roda o modelo em **todas** as faixas → `data/outputs/lf1_raw.csv`.
- `testar_musica.py` — roda o modelo em **uma** música e imprime V/A + moods.
- `testar.ps1` — atalho de uma linha para o `testar_musica.py`.
- `requirements-lf1.txt` — deps do ambiente isolado (referência; o Dockerfile já cuida).

---

## Passo 0 — Docker Desktop (uma vez)

Instale o **Docker Desktop** (<https://www.docker.com/products/docker-desktop/>),
abra e espere a baleia ficar verde. Teste: `docker --version`.

## Passo 1 — Colocar os áudios

Ponha os excertos (~30 s) em **`data/corpus/audio/`**. O **nome do arquivo vira o
id** (ex.: `BOS01.mp3` → `id=BOS01`). Aceita `.mp3, .wav, .flac, .m4a, .ogg`.

## Passo 2 — Construir a imagem (uma vez, demora)

Na **raiz do projeto** (não dentro de `docker/`):

```powershell
docker build -t tcc-lf1 -f docker/Dockerfile .
```

Baixa Python, torch e o Music2Emo — alguns GB na primeira vez. Só repita se mudar
o `Dockerfile`.

## Passo 3 — Testar UMA música (o teste manual)

```powershell
.\docker\testar.ps1 "data\corpus\audio\BOS01.mp3"
```

Funciona com música de qualquer pasta (ex.: um download). Ele imprime valência,
arousal (nas escalas 1–9 e −1..+1), o quadrante e os moods previstos.

## Passo 4 — Smoke test em lote (5 faixas)

```powershell
docker run --rm `
  -e LF1_LIMIT=5 `
  -v "${PWD}/data:/app/data" `
  -v "lf1-hfcache:/root/.cache/huggingface" `
  tcc-lf1
```

- `-v "${PWD}/data:/app/data"` liga a sua pasta `data/` ao container (áudio entra, CSV sai).
- `-v "lf1-hfcache:..."` guarda os pesos do MERT para não rebaixar toda vez.

## Passo 5 — Corpus inteiro

Igual ao smoke test, só **sem** o `-e LF1_LIMIT=5`. Resultado:

```
data/outputs/lf1_raw.csv     <- id, V_raw, A_raw  (é o que o pipeline usa)
data/outputs/lf1_meta.json   <- modelo, data, contagens (reprodutibilidade)
```

---

## Solução de problemas

- **"docker: command not found" / erro de conexão** — Docker Desktop não está aberto.
- **`-v "${PWD}/data:..."` não montou** — use caminho absoluto entre aspas.
- **Rodando em Git Bash (não PowerShell) e a pasta "não existe" dentro do
  container, mesmo com o `-v` certo** — é um bug conhecido do Git Bash/MSYS: ele
  "traduz" automaticamente qualquer argumento que comece com `/` para um caminho
  Windows, inclusive o lado do CONTAINER (`/app/data`), virando algo tipo
  `C:/Program Files/Git/app/data`. Prefixe o comando com `MSYS_NO_PATHCONV=1`:
  ```bash
  MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "lf1-hfcache:/root/.cache/huggingface" \
    tcc-lf1
  ```
  (No PowerShell isso não acontece — os exemplos com `${PWD}` acima já funcionam.)
- **Erro de checkpoint / `saved_models` faltando** — o `Music2emo()` tenta obter os
  pesos; se reclamar, veja o README do repositório
  (<https://github.com/AMAAI-Lab/Music2Emotion>) e me mande a mensagem — há uma
  linha comentada no `Dockerfile` pronta para baixá-lo no build.
- **Lento** — é CPU. Para ~53 faixas são minutos. Aumente a RAM do Docker em
  *Settings → Resources* se travar.
- **Recomeçar do zero** — `docker rmi tcc-lf1` e reconstrua (Passo 2).

## Importante

- **Não versionar áudio** (`data/corpus/audio/` está no `.gitignore` — copyright).
- **Versionar sim** o `lf1_raw.csv` se quiser (features são distribuíveis) — hoje a
  pasta `data/outputs/` está no `.gitignore` como regenerável; ajuste se precisar.
- Esta imagem é **só para a LF1**. O resto do projeto roda no seu Python normal.
