# Usina da LF1 — guia passo a passo (para quem nunca usou Docker)

Esta pasta gera o `data/lf_outputs/lf1_raw.csv` (valência/arousal do **áudio**)
rodando o modelo Music2Emo dentro de um **container Docker**. Você não precisa
instalar Python 3.10, torch, MERT nem nada disso no Windows — fica tudo dentro do
container. O resto do projeto só consome o CSV.

---

## Antes: entendendo o Docker em 3 ideias

Docker deixa você rodar um "computador dentro do computador", já com tudo
instalado do jeito certo. Três palavras que vão aparecer:

- **Imagem** = uma *receita congelada* de um ambiente (Python 3.10 + Music2Emo +
  todas as libs). Você constrói a imagem **uma vez** (`docker build`).
- **Container** = uma *execução* dessa imagem. É o que roda quando você faz
  `docker run`. Termina o trabalho e some (com `--rm`).
- **Volume** = uma *pasta compartilhada* entre o seu Windows e o container. É por
  ela que o áudio **entra** e o CSV **sai**. Sem volume, o que o container gera
  ficaria preso dentro dele.

O fluxo inteiro: **construir a imagem 1x → rodar o container → o CSV aparece na
sua pasta `data/`.**

---

## Passo 0 — Instalar o Docker Desktop (uma vez)

1. Baixe em <https://www.docker.com/products/docker-desktop/> e instale.
   No Windows 11 ele usa o **WSL2** — o instalador cuida disso; se pedir, aceite
   habilitar o WSL2 e reinicie.
2. Abra o **Docker Desktop** e espere o ícone da baleia ficar verde ("Engine
   running"). **Ele precisa estar aberto** sempre que você for usar os comandos.
3. Teste no PowerShell:

   ```powershell
   docker --version
   ```

   Se aparecer uma versão (ex.: `Docker version 27...`), está pronto.

---

## Passo 1 — Colocar os excertos de áudio

Ponha os áudios cortados (~30-45s) em:

```
data/stimuli/audio/
```

O **nome do arquivo vira o id** da faixa no CSV. Ex.: `id001.mp3` → linha com
`id=id001`. Aceita `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`.

> Ainda não tem excertos cortados? Tudo bem — dá para testar a usina com 2-3
> arquivos quaisquer só para ver o pipeline funcionar de ponta a ponta.

---

## Passo 2 — Construir a imagem (uma vez, demora)

No PowerShell, **na raiz do projeto** (a pasta `tcc_mer`, não dentro de `usina/`):

```powershell
docker build -t lf1-usina -f usina/Dockerfile .
```

- `-t lf1-usina` dá o nome "lf1-usina" à imagem.
- `-f usina/Dockerfile` diz qual receita usar.
- O `.` no final é o *contexto* (a raiz do projeto).

Isso baixa Python, torch, o Music2Emo e as libs. **Vai demorar vários minutos e
baixar alguns GB na primeira vez.** É normal. Você só repete este passo se mudar
o `Dockerfile`.

---

## Passo 3 — Smoke test: rodar em poucas faixas primeiro

Antes das 55, rode em **5 faixas** para conferir se os números fazem sentido
(exatamente o que o roteiro pede):

```powershell
docker run --rm `
  -e LF1_LIMIT=5 `
  -v "${PWD}/data:/app/data" `
  -v "lf1-hfcache:/root/.cache/huggingface" `
  lf1-usina
```

Explicando os pedaços (o crase ` no fim das linhas é continuação no PowerShell):

- `--rm` → apaga o container ao terminar (não deixa lixo).
- `-e LF1_LIMIT=5` → processa só as 5 primeiras faixas (smoke test).
- `-v "${PWD}/data:/app/data"` → **liga a sua pasta `data/` ao container**. É por
  aqui que o áudio entra e o CSV sai.
- `-v "lf1-hfcache:/root/.cache/huggingface"` → guarda os pesos do MERT baixados,
  para não rebaixar toda vez.
- `lf1-usina` → o nome da imagem que você construiu.

Na **primeira** execução ele ainda baixa os pesos do modelo. No fim, imprime uma
amostra tipo:

```
[usina] pronto: /app/data/lf_outputs/lf1_raw.csv  (5/5 com V/A válidos)
    id001: V=+0.320  A=+0.550  moods=['happy', 'energetic']
    ...
```

Confira: faixas animadas devem ter V/A mais altos; tristes, mais baixos.

---

## Passo 4 — Rodar no corpus inteiro

Se o smoke test ficou coerente, rode em tudo (só **tira** o `-e LF1_LIMIT=5`):

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "lf1-hfcache:/root/.cache/huggingface" `
  lf1-usina
```

Resultado final na sua máquina:

```
data/lf_outputs/lf1_raw.csv     <- id, V_raw, A_raw  (é o que o pipeline usa)
data/lf_outputs/lf1_meta.json   <- modelo, data, contagens (reprodutibilidade)
```

Pronto. A usina cumpriu o papel. Daqui pra frente o nosso código de análise
(no seu Python normal) lê esse CSV — Docker não entra mais.

---

## Solução de problemas

**"docker: command not found" ou erro de conexão** — o Docker Desktop não está
aberto/rodando. Abra e espere a baleia ficar verde.

**O `-v "${PWD}/data:..."` não montou** — use o caminho completo entre aspas, ex.:
`-v "C:/Users/lipes/OneDrive/Área de Trabalho/TCC_2/tcc_mer/data:/app/data"`.

**Erro sobre checkpoint / `saved_models` faltando** — este é o passo mais provável
de precisar de ajuste. O `Music2emo()` tenta obter os pesos; se ele reclamar de um
arquivo de modelo ausente, veja o README do repositório
(<https://github.com/AMAAI-Lab/Music2Emotion>) para o link do checkpoint e me
mande a mensagem de erro — eu ajusto o `Dockerfile` para baixá-lo no build
(há uma linha comentada pronta para isso).

**Muito lento** — é CPU, sem GPU. Para ~55 faixas são minutos, tudo bem. Se
travar por memória, feche outros programas ou aumente a RAM do Docker em
*Settings → Resources*.

**Quero recomeçar do zero** — `docker rmi lf1-usina` remove a imagem; construa de
novo no Passo 2.

---

## Importante

- **Não versionar o áudio** (`data/stimuli/audio/` está no `.gitignore` — copyright).
- **Versionar sim** o `lf1_raw.csv` (features/rótulos são distribuíveis).
- Esta imagem é **só para a LF1**. O resto do projeto continua leve no seu Python
  normal — não junte os dois ambientes.
```
