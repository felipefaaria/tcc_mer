# Reprodutibilidade

Como repetir os resultados do artigo. Há três níveis, conforme os dados que você tem.

| Nível | O que precisa | O que reproduz |
|---|---|---|
| 1 | Só este repositório | A tabela do corpus anotado e a dissociação no corpus (Seção 4.2) |
| 2 | Mais as anotações humanas e as respostas do experimento | Gold set, experimento com ouvintes, robustez (Seções 4.1 e 4.3) |
| 3 | Mais o áudio, as letras e as chaves de API | A rotulagem inteira, do zero |

Ambiente em que tudo foi verificado: Python 3.11.9, numpy 2.3.4, scipy 1.16.2, pandas 2.3.3,
krippendorff 0.8.2, openpyxl 3.1.5, demucs 4.0.1, torch 2.9.0.

```bash
pip install -e ".[dev,validation]"
python -m pytest
```

## Nível 1: só o repositório

```bash
python scripts/gerar_dataset.py              # regenera dataset/corpus_anotado.csv
python scripts/analise/analisar_corpus.py    # Seção 4.2, a partir da tabela
```

`gerar_dataset.py` precisa das saídas brutas das LFs (`data/outputs/lf1_raw.csv` e
`lf2_raw.csv`), que não são versionadas. Sem elas, **use a tabela já pronta**:
`analisar_corpus.py` lê só `dataset/corpus_anotado.csv`.

Resultado esperado de `analisar_corpus.py` (588 faixas com os dois canais):

| Quantidade | Valor |
|---|---|
| Quadrantes diferentes entre áudio e letra | 354 (60,2%) |
| Sinal de valência oposto / sinal de arousal oposto | 265 (45,1%) / 232 (39,5%) |
| Faixas em Q2 (tensão), áudio / letra | 6 / 128 |
| Fusão simples muda o quadrante do áudio / o sinal de V | 265 (45,1%) / 209 (35,5%) |
| Fusão por dimensão muda o quadrante do áudio | 258 (43,9%) |
| Média de `D_V` | -0,12 |
| Maiores `\|D_V\|` | *Opa Cadê Eu* (-1,29), *Lorraburra* (-1,26), *De Frente pro Crime* (-1,25) |

## Nível 2: com os dados humanos

Precisa de `data/annotations/Anotadores.xlsx` (gold set) e da planilha de respostas do
experimento (aba `respostas`). O mapa de itens já está em `data/validacao/mapa_itens.csv`.
Nenhum dos dois arquivos de respostas é versionado, porque trazem dados pessoais.

| Seção do artigo | Comando | Resultado esperado |
|---|---|---|
| 4.1, concordância entre avaliadores | `python scripts/analise/gerar_gold.py` | α do gold: melodia 0,12 (V) e 0,65 (A); letra 0,64 e 0,25; completa 0,36 e 0,69. 11 anotadores (3, 3 e 5) |
| 4.1 e 4.2, canais e `D` contra o gold | `python scripts/analise/validar_canais_gold.py` | LF1: CCC 0,47 (V) e 0,65 (A). LF2: 0,71 e 0,23. `D` estimado contra humano: CCC 0,40 (V) e 0,19 (A) |
| 4.3, experimento | `python scripts/validacao_humana/analisar_experimento.py --respostas R.xlsx` | 28 faixas analisáveis; fusão 19, áudio 7, nenhum 2; 73% (IC 52 a 88%), p = 0,029; α ordinal 0,27 (V) e 0,58 (A) |
| 4.3, fusão por dimensão (exploratória) | o mesmo comando | 23 de 28 acertos; 19 × 3 contra o áudio, p = 0,0009 |
| 4.3, qualidade e jackknife | `python scripts/validacao_humana/qc_avaliadores.py --respostas R.xlsx` | Um avaliador sinalizado; sem ele 20 × 7 (p = 0,019); sem ele e o mais rápido 23 × 8 (p = 0,011) |
| 4.3 e Apêndice A, robustez | `python scripts/validacao_humana/robustez.py --respostas R.xlsx` | 63 a 76% a favor da fusão, p < 0,05 em 14 das 18 retiradas; regra pela média 27 × 17 (p = 0,174) |
| 3.7, desvio do protocolo | `python scripts/exploratorio/fusao_calibrada.py --respostas R.xlsx` | Fusão calibrada no gold: 24 de 28; CCC fora da amostra 0,45/0,71 (áudio), 0,66/0,03 (letra), 0,62/0,71 (fusão) |

A seleção das 50 faixas também é reproduzível, a partir das letras e do áudio:

```bash
python scripts/validacao_humana/selecionar_faixas.py --pool
python scripts/validacao_humana/medir_voz.py            # Demucs, cerca de 10 s por faixa na CPU
python scripts/validacao_humana/selecionar_faixas.py --selecionar
```

O resultado coincide com as 50 faixas-alvo de `data/validacao/mapa_itens.csv`, na mesma ordem. O
procedimento automático dá duas faixas de letra em inglês (2008-01 e 2017-02), trocadas à mão por
2000-02 e 2007-02. A troca está em `data/validacao/substituicoes_manuais.csv`, e o script a aplica.

## Nível 3: do zero

1. **Corpus** (`scripts/coleta/`). Letras do Genius e prévias de 30 s do Deezer, do iTunes e, em
   último caso, do YouTube. Precisa de `GENIUS_ACCESS_TOKEN` num `.env`. Os scripts operam sobre o
   estado da coleta, anterior à consolidação dos ids, e **não foram reexecutados** depois dela; o
   produto versionado é `data/corpus/manifest_historico.csv`. Os arquivos de áudio e de letra têm direitos
   autorais e não vêm no repositório.
2. **LF1** (`docker/`). Music2Emo no container, que grava `data/outputs/lf1_raw.csv`.
3. **LF2** (`scripts/rotulagem/gerar_lf2_llm.py`). Claude Sonnet 5, esforço baixo, prompt v1
   (`prompts/lf2_zero_shot.md`), uma chamada por letra. Precisa de `ANTHROPIC_API_KEY`. As 588 letras
   custam cerca de US$ 1,40. A comparação entre modelos está em
   `scripts/rotulagem/comparar_modelos_llm.py`.
4. **Tabela** (`scripts/gerar_dataset.py`).

**O prompt é parte do instrumento.** `prompts/lf2_zero_shot.md` é enviado inteiro ao modelo, com
o cabeçalho (que ainda diz "LF3", o nome antigo da LF2). Não o edite: um teste confere o hash do
arquivo, e qualquer mudança exige incrementar `PROMPT_VERSION`.

**Valores da LF2.** Os valores usados no artigo são os da tabela `dataset/corpus_anotado.csv`, e
`dataset/proveniencia.json` registra o modelo, a versão do prompt e o hash das entradas. Para repetir uma
análise, parta da tabela.

## Discrepâncias conhecidas com o texto do artigo

Encontradas ao preparar este repositório. O texto ainda não foi atualizado.

1. **Instrumentais.** São 18, e não 15. Três instrumentais do gold (INS01 a INS03) tinham um arquivo
   de letra com o texto "(instrumental — sem letra)", e o LLM deu nota a ele. A tabela as trata como
   instrumentais. O número de faixas com os dois canais é **588**, e não 591, e os percentuais do
   corpus mudam na segunda casa (valores acima).
2. **Seleção das faixas.** O texto não menciona as duas trocas manuais de letra em inglês descritas
   acima.

## O que foi verificado

**Reexecutados e conferidos contra os números do artigo:** os testes, `gerar_dataset.py`,
`analisar_corpus.py`, `gerar_gold.py`, `validar_canais_gold.py`, `analisar_experimento.py`,
`qc_avaliadores.py`, `robustez.py`, `fusao_calibrada.py`, `selecionar_faixas.py`, `gerar_lexicon.py` e a
medição de voz (numa faixa).

**Verificados ao vivo, em pequena escala:**

* **LF1 no container.** A imagem foi reconstruída a partir do `Dockerfile` e o ponto de entrada rodou
  em 2 faixas, com o código de dentro da imagem. Junto com o `reprocessar_ids.py`, reproduziu os
  valores de `lf1_raw.csv` com diferença exatamente zero. O atalho `testar.ps1` também funcionou.
* **LF2 com a API.** `gerar_lf2_llm.py` rodou em 3 faixas, e o custo real (US$ 0,0055) bateu com a
  estimativa.
* **Coleta.** `extrair_corpus_historico.py` (3 faixas: letra do Genius e prévia do Deezer),
  `extrair_audio_youtube.py --dry-run` e `verificar_disponibilidade.py` funcionaram. A validação de
  título barrou o caso conhecido de *Favorito*.

**Verificados com uma coleta sintética** (`tests/test_collection_scripts.py`): `consolidar_corpus_historico.py`
(ensaio e `--apply`), `auditar_coleta.py`, `gerar_planilha_candidatas.py` e `corrigir_audio_m4a.py`. Eles operam
sobre o estado da coleta anterior à consolidação, que já não existe no disco real.

**Não repetidos:**

* as rodadas completas da LF1, da LF2 e da coleta sobre as 606 faixas: o produto versionado é a tabela
  `dataset/corpus_anotado.csv`;
* `extrair_corpus_expansao.py` (a expansão por gêneros), cujo csv de candidatas não é versionado.
