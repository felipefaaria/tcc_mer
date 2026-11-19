# Valência-arousal e dissociação áudio-letra em música brasileira

Código e dados do TCC II (PUC Minas, ICEI). O trabalho estima valência e arousal de 606
faixas da música popular brasileira por dois canais, o áudio e a letra, e mede a
**dissociação** entre eles, `D_V = V_letra - V_áudio` e `D_A = A_letra - A_áudio`. A
diferença entre o que o som expressa e o que a letra diz é tratada como um fenômeno a medir,
e não como erro do modelo.

- **LF1 (áudio):** [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) sobre um trecho de
  30 s.
- **LF2 (letra):** Claude Sonnet 5, zero-shot, sobre a letra completa.
- **Fusão:** média simples dos dois canais, e uma regra por dimensão (valência da letra,
  arousal do áudio).
- **Referência humana:** um gold set de 53 faixas e um experimento às cegas com 18 ouvintes.
- **Sem calibração:** os rótulos são os valores brutos dos modelos, em [-1, 1].

Para repetir os resultados do artigo, siga o [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## A tabela do corpus anotado

[`dataset/corpus_anotado.csv`](dataset/corpus_anotado.csv) tem uma linha por faixa, com os
metadados, o rótulo de cada canal, as duas fusões, a dissociação e um rótulo de emoção
(alegria, tensão, tristeza ou calma) para cada fusão. O dicionário das colunas está em
[`dataset/README.md`](dataset/README.md). Não há áudio nem letra: só metadados e valores
derivados.

## Instalação

```bash
pip install -e ".[dev]"
python -m pytest
```

Requer Python 3.10 ou mais novo. As etapas que precisam de mais dependências têm um extra no
`pyproject.toml`: `llm`, `lexicon`, `collection` e `validation`. A LF1 roda num container
Docker, à parte (ver [`docker/README.md`](docker/README.md)).

## Estrutura

```
tcc_mer/                    biblioteca
  scale.py                    conversão de escala para [-1, 1]
  quadrant.py                 quadrantes de Russell e rótulos de emoção
  fusion.py                   fusão simples e fusão por dimensão
  dissociation.py             D_V, D_A e a magnitude
  metrics.py, agreement.py    CCC, MAE, bootstrap, alfa de Krippendorff
  dataset.py                  monta a tabela do corpus anotado
  lf1_audio.py, lf2_llm.py    funções de rotulagem (áudio e letra)
  lexicon.py                  léxico ANEW-Br (fonte descartada)
  aggregate_annotations.py    gold set a partir das anotações humanas
  selection.py                seleção das 50 faixas do experimento
  validation.py, rater_qc.py  análise do experimento com ouvintes
  sources.py, corpus_files.py coleta (Genius, Deezer, iTunes) e checagem dos arquivos
  exploratory/                fusão calibrada no gold set (abandonada)
scripts/
  gerar_dataset.py            gera a tabela do corpus anotado
  coleta/                     monta o corpus (rede, direitos autorais)
  rotulagem/                  LF2 com o LLM, léxico, testes manuais
  analise/                    gold set e dissociação no corpus
  validacao_humana/           seleção das faixas e análise do experimento
  exploratorio/               a fusão calibrada que ficou fora
docker/                     LF1 (Music2Emo) em container
dataset/                    tabela do corpus anotado e dicionário
data/corpus/                manifestos das faixas (histórico, gold, expansão); áudio e letra ficam fora do git
data/validacao/             mapa de itens e trocas manuais do experimento com ouvintes
prompts/                    prompt versionado da LF2
tests/                      testes com dados sintéticos
```

## Convenções que o código mantém

- **Escala.** Toda saída de modelo e toda anotação humana passa por `tcc_mer.scale.to_unit`
  na borda, nunca inline.
- **Dissociação.** Os canais de áudio e de letra chegam separados; nunca são fundidos antes
  de calcular `D`.
- **Instrumentais.** O que depende da letra fica `NaN`, nunca zero. O manifesto manda: uma faixa
  marcada como instrumental não tem letra mesmo que o arquivo exista.
- **Métrica.** CCC de Lin, por dimensão. Spearman acompanha, para separar ordenação de viés.
- **Quadrante.** O zero conta como positivo. Nas respostas de ouvintes, uma mediana neutra deixa
  a faixa sem quadrante.
- **Priors de amostragem** (`estrato_congruencia_HIP`, `quadrante_prior_HIP`) não são rótulos.

## Licença

- **Código:** [MIT](LICENSE).
- **Tabela do corpus anotado** (`dataset/`): [CC BY 4.0](dataset/LICENSE.md). Ao usá-la, cite o
  trabalho e indique o link da licença.

O áudio e as letras das músicas não fazem parte do repositório nem de nenhuma das duas licenças.
