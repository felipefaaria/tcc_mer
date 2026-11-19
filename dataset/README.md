# Corpus anotado

Tabela com 606 faixas da música popular brasileira (1901 a 2025) e a estimativa de
valência e arousal de cada uma, por canal (áudio e letra), por duas fusões, mais a
dissociação áudio-letra e um rótulo de emoção. Acompanha o artigo; o artigo explica o
método e o que cada número validou.

* `corpus_anotado.csv`: a tabela (UTF-8, vírgula, ponto decimal, célula vazia = ausente).
* `proveniencia.json`: contagens, hash das entradas e modelo e versão do prompt da letra.

Licença: [CC BY 4.0](LICENSE.md).

**Não há áudio nem letra aqui**, só metadados e valores derivados. Os valores são
estimativas de modelos, não anotações humanas.

## Como foi gerada

```bash
python scripts/gerar_dataset.py
```

A tabela é função de três entradas: os manifestos das faixas (`data/corpus/manifest_historico.csv`,
`manifest_gold.csv` e `manifest_expansao.csv`) e as saídas
brutas das duas funções de rotulagem (`data/outputs/lf1_raw.csv` e `lf2_raw.csv`). Todo
o resto é calculado em `tcc_mer/dataset.py`. Ver [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## Convenções

* **Escala.** Valência e arousal ficam em [-1, 1], com 0 no ponto neutro. As saídas dos
  modelos (1 a 9) são convertidas por `(x - 5) / 4`. **Não há calibração**: os valores
  são os brutos de cada modelo.
* **Quadrantes** do plano de Russell. O zero conta como positivo.

  | Quadrante | Valência | Arousal | Emoção |
  |---|---|---|---|
  | Q1 | >= 0 | >= 0 | alegria |
  | Q2 | < 0 | >= 0 | tensão |
  | Q3 | < 0 | < 0 | tristeza |
  | Q4 | >= 0 | < 0 | calma |

* **Instrumentais** (18) não têm letra: tudo que depende dela fica vazio (nunca zero).
  O áudio continua rotulado (`lf1_*` e `quadrante_lf1`).
* **Fusão** só existe quando os dois canais existem (588 faixas).

## Colunas

### Identificação

| Coluna | Descrição |
|---|---|
| `id` | Identificador da faixa. Históricas: `<ano>-<seq>`; gold e expansão: código do gênero (`BOS01`, `RAP03`). |
| `grupo` | `historica` (516), `gold` (53, as anotadas por humanos) ou `expansao` (37). |
| `ano` | Ano de lançamento. |
| `ano_aproximado` | `True` no gold e na expansão, em que o ano é aproximado. |
| `titulo`, `interprete` | Como constam no manifesto. |
| `genero` | Gênero do manifesto. Nas históricas é o gênero específico e está vazio em 253 faixas; no gold e na expansão é o macrogênero. |
| `instrumental` | `True` se a faixa não tem letra. |

### Rótulos por canal (valores brutos)

| Coluna | Descrição |
|---|---|
| `lf1_audio_valencia`, `lf1_audio_arousal` | LF1: Music2Emo sobre o trecho de 30 s de áudio. |
| `lf2_llm_valencia`, `lf2_llm_arousal` | LF2: LLM (Claude Sonnet 5, zero-shot, prompt v1) sobre a letra completa. |

### Fusões

| Coluna | Descrição |
|---|---|
| `fusao_simples_valencia`, `fusao_simples_arousal` | Média de LF1 e LF2 em cada dimensão. É o rótulo fundido do teste pré-especificado com ouvintes. |
| `fusao_dimensao_valencia`, `fusao_dimensao_arousal` | Valência de LF2 e arousal de LF1. Regra fixa, definida depois do experimento: exploratória. |

### Dissociação (letra menos áudio)

| Coluna | Descrição |
|---|---|
| `dissociacao_valencia`, `dissociacao_arousal` | `D_V = V_letra - V_áudio` e `D_A = A_letra - A_áudio`, em [-2, 2]. |
| `dissociacao_magnitude` | Norma `sqrt(D_V² + D_A²)`. |
| `dissociacao_quadrante` | `True` se `quadrante_lf1` e `quadrante_lf2` diferem. |

### Quadrantes e emoção

| Coluna | Descrição |
|---|---|
| `quadrante_lf1`, `quadrante_lf2` | Quadrante de cada canal. |
| `quadrante_fusao_simples`, `emocao_fusao_simples` | Quadrante e emoção da fusão simples. |
| `quadrante_fusao_dimensao`, `emocao_fusao_dimensao` | Quadrante e emoção da fusão por dimensão. |
| `margem_fusao_simples`, `margem_fusao_dimensao` | `min(abs(valência), abs(arousal))` da fusão: distância ao eixo mais próximo. Margem pequena indica um ponto de fronteira, onde o rótulo muda com pouca variação da estimativa. |

### Experimento

| Coluna | Descrição |
|---|---|
| `no_experimento` | `True` nas 50 faixas avaliadas por 18 ouvintes no experimento de validação. |

## O que esperar dos valores

* A LF1 concorda melhor com ouvintes em arousal do que em valência; a LF2 faz o
  inverso e é fraca em arousal (CCC de 0,23 contra o gold set). Por isso
  `dissociacao_arousal` e o arousal da letra **não estão validados**. O artigo tem os
  números.
* A dissociação é bruta: serve para **ordenar** faixas, não como medida absoluta (a
  média de `D_V` é mais negativa nas estimativas do que nas notas humanas).
* O rótulo de emoção é a leitura de um quadrante, não uma anotação humana. As faixas
  do experimento foram escolhidas por divergência entre áudio e fusão, então os
  resultados com ouvintes descrevem faixas dissociadas, não o corpus.
* O corpus não é uma amostra aleatória: reflete as listas de referência usadas na
  coleta (Seção 3.1 do artigo).
