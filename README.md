# TCC2 — MER em Música Brasileira (Valência-Arousal + Dissociação áudio-letra)

Reconhecimento e **dissociação** de emoções (Valência-Arousal) em música
brasileira, via *weak supervision* com três labeling functions (áudio, léxico,
LLM), fusão tardia calibrada e um gold set humano (SAM). Contexto científico
completo em [`docs/instrucoes/`](docs/instrucoes/).

**Objeto de estudo:** a discrepância entre o que o áudio "sente" e o que a letra
"diz" — `D_V = V_letra − V_audio`, `D_A = A_letra − A_audio` — não é erro a
corrigir, é o fenômeno a medir.

## Estrutura

```
src/                # biblioteca (núcleo científico)
  scale.py          #   to_unit(x, lo, hi) -> [-1, 1]  (única fonte de verdade)
  metrics.py        #   CCC de Lin, MAE/RMSE, quadrante, F1-macro, bootstrap CI
  agreement.py      #   alfa de Krippendorff intervalar + alfa-se-removido
  dissociation.py   #   D_V, D_A a partir dos canais modais separados
  calibration.py    #   AffineCalibration.fit/apply (afim por LF e por dimensão)
  fusion.py         #   pesos no simplex por CCC (SLSQP) + k-fold fora-da-amostra
  lf1_audio.py / lf2_lexicon.py / lf3_llm.py   # as 3 labeling functions
scripts/            # runners no Python normal (gerar_lf2/lf3, gold, fusão)
docker/             # LF1 de áudio (Music2Emo) isolada em container + teste manual
tests/              # suíte pytest (dados sintéticos)
prompts/            # prompts versionados das LFs
docs/               # documentação, relatórios e instruções de orientação
data/               # NÃO versionado (copyright/PII/regenerável):
  corpus/           #   audio/ + lyrics/ (estímulos)
  annotations/      #   anotações humanas + gold derivado
  lexicons/         #   ANEW-Br
  outputs/          #   saídas das LFs + dissociação
```

## Rodando os testes

```bash
pip install -r requirements.txt
python -m pytest
```

## Estado da implementação

Regra do projeto: implementar e testar **tudo que não depende do gold set
fechado**, com dados sintéticos.

- [x] **A1** `scale.py` + testes
- [x] **A2** `metrics.py` + testes (inclui teste que prova CCC ≠ Pearson sob viés)
- [x] **A3** `agreement.py` + testes
- [x] **A4** `dissociation.py` + testes
- [x] **D1** `calibration.py` + testes (sintéticos)
- [x] **D2** `fusion.py` + testes (sintéticos; k-fold fora-da-amostra)
- [ ] **B1** `lf1_audio.py` — Music2Emo/MERT (precisa do modelo + áudio)
- [ ] **B2** `lf2_lexicon.py` — ANEW-Br (precisa do léxico + letras)
- [ ] **B3** `lf3_llm.py` — LLM zero-shot (precisa de API + letras)
- [ ] **C1** `scripts/cut_excerpts.py` — corte de excertos (precisa do áudio fonte)
- [ ] **C2** `aggregate_annotations.py` — parser de JSONs (precisa do schema/dados)
- [ ] **C3** `annotator_qc.py` — camadas de verificação de anotadores

> D1/D2 estão implementados e testados com sintéticos; a **calibração/fusão real**
> só roda quando o gold set estiver suficientemente completo.

## Convenções invioláveis (armadilhas a não reintroduzir)

- **Escala:** uma única função `to_unit`; converter na borda de cada ferramenta,
  nunca ad hoc.
- **Dissociação:** nunca somar/fundir as LFs antes de calcular `D` — os canais
  áudio e letra ficam separados.
- **Métrica:** CCC de Lin, reportado **separado** para V e A (nunca combinar).
- **Krippendorff:** uma chamada por dimensão; alfa menor em Valência é *resultado*.
- **Instrumentais** (`tem_letra=não`): LFs de letra e `D` → `NaN` explícito, nunca zero.
- **Avaliação:** sempre k-fold, CCC **fora-da-amostra**; nunca calibrar e avaliar
  no mesmo fold.
- **Não** usar `estrato_congruencia_HIP` / `quadrante_prior_HIP` como rótulo —
  são priors de amostragem, não verdade.
