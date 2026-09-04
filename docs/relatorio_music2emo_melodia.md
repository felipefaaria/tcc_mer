# Relatório — Teste Music2Emo na melodia de músicas brasileiras

## Objetivo

Verificar, na prática, se o **Music2Emo** que ouve **apenas o áudio**, sem ler a
letra, consegue extrair a emoção da **melodia** de músicas brasileiras. Cada faixa
foi dada ao modelo individualmente e a saída foi julgada manualmente.

## Como ler os números

O modelo devolve duas métricas, cada uma comparada ao **centro da escala**:

- **Valência (V)** — quão _agradável/positiva_ soa. Centro = 5 (de 1 a 9).
- **Arousal (A)** — quão _agitada/energética_ soa. Mesmo centro.

O **quadrante** é a combinação dos dois lados do centro:

|                            | Arousal alto (A ≥ 0) | Arousal baixo (A < 0) |
| -------------------------- | -------------------- | --------------------- |
| **Valência alta (V ≥ 0)**  | Q1 · feliz/animada   | Q4 · serena/relaxada  |
| **Valência baixa (V < 0)** | Q2 · tensa/agitada   | Q3 · triste/calma     |

## Resumo dos 5 casos

| Faixa | Música                            | V     | A     | Quadrante          | Faz sentido?     |
| ----- | --------------------------------- | ----- | ----- | ------------------ | ---------------- |
| AXE02 | Vai Rolar a Festa (Ivete Sangalo) | +0.67 | +0.78 | Q1 feliz/animada   | ✅ sim           |
| MPB02 | Cálice                            | −0.16 | −0.26 | Q3 triste/calma    | ✅ sim           |
| MET01 | Roots Bloody Roots (Sepultura)    | −0.25 | +0.37 | Q2 tensa/agitada   | ✅ sim           |
| BOS01 | Garota de Ipanema (Tom Jobim)     | +0.13 | −0.12 | Q4 serena/relaxada | ✅ sim           |
| FOR01 | Asa Branca (Luiz Gonzaga)         | +0.54 | +0.53 | Q1 (alto arousal)  | ✅ sim (melodia) |

Os cinco casos cobrem **os quatro quadrantes** do modelo afetivo, mostrando que o
Music2Emo acerta a emoção da melodia em todo o espectro do eufórico ao triste.

---

## Casos

### 1. AXE02 — "Vai Rolar a Festa" (Ivete Sangalo) · axé

- **Contexto:** axé baiano, música explicitamente de festa, muito animada.
- **Resultado:** V = 7.68/9 (**+0.67**) · A = 8.14/9 (**+0.78**) · **Q1 · feliz/animada**.
  Moods: _energetic, fast, fun, groovy, happy, party, positive, upbeat_.
- **Faz sentido?** **Sim, e é o caso mais claro.** Melodia de festa → valência e
  arousal altíssimos. O modelo capturou exatamente o clima festivo do som.

### 2. MPB02 — "Cálice" · MPB

- **Contexto:** canção de 1973 (Chico Buarque e Gilberto Gil), símbolo de resistência
  à ditadura, melodia lenta e sombria, conotação triste.
- **Resultado:** V = 4.34/9 (**−0.16**) · A = 3.95/9 (**−0.26**) · **Q3 · triste/calma**.
  Moods: _ballad, melancholic, sad, slow_.
- **Faz sentido?** **Sim.** É o oposto do AXE02: melodia grave e lenta → valência e
  arousal abaixo do centro. O modelo colocou a faixa no quadrante triste/calmo, com
  os moods _sad / melancholic / slow_. Prova que ele também reconhece o negativo.

### 3. MET01 — "Roots Bloody Roots" (Sepultura) · metal

- **Contexto:** metal pesado e agressivo do **Sepultura — banda brasileira** (de
  Belo Horizonte). Além de acertar o quadrante, é um caso de música brasileira.
- **Resultado:** V = 3.99/9 (**−0.25**) · A = 6.50/9 (**+0.37**) · **Q2 · tensa/agitada**.
  Moods: _action, dark, fast, heavy, powerful_.
- **Faz sentido?** **Sim.** Este é o quadrante mais difícil (valência negativa +
  energia alta). O modelo acertou: melodia pesada → tensa e agitada, com moods
  _dark / heavy / powerful_.

### 4. BOS01 — "Garota de Ipanema" (Tom Jobim e Vinícius de Moraes) · bossa nova

- **Contexto:** bossa nova clássica, melodia suave e tranquila.
- **Resultado:** V = 5.54/9 (**+0.13**) · A = 4.54/9 (**−0.12**) · **Q4 · serena/relaxada**.
  Moods: _ballad, meditative, romantic, soft, dream_.
- **Faz sentido?** **Sim.** Levemente agradável e pouco agitada → sereno/relaxado.
  Os moods _soft / meditative / romantic_ descrevem bem a bossa nova.

### 5. FOR01 — "Asa Branca" (Luiz Gonzaga) · forró — _caso de dissociação_

- **Contexto:** forró veloz, marcado pelo triângulo e pela zabumba (melodia muito
  agitada), **mas a letra é triste** (a seca e o êxodo do sertão).
- **Resultado:** V = 7.15/9 (**+0.54**) · A = 7.10/9 (**+0.53**) · **Q1 (alto arousal)**.
  Moods: _energetic, fast, groovy, upbeat_.
- **Faz sentido?** **Sim — e é o caso mais importante.** O modelo ouviu só o **som**
  e disse "agitado/animado", que é exatamente a _melodia_ (forró veloz). A tristeza
  está na **letra**, que é outro canal. Ou seja: o modelo faz o que deve — extrai a
  emoção da **melodia**, não da letra. É a prova direta de que ele lê o áudio.

---

## Conclusão

Nos cinco casos, o Music2Emo colocou cada música brasileira no quadrante emocional
**correto para a sua melodia**, cobrindo todo o espectro afetivo (festivo, triste,
tenso, sereno). O caso de "Asa Branca" mostra de forma limpa que a leitura é
**melódica** (do áudio), independente da letra. Conclusão: **o Music2Emo é adequado
para o canal de melodia/áudio do trabalho**.

**Observação honesta (limitação):** a **valência tende a puxar para o positivo**.
Músicas muito dançantes com carga negativa na letra (ex.: um pagode veloz que fala
de tema pesado) saem com valência levemente positiva, e faixas de rap socialmente
duras não ficam tão negativas quanto se esperaria. Isso é consistente com o já
sabido: o **arousal** é o eixo forte e confiável do áudio; a **valência** é mais
frágil no som — e, por isso, no trabalho ela vem do **canal da letra**, não do áudio.
