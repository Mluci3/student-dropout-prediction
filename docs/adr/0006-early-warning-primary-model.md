# ADR 0006 — Modelo principal: alerta precoce ao fim do 1º semestre

- **Status**: Aceito
- **Data**: 2026-10-05

## Contexto

A base traz o desempenho acadêmico do 1º **e** do 2º semestre. Um modelo com todos os
dados chega a ROC-AUC de teste 0.935, mas parte desse desempenho vem de informação
que, na prática, já descreve a evasão: dos 869 alunos com **0 disciplinas aprovadas no
2º semestre, 84% evadiram** (contra 20% dos demais). Nesse ponto o modelo reconhece
uma evasão que já aconteceu, quando o objetivo do projeto é permitir que a
coordenação **intervenha a tempo**.

## Decisão

Definir o **momento da previsão** como o **fim do 1º semestre** e usar como modelo
principal (servido pelo app) a variante `early_warning`, que só recebe dados de
ingresso + 1º semestre. A variante `full` (com o 2º semestre) é treinada com o mesmo
protocolo e os mesmos alunos de treino/teste, mas serve apenas como **teto de
desempenho** na documentação.

As variantes são definidas em um único lugar (`VARIANT_SEMESTERS` em
`src/config.py`; a principal é escolhida em `config.yaml`). Treino, app e inferência
derivam suas colunas dessa definição, e testes automatizados falham se alguma
informação do 2º semestre chegar ao modelo principal.

## Resultado (conjunto de teste, n = 885)

| | Principal (`early_warning`) | Referência (`full`) |
|---|---|---|
| ROC-AUC (IC 95%) | 0.912 [0.887, 0.934] | 0.935 [0.914, 0.953] |
| Recall | **0.838** | 0.835 |
| Precisão | 0.730 | 0.798 |
| F1 | 0.780 | 0.816 |

*(protocolo final: modelos tunados, probabilidades calibradas e threshold para
recall ≥ 0.80 — ADRs 0003 e 0004.)*

## Alternativas consideradas

1. **Modelo com os dois semestres como principal** — maior ROC-AUC, mas prevê tarde
   demais e parte do ganho vem de informação que já descreve a evasão.
2. **Os dois modelos lado a lado no app** — mais complexo para o usuário e dilui a
   mensagem; a comparação fica na documentação e na página de desempenho do app.
3. **Só dados de ingresso** (previsão na matrícula) — ROC-AUC de teste ≈ 0,82; útil
   como triagem futura, fraco como modelo principal.

## Consequências

- Perda de ~0.02 de ROC-AUC (intervalos de confiança se sobrepõem) em troca de um
  modelo utilizável um semestre antes. Com o mesmo recall-alvo, os dois detectam
  ~84% das evasões; o custo é um pouco mais de falsos alarmes (precisão 0.73 vs 0.80).
- O formulário do app não pede dados do 2º semestre.
- Alunos que evadem durante o 1º semestre continuam fora do alcance do alerta —
  registrado como limitação em `docs/analysis.md`.
