# ADR 0004 — Desbalanceamento de classes e threshold de decisão

- **Status**: Aceito
- **Data**: 2026-10-05

## Contexto

A classe positiva (`Desistente`) representa ~32% da base. Um threshold ingênuo de
0,5 e a ausência de ponderação tendem a favorecer a classe majoritária, enquanto o
custo de negócio está em **não detectar** um aluno em risco (falso negativo).

## Decisão

1. **Ponderação de classes**: `class_weight="balanced"` em todos os candidatos.
2. **Probabilidades calibradas** (`CalibratedClassifierCV`, sigmoid): a ponderação
   infla as probabilidades (média prevista 0.407 para uma taxa real de 0.321); após
   a calibração, 0.315. Assim o número exibido no app é uma probabilidade de verdade.
3. **Threshold por recall-alvo**: o corte é o **mais alto que detecta ao menos 80% das
   evasões** (`train.target_recall`), calculado sobre probabilidades calibradas
   **out-of-fold** (`cross_val_predict`) no conjunto de treino — nunca no teste. Como
   o critério só depende da ordem das probabilidades, ele não muda com a calibração.

**Resultado** (modelo principal, fim do 1º semestre): threshold ≈ **0,29**; no teste,
recall 0,84 e precisão 0,73 (F1 0,78), com 37 alertas a cada 100 alunos.

## Alternativas consideradas

| Critério | Recall (teste) | Precisão | Alertas / 100 alunos |
|---|---|---|---|
| F1 máximo (critério anterior) | 0.78 | 0.77 | 33 |
| **Recall-alvo 0.80 (escolhido)** | **0.84** | **0.73** | **37** |
| Recall-alvo 0.85 | 0.88 | 0.68 | 42 |
| Recall-alvo 0.90 | 0.93 | 0.58 | 51 |

*(valores medidos antes da calibração e da remoção dos atributos sensíveis; a
escolha do critério não depende disso.)*

- **F1 máximo**: trata falso negativo e falso positivo como equivalentes, o que não
  vale para um alerta precoce.
- **Recall-alvo maior**: os alertas se aproximam de metade da turma e a precisão cai
  rápido.
- **Oversampling/SMOTE**: adiciona complexidade; ponderação + threshold resolvem.

## Consequências

- Decisão de corte transparente e reprodutível, persistida em `metrics.json` e
  reutilizada pelo app (`src/predict.get_threshold`).
- O recall-alvo é um parâmetro de negócio explícito (`config.yaml`): pode ser
  ajustado à capacidade de atendimento da coordenação.
