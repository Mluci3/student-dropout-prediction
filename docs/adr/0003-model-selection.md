# ADR 0003 — Seleção do modelo

- **Status**: Aceito
- **Data**: 2026-10-05

## Contexto

Precisamos de um classificador binário com boa performance, validado por
validação cruzada, e idealmente interpretável (o caso de uso — risco de evasão —
exige explicar *o porquê* de cada previsão).

## Decisão

Tunar três candidatos sob o **mesmo pré-processamento** com `RandomizedSearchCV`
(`StratifiedKFold`, 5 folds, ROC-AUC) e selecionar com a **regra de um erro padrão**:
entre os candidatos cuja ROC-AUC de CV fica a menos de um erro padrão do melhor, fica
o mais simples (`COMPLEXITY_ORDER` em `src/train.py`):

1. `LogisticRegression` (mais simples);
2. `HistGradientBoostingClassifier`;
3. `RandomForestClassifier` (mais complexo).

O mesmo protocolo é aplicado às duas variantes de dados
([ADR 0006](0006-early-warning-primary-model.md)).

**Resultado no modelo principal** (ingresso + 1º semestre), após o tuning:

| Modelo | CV ROC-AUC | Treino | Gap |
|---|---|---|---|
| **Regressão Logística** (escolhida) | 0.893 ± 0.012 | 0.901 | 0.008 |
| Random Forest | 0.894 ± 0.008 | 0.978 | 0.084 |
| HistGradientBoosting | 0.891 ± 0.009 | 0.935 | 0.045 |

A Random Forest tem a maior média, mas a diferença (0.001) é menor que o erro padrão
(0.004): empate estatístico. A Regressão Logística foi selecionada (teste ROC-AUC
0.912). Escolher pelo maior número absoluto teria levado ao modelo mais complexo e
com overfitting, sem ganho real.

## Justificativa

- Com as features engenhadas (contagens e taxas de aprovação), o sinal é largamente
  linear; o modelo mais simples iguala os ensembles.
- A Regressão Logística é **mais rápida e mais leve** (artefato pequeno para o
  deploy) e tem coeficientes diretamente legíveis, que complementam o SHAP.
- Navalha de Occam: sem ganho dos modelos complexos, escolhe-se o simples.
- Os ensembles têm ROC-AUC de treino bem acima da validação cruzada (RF 0.978 e
  HGB 0.935 contra ≈ 0.89): memorizam o treino sem generalizar melhor.

## Consequências

- Explicabilidade reforçada (coeficientes + SHAP). Como as variáveis de desempenho
  são correlacionadas, os coeficientes devem ser lidos em conjunto.
- Pipeline e código já suportam trocar o modelo escolhido automaticamente se dados
  futuros favorecerem um ensemble — a seleção é feita por CV, não fixada.
