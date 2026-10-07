# 🎓 Previsão de Evasão Estudantil · Student Dropout Prediction

> **EN — TL;DR:** End-to-end binary ML pipeline that flags university students at
> risk of dropping out **at the end of their 1st semester** — early enough for the
> institution to intervene (4,424 students, UCI *Predict Students' Dropout and
> Academic Success*). A model that also uses 2nd-semester data scores slightly
> higher (ROC-AUC 0.935 vs 0.912) but largely detects dropouts that have already
> happened, so it is kept only as a performance ceiling. Includes a reproducible
> scikit-learn pipeline, tuned + cross-validated model selection (one-standard-error
> rule), calibrated probabilities, a recall-targeted alert threshold, baselines,
> over/underfitting analysis, a fairness audit (sensitive attributes excluded),
> SHAP explainability (global + per-student), leakage-guard tests and a Streamlit
> app on Streamlit Community Cloud. Code and identifiers are in English; the UI/docs are in Portuguese. See
> [Quickstart](#-como-rodar) and [docs/analysis.md](docs/analysis.md).

Projeto da **Prova Substitutiva — Fase 3 (Machine Learning Engineering)**,
estruturado também como peça de **portfólio**.

---

## 📌 Problema

Prever, de forma **binária**, se um estudante vai **evadir** (`Desistente`) ou não
(`Graduado`/`Matriculado`), **ao fim do 1º semestre**, para que a coordenação
acadêmica possa agir enquanto ainda há tempo.

- **Base**: 4.424 estudantes × 28 variáveis (dados de ingresso, desempenho por
  semestre, perfil demográfico e contexto macroeconômico).
- **Alvo**: `Target` original tem 3 classes → binarizado em `Desistente` (1) vs
  resto (0). Taxa de evasão ≈ 32%.

### Por que prever ao fim do 1º semestre?

Com dados do 2º semestre o modelo fica um pouco melhor, mas por um motivo ruim:
dos 869 alunos com **0 disciplinas aprovadas no 2º semestre, 84% já evadiram**.
Nesse ponto, o modelo reconhece uma evasão que já aconteceu, em vez de antecipá-la.
Por isso o modelo em uso só vê **dados de ingresso + 1º semestre**; o modelo com os
dois semestres é mantido apenas como **teto de desempenho**
([ADR 0006](docs/adr/0006-early-warning-primary-model.md)).

## 🧹 Qualidade dos dados — o reparo das notas

As duas colunas de nota (`UnidadesCurriculares{1,2}SemestreGrau`) vinham com
**~40% dos valores corrompidos**: o ponto decimal foi deslocado na exportação
(ex.: a nota `13,43` aparecia como `1.34e16` ou `13875`). A regra de reparo
(`dividir por 10 até o valor cair em [0, 20]`) recupera **100% dos valores** sem
nenhum caso ambíguo. O reparo é um transformer dentro do pipeline
([`GradeRepairTransformer`](src/features.py)), então roda igual no treino e no app.

## 🛠️ Abordagem (pipeline)

1. **Conjuntos de variáveis por momento da previsão** (`src/config.py`): a variante
   `early_warning` (principal) usa ingresso + 1º semestre; a `full` (referência)
   usa também o 2º semestre. Testes automatizados garantem que nenhuma informação
   do 2º semestre chegue ao modelo principal.
2. **Feature engineering** (`src/features.py`): reparo de notas → features
   derivadas (taxas de aprovação e de avaliação) → one-hot (categóricos) +
   `StandardScaler` (numéricos). Tudo em um único `Pipeline`/`ColumnTransformer`.
   Gênero, nacionalidade e estado civil ficam **fora do modelo**
   ([ADR 0007](docs/adr/0007-sensitive-attributes.md)).
3. **Split estratificado** treino/teste (80/20), com os mesmos alunos para as duas
   variantes.
4. **Tuning e seleção por validação cruzada** (`StratifiedKFold`, 5 folds): os três
   candidatos (Regressão Logística · Random Forest · HistGradientBoosting) são
   tunados com `RandomizedSearchCV`; entre os estatisticamente empatados com o
   melhor, fica o mais simples (regra de um erro padrão).
5. **Calibração** das probabilidades (`CalibratedClassifierCV`, sigmoid), para que
   "30%" no app signifique de fato 30%.
6. **Threshold por recall-alvo**: o corte de alerta é o mais alto que detecta ao
   menos **80% das evasões** nas probabilidades *out-of-fold* do treino (sem
   vazamento).
7. **Avaliação** no conjunto de teste com IC 95% (bootstrap), **baselines**,
   **análise de over/underfitting** (três modelos + curva de aprendizado) e
   **auditoria de recall por gênero**.
8. **Explicabilidade SHAP**: resumo global + fatores locais por aluno.
9. **Serialização** do pipeline calibrado do modelo principal (`models/pipeline.joblib`).

## 📊 Resultados

> As métricas são geradas por `src/train.py` e gravadas em
> [`models/metrics.json`](models/metrics.json); as figuras ficam em
> [`reports/figures/`](reports/figures). Veja a discussão completa em
> [`docs/analysis.md`](docs/analysis.md).

Modelo em uso: **Regressão Logística** calibrada, com dados de ingresso + 1º
semestre. Após o tuning, ela empatou com a Random Forest na validação cruzada
(0.893 vs 0.894 de ROC-AUC, diferença menor que o erro padrão) e foi escolhida por
ser a mais simples — a Random Forest, além disso, mostra overfitting (treino 0.978).

| Conjunto de teste (n = 885) | **Principal** — fim do 1º sem. | Referência — fim do 2º sem. |
|---|---|---|
| ROC-AUC (IC 95%) | **0.912** [0.887, 0.934] | 0.935 [0.914, 0.953] |
| PR-AUC | 0.866 | 0.899 |
| Recall (evasão) | **0.838** | 0.835 |
| Precisão (evasão) | 0.730 | 0.798 |
| F1 (evasão) | 0.780 | 0.816 |
| Acurácia | 0.849 | 0.879 |
| Alertas a cada 100 alunos | 37 | 34 |

Com o mesmo recall-alvo, os dois modelos detectam **~84% dos alunos que evadem**; o
custo de avisar um semestre antes é um pouco mais de falsos alarmes (precisão 0.73
vs 0.80).

**Contra baselines** (mesmo teste): uma regra simples — mensalidades atrasadas OU 0
aprovações no 1º semestre — tem F1 0.693 (recall 0.59); um classificador aleatório,
ROC-AUC 0.50. O modelo chega a F1 0.780 com recall 0.838.

**Over/underfitting** (modelo principal): treino 0.901 · CV 0.893 · teste 0.912 →
sem overfitting nem underfitting relevante (confirmado pela curva de aprendizado).
Na mesma comparação, a Random Forest tem gap treino–CV de 0.084.

**Calibração**: probabilidade média prevista 0.315 para uma taxa real de 0.321
(antes da calibração: 0.407).

## 🚀 Como rodar

```bash
# 1. Ambiente
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # runtime (requirements.txt) + pytest/nbconvert

# 2. Treinar (gera models/pipeline.joblib, metrics.json e figuras)
python -m src.train

# 3. Testes
pytest

# 4. App local
streamlit run app/streamlit_app.py
```

## 🌐 Deploy (Streamlit Community Cloud)

O app é publicado no **Streamlit Community Cloud** a partir deste repositório:
arquivo principal `app/streamlit_app.py`, dependências do `requirements.txt` (versões
fixadas) e Python 3.11. O `models/pipeline.joblib` é versionado para que o app
funcione sem re-treinar, e cada push na `main` atualiza o app
([ADR 0005](docs/adr/0005-deployment-streamlit-cloud.md)).

## 🗂️ Estrutura

```
src/        # config (variantes), data, features, train, evaluate, explain, predict
app/        # streamlit_app.py
tests/      # pytest (reparo de notas, binarização, guardas contra vazamento, inferência)
models/     # pipeline.joblib + metrics.json (principal + referência)
reports/    # figuras do modelo principal (ROC, PR, matriz de confusão, SHAP, curva de aprendizado)
docs/       # analysis.md + ADRs (decisões de arquitetura)
```

## 📚 Dados e citação

A base usada é uma versão traduzida para português e preparada (`StudentsPrepared.xlsx`)
do dataset público **Predict Students' Dropout and Academic Success**, licenciado sob
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict
  Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5MC89>
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting Student
  Dropout and Academic Success. *Data*, 7(11), 146.
  <https://doi.org/10.3390/data7110146>

## 🔗 Entregáveis

- **Repositório GitHub**: <https://github.com/Mluci3/student-dropout-prediction>
- **App (Streamlit Community Cloud)**: <https://evasao-estudantil.streamlit.app/>
- **Vídeo (≥5 min)**: _link_

## ⚠️ Limitações

Parte dos alunos evade ainda durante o 1º semestre, antes do momento da previsão; e
`Matriculado` é tratado como não-evasão, embora esses alunos ainda possam evadir.
Detalhes em [`docs/analysis.md`](docs/analysis.md#10-limitações-e-trabalhos-futuros).
