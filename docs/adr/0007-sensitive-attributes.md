# ADR 0007 — Atributos sensíveis fora do modelo

- **Status**: Aceito
- **Data**: 2026-10-07

## Contexto

A base traz `Genero`, `Nacionalidade` e `EstadoCivil`. O modelo sinaliza alunos
para uma ação da coordenação; pontuar um aluno pelo que ele *é* (e não pelo que ele
*faz* no curso) é eticamente questionável e pode reforçar desigualdades. Além disso,
o modelo original deixava passar mais mulheres em risco do que homens.

## Decisão

Retirar os três atributos de **todas** as variantes do modelo
(`SENSITIVE_COLUMNS` em `src/config.py`; testes falham se algum deles voltar a ser
feature). `Genero` continua disponível **apenas** para auditar o modelo: o treino
calcula recall, precisão e taxa de alerta por gênero no conjunto de teste
(`fairness` em `models/metrics.json`).

## Evidência

- **Custo de desempenho ≈ zero**: com o mesmo modelo, a ROC-AUC de validação cruzada
  foi de 0.8940 (com os três atributos) para 0.8929 (sem eles) — diferença bem menor
  que a variação entre folds (~0.012).
- **Auditoria por gênero** do modelo final (teste, threshold de recall-alvo 0.80):

| Gênero | n | Taxa real de evasão | Recall | Precisão | Taxa de alerta |
|---|---|---|---|---|---|
| Feminino | 563 | 0.24 | 0.825 | 0.681 | 0.29 |
| Masculino | 322 | 0.46 | 0.850 | 0.781 | 0.50 |

O modelo anterior (com os atributos e threshold de F1-máx) tinha recall de 0.72
para mulheres e 0.83 para homens. Parte dessa melhora vem da troca do critério de
threshold (ADR 0004), não só da remoção dos atributos.

## Alternativas consideradas

1. **Manter e apenas reportar** — modelo usaria diretamente o gênero do aluno.
2. **Remover só gênero** — nacionalidade e estado civil também descrevem quem o
   aluno é, sem ganho de desempenho que justifique.

## Consequências

- O formulário do app não pede esses dados.
- A diferença de precisão entre grupos (0.68 vs 0.78) reflete taxas de evasão reais
  diferentes (0.24 vs 0.46) e deve ser acompanhada se o modelo for usado.
- Outras variáveis podem funcionar como *proxy* desses atributos; a auditoria por
  gênero é a forma de verificar isso.
