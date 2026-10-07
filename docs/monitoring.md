# Plano de monitoramento do modelo

Como acompanhar o modelo depois do deploy, para saber **quando ele deixa de ser
confiável** e precisa ser retreinado.

## O desafio: o desfecho chega tarde

O modelo prevê ao fim do 1º semestre, mas só se sabe se o aluno de fato evadiu
meses — ou anos — depois. Até lá não dá para medir recall ou precisão reais. Por
isso o monitoramento combina:

- **sinais antecipados**, disponíveis assim que um lote novo de alunos chega
  (qualidade dos dados, mudança de perfil e comportamento das previsões);
- **sinais definitivos**, calculados quando os desfechos da coorte forem conhecidos
  (desempenho real, calibração e fairness).

## O que monitorar

| # | Sinal | Métrica | Quando | Limiar de atenção | Ação |
|---|---|---|---|---|---|
| 1 | Qualidade dos dados | colunas ausentes, valores ausentes, notas > 20 (corrompidas), categorias nunca vistas | a cada lote, **antes** de usar as previsões | qualquer coluna ausente; aumento das notas corrompidas | corrigir a extração; colunas ausentes bloqueiam o lote |
| 2 | Drift das entradas | PSI por variável, contra os dados de treino | a cada lote (fim de cada semestre) | PSI > 0,10 atenção; > 0,25 alerta | investigar a causa; alerta em variável importante → avaliar retreino |
| 3 | Drift das previsões | taxa de alerta e probabilidade média | a cada lote | desvio > 10 p.p. da referência (36,8% de alertas; prob. média 0,315) | conferir se a mudança é real (ex.: turma com mais risco) ou problema de dados |
| 4 | Desempenho real | recall no threshold (meta ≥ 0,80), precisão, ROC-AUC | quando os desfechos da coorte forem conhecidos | recall < 0,75 | retreinar e recalibrar |
| 5 | Calibração | Brier score; prob. média vs taxa real de evasão | idem | diferença > 5 p.p. entre prob. média e taxa real | recalibrar (sem necessariamente retreinar) |
| 6 | Fairness | recall por gênero ([ADR 0007](adr/0007-sensitive-attributes.md)) | idem | diferença de recall > 0,10 entre grupos | investigar variáveis *proxy*; ajustar o modelo |
| 7 | Operação | app no ar e sem erros; testes do CI | contínuo | app fora do ar; CI vermelho | logs do Streamlit Cloud; corrigir antes do merge |

Os itens 1–3 são automatizados por `src/monitor.py`. Os itens 4–6 usam as mesmas
funções do treino (`compute_metrics`, `group_metrics` em `src/evaluate.py`) sobre os
alunos cujo desfecho já é conhecido.

## Gatilhos de retreino

Retreinar (`python -m src.train`) quando ocorrer qualquer um destes:

- recall real abaixo de 0,75 numa coorte com desfecho conhecido;
- PSI de alerta (> 0,25) em uma das variáveis mais importantes do modelo — disciplinas
  aprovadas/inscritas no 1º semestre ou mensalidades em dia;
- mudança estrutural: novos cursos, mudança de currículo ou de regras de matrícula;
- no mínimo uma vez por ano, com a coorte mais recente.

Antes de substituir o modelo em produção: os testes precisam passar (CI verde) e o
novo `metrics.json` deve ser comparado com o anterior.

## Como rodar

```bash
python -m src.monitor caminho/do/lote.xlsx --output drift.csv
```

O lote precisa ter as mesmas colunas de `data/raw/StudentsPrepared.xlsx` (a coluna
`Target` é opcional). O relatório mostra a qualidade dos dados, a tabela de PSI por
variável (ordenada da mais crítica para a menos) e o comportamento do modelo no lote.

### Exemplo 1 — alunos do conjunto de teste (sem drift esperado)

- todas as 18 variáveis com status **ok** (PSI ≤ 0,02);
- taxa de alerta 36,8% e probabilidade média 0,315 — iguais à referência;
- qualidade: 41,8% das notas do 1º semestre chegam corrompidas (o problema conhecido,
  resolvido pelo reparo do pipeline) e uma categoria de `QualificacaoAnterior`
  ("10º Ano de Escolaridade") que **não aparece no treino** — o modelo a ignora com
  segurança, e é exatamente o tipo de caso que o monitoramento deve sinalizar.

### Exemplo 2 — drift simulado

Lote com menos disciplinas aprovadas no 1º semestre, mais mensalidades atrasadas e
mais notas corrompidas:

- **alerta** em disciplinas aprovadas no 1º semestre (PSI 2,08) e em mensalidades em
  dia (PSI 0,28); demais variáveis **ok**;
- notas corrompidas sobem de 42% para 54% (sinal de problema na extração), enquanto
  o PSI da nota continua ok, porque o drift é medido depois do reparo.

## O que este projeto não faz (e por quê)

- **Não registra as entradas do app.** O app é público e as entradas seriam dados de
  alunos; além disso, o Streamlit Community Cloud não tem armazenamento persistente.
  Em uso real, os lotes viriam do sistema acadêmico da instituição, com controle de
  acesso, e seriam monitorados com `src/monitor.py`.
- **Não monitora em tempo real.** O modelo é usado uma vez por semestre; monitoramento
  em lote, no mesmo ritmo, é suficiente.
