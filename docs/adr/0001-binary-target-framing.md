# ADR 0001 — Enquadramento do alvo binário

- **Status**: Aceito
- **Data**: 2026-10-05

## Contexto

A coluna `Target` tem **três** classes: `Graduado` (2209), `Desistente` (1421) e
`Matriculado` (794). O requisito da prova é um **modelo binário** de previsão de
**evasão**.

## Decisão

Binarizar como **`Desistente` = 1** (positivo) vs **resto = 0** (`Graduado` +
`Matriculado`), usando todas as 4.424 linhas.

## Alternativas consideradas

1. **Remover `Matriculado`** e treinar `Desistente` vs `Graduado` (3.630 linhas).
   Mais limpo academicamente, mas descarta ~18% dos dados e não reflete o objetivo
   de negócio ("quem vai evadir?").
2. **Manter 3 classes** (multiclasse) — viola o requisito de modelo binário.

## Consequências

- Framing fiel ao objetivo de negócio ("prever evasão") e aproveita toda a base.
- `Matriculado` entra como negativo, o que é uma simplificação (aluno ainda em
  curso pode vir a evadir); registrado como limitação em `docs/analysis.md`.
- Leve desbalanceamento (~32% positivos) — tratado em [ADR 0004](0004-imbalance-and-threshold.md).
