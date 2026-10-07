# ADR 0002 — Estratégia de reparo das notas

- **Status**: Aceito
- **Data**: 2026-10-05

## Contexto

As colunas `UnidadesCurriculares1SemestreGrau` e `...2SemestreGrau` deveriam estar
na escala portuguesa **[0, 20]**, mas ~40% dos valores estão corrompidos por
**deslocamento do ponto decimal** na exportação:

- `13.4285714285714` → `1.34285714285714e16`
- `13.875` → `13875.0`
- `14.545` → `14545.0`

Os fatores de escala são inconsistentes (×10³ a ×10¹⁵).

## Decisão

Reparar com a regra **"dividir por 10 até o valor cair em [0, 20]"**, implementada
como `GradeRepairTransformer` dentro do pipeline sklearn.

## Alternativas consideradas

1. **Descartar as colunas** — desperdiça sinal: na análise univariada a nota separa
   bem os desfechos (nota média: 7,3 em evasão vs 12,6 em formados).
2. **Imputar pela mediana** nos valores inválidos — perde o sinal real de cada aluno.
3. **Tratar como outliers / winsorizar** — não corrige; mantém valores sem sentido.

## Validação

A regra foi verificada em toda a base: **100% dos valores** caem em [0, 18.88] e
há **zero casos ambíguos** no intervalo (20, 200] (onde `14.5` vs `1.45` seria
indistinguível). Coberto por testes em `tests/test_features.py`.

## Consequências

- Recupera a informação da nota sem perda de dados. No modelo final o peso da nota é
  pequeno, porque a mesma informação já está nas contagens de disciplinas aprovadas
  (ver `docs/analysis.md`, seção 9); o reparo continua necessário para que o modelo
  não receba valores sem sentido.
- Reparo roda identicamente no treino e na inferência (está no pipeline), evitando
  divergência treino/serving.
- Risco residual: se dados futuros trouxerem notas legítimas > 20 (outra escala),
  a regra precisaria ser revista — documentado.
