# ADR 0005 — Deploy no Streamlit Community Cloud

- **Status**: Aceito (revisado em 2026-10-07; substitui a decisão original de usar
  Hugging Face Spaces)
- **Data**: 2026-10-05

## Contexto

O trabalho exige o deploy do modelo via **Streamlit**, com um link público para
entrega. A estrutura do projeto é modular (pacote `src/`, app em `app/`).

A decisão original era publicar em um Space do Hugging Face. Na tentativa de
criação (2026-10-07), o Hugging Face recusou o SDK Streamlit — hoje só aceita
Gradio, Docker ou estático — e informou que hospedar Gradio/Docker no hardware
gratuito exige assinatura PRO.

## Decisão

Publicar no **Streamlit Community Cloud**, a partir do repositório público no
GitHub:

- arquivo principal: `app/streamlit_app.py`;
- dependências: `requirements.txt` na raiz (versões fixadas com `==`);
- Python: 3.11 (escolhido nas configurações avançadas do deploy).

O artefato treinado (`models/pipeline.joblib`, calibrado) é **versionado no
repositório** para que o app funcione sem re-treinar no boot.

## Alternativas consideradas

- **Hugging Face Spaces (Docker)**: exigiria assinatura paga e um Dockerfile.
- **Hugging Face Space estático com stlite** (Streamlit no navegador): gratuito, mas
  bibliotecas como SHAP podem não funcionar nesse ambiente.
- **Re-treinar no startup do app**: lento e frágil; serializar o pipeline é a
  prática correta (separa treino de serving).

## Consequências

- É a plataforma oficial do Streamlit, exatamente o que o enunciado pede, e gratuita.
- Cada push na branch `main` atualiza o app automaticamente.
- `models/pipeline.joblib` precisa ser commitado (não entra no `.gitignore`), e a
  versão do scikit-learn em `requirements.txt` precisa ser a mesma do treino.
- Preprocessamento idêntico no treino e no serving (um único pipeline serializado).
