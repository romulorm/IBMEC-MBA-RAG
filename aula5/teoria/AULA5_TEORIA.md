# Aula 5 — Avaliação e Observabilidade: RAGAS, DeepEval e LangFuse Avançado
## Medindo e Melhorando a Qualidade do Pipeline RAG Jurídico
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Carga teórica:** 1h15 (30% da aula) | **Proporção:** 30% teoria / 70% prática  
**Referência normativa:** ABNT NBR 6023:2018 (referências), NBR 10520:2023 (citações)

---

## §1 — O Problema da "Alucinação Silenciosa" em Sistemas RAG Jurídicos

Nas aulas anteriores implementamos progressivamente pipelines mais sofisticados: Naive RAG, Advanced RAG com reranking, busca híbrida no OpenSearch e Contextual Retrieval. Mas como saber se esses sistemas estão realmente funcionando bem?

Considere o seguinte cenário: um assistente jurídico baseado em RAG responde com confiança que "o prazo para interposição de apelação criminal é de 10 dias, conforme o art. 593 do CPP". A resposta é fluente, bem escrita e parece autoritativa — mas está incorreta para determinadas hipóteses. Pior: o documento recuperado citava outro prazo em contexto diferente, e o modelo "completou" a informação com um dado errado de seu treinamento.

Este fenômeno — respostas incorretas mas plausíveis — é chamado de **alucinação**. Em sistemas RAG, há uma forma especialmente perigosa: a **alucinação silenciosa**, onde o modelo ignora ou contradiz o contexto recuperado sem sinalizar incerteza.

### 1.1 Por que Métricas Tradicionais Não Funcionam

As métricas usadas em NLP clássico são inadequadas para RAG:

| Métrica Tradicional | Problema para RAG |
|---|---|
| **BLEU / ROUGE** | Medem sobreposição de n-gramas — não capturam se a resposta está fundamentada no contexto |
| **Perplexidade** | Mede fluência do LLM, não acurácia factual |
| **Acurácia (classificação)** | Requer rótulos binários — inviável para respostas longas abertas |
| **Avaliação humana** | Precisa, mas cara, lenta e não escalável para monitoração contínua |

O que precisamos é de um framework que avalie simultaneamente **o retriever** (qualidade dos contextos recuperados) e **o gerador** (fidelidade da resposta ao contexto).

### 1.2 A Abordagem LLM-as-Judge

O RAGAS (ES et al., 2023) propõe usar um LLM como árbitro: em vez de comparar com strings de referência, usamos um modelo de linguagem para avaliar se a resposta está fundamentada nos contextos recuperados. Esta abordagem tem alta correlação com avaliação humana e escala automaticamente.

---

## §2 — Componentes de um Sistema RAG Avaliável

Para avaliar um pipeline RAG, precisamos de três ingredientes:

```
┌──────────────────────────────────────────────────────────┐
│          TRIO DE AVALIAÇÃO RAG                           │
│                                                          │
│  1. PERGUNTA (question)                                  │
│     Query do usuário ao sistema                          │
│                                                          │
│  2. CONTEXTOS (contexts)                                 │
│     Lista de chunks recuperados pelo retriever           │
│                                                          │
│  3. RESPOSTA (answer)                                    │
│     Texto gerado pelo LLM com base nos contextos         │
│                                                          │
│  4. GROUND-TRUTH (ground_truth) — opcional              │
│     Resposta de referência para métricas que             │
│     precisam de baseline (Context Recall)                │
└──────────────────────────────────────────────────────────┘
```

O **ground-truth** é o elemento mais trabalhoso de construir, mas é fundamental para as métricas que medem cobertura do retriever. No LAB1 construiremos um dataset com 50 pares QA jurídicos anotados.

---

## §3 — As 4 Métricas RAGAS

### 3.1 Faithfulness (Fidelidade)

**O que mede:** Se cada afirmação da resposta pode ser inferida dos contextos recuperados.

**Como é calculado:**
```
Faithfulness = Nº de afirmações suportadas pelo contexto
               ─────────────────────────────────────────
               Nº total de afirmações na resposta
```

O RAGAS usa um LLM para decompor a resposta em afirmações atômicas e verifica cada uma contra os contextos.

**Exemplo jurídico:**
- Resposta: *"O habeas corpus (art. 5º, LXVIII, CF) pode ser impetrado por qualquer pessoa (A), independente de advogado (B), em favor de quem sofrer violência em sua liberdade de locomoção (C)."*
- Contexto recuperado: menciona A e C, mas não B.
- Faithfulness = 2/3 ≈ **0.67** (abaixo da meta ≥ 0.80)

**Meta do syllabus:** ≥ 0.80

### 3.2 Answer Relevancy (Relevância da Resposta)

**O que mede:** Se a resposta é pertinente à pergunta — penaliza respostas vagas, incompletas ou que desviam do tema.

**Como é calculado:**
```
Answer Relevancy = similaridade coseno entre embedding(pergunta)
                   e embeddings de perguntas geradas a partir da resposta
```

O RAGAS gera N perguntas sintéticas que a resposta responderia e mede sua similaridade com a pergunta original. Alta relevância significa que a resposta responde exatamente ao que foi perguntado.

**Exemplo jurídico:** Query sobre "requisitos do HC" → resposta que discorre extensamente sobre "histórico do HC no Brasil" teria Answer Relevancy baixa mesmo sendo factualmente correta.

**Meta do syllabus:** ≥ 0.75

### 3.3 Context Recall (Cobertura do Contexto)

**O que mede:** Se o ground-truth pode ser derivado dos contextos recuperados. Avalia a **completude do retriever**.

**Como é calculado:**
```
Context Recall = Nº de sentenças do ground-truth atribuíveis ao contexto
                 ────────────────────────────────────────────────────────
                 Nº total de sentenças no ground-truth
```

Requer ground-truth anotado. Um Context Recall baixo indica que o retriever está perdendo documentos importantes — o problema está no chunking, embedding ou estratégia de busca.

**Meta do syllabus:** ≥ 0.70

### 3.4 Context Precision (Precisão do Contexto)

**O que mede:** Se os contextos recuperados são relevantes para o ground-truth. Avalia o **ruído no retriever**.

**Como é calculado:**
```
Context Precision = Média ponderada da precisão em cada posição k
                    onde um documento relevante aparece no top-k
```

Um Context Precision baixo indica que o retriever está trazendo documentos irrelevantes junto com os relevantes, "diluindo" a qualidade do contexto fornecido ao LLM.

**Meta do syllabus:** ≥ 0.70

### 3.5 Diagrama das 4 Métricas

```
       PERGUNTA ─────────────────────────────────────────────┐
          │                                                   │
          ▼                                                   ▼
       RETRIEVER                                         GENERATOR
          │                                                   │
          ▼                                                   ▼
       CONTEXTOS ─────────────────────────────────────── RESPOSTA
          │    │                                         │    │
          │    └─── Context Precision ──── (precisão) ───┘    │
          │                                                   │
          └──── Context Recall ──── (via ground-truth) ───────┤
                                                              │
       GROUND-TRUTH ─── Context Recall                       │
                                                              │
       PERGUNTA ─────────────────────────────────────────────►│
                                         Answer Relevancy ◄──┘
                                         Faithfulness ◄──────┘
```

---

## §4 — Implementação do RAGAS

### 4.1 Estrutura do Dataset de Avaliação

```python
from datasets import Dataset

pares_avaliacao = [
    {
        "question":     "Quais são os requisitos do art. 312 do CPP para prisão preventiva?",
        "answer":       resposta_gerada_pelo_pipeline,  # string
        "contexts":     contextos_recuperados,           # list[str]
        "ground_truth": "A prisão preventiva exige fumus comissi delicti e periculum libertatis...",
    },
    # ... mais pares
]

dataset = Dataset.from_list(pares_avaliacao)
```

### 4.2 Executando a Avaliação

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Usar vLLM local como LLM judge
llm_judge = LangchainLLMWrapper(ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0.0,
))

resultado = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=llm_judge,
)

print(resultado)
# Output: {'faithfulness': 0.82, 'answer_relevancy': 0.78,
#          'context_recall': 0.73, 'context_precision': 0.71}
```

### 4.3 Interpretando os Resultados

| Resultado | Interpretação | Ação Recomendada |
|---|---|---|
| Faithfulness baixo | LLM alucinando além do contexto | Melhorar prompt (role + instrução de fidelidade) |
| Answer Relevancy baixo | Resposta vaga ou desviada | Reformular prompt; verificar context window |
| Context Recall baixo | Retriever perdendo docs relevantes | Revisar chunking; aumentar k; trocar embedding model |
| Context Precision baixo | Retriever trazendo ruído | Usar reranking; ajustar alpha na busca híbrida |

---

## §5 — LangFuse Scores API

### 5.1 O que é a Scores API

A LangFuse Scores API permite associar métricas numéricas (scores) a qualquer trace, span ou geração registrada no LangFuse. Isso transforma o LangFuse de uma ferramenta de rastreamento para um **sistema de monitoração de qualidade em produção**.

```
Execução do pipeline RAG
        │
        ├── LangFuse trace (rastreamento)
        │       ├── span: retrieval
        │       ├── span: reranking
        │       └── span: generation
        │
        └── LangFuse scores (qualidade)
                ├── score: faithfulness = 0.83
                ├── score: answer_relevancy = 0.77
                ├── score: context_recall = 0.71
                └── score: context_precision = 0.74
```

### 5.2 Enviando Scores Automaticamente

```python
from langfuse import Langfuse

langfuse = Langfuse(public_key="...", secret_key="...", host="...")

def avaliar_e_registrar(trace_id: str, dataset: Dataset):
    """Calcula RAGAS e envia scores ao LangFuse automaticamente."""
    resultado = evaluate(dataset=dataset, metrics=[...], llm=llm_judge)

    for metrica, valor in resultado.items():
        langfuse.score(
            trace_id=trace_id,
            name=metrica,
            value=float(valor),
            comment=f"RAGAS v0.2 — avaliação automática"
        )

    langfuse.flush()
    return resultado
```

### 5.3 Monitoração Contínua em Produção

Com a integração RAGAS + LangFuse configurada, é possível configurar alertas automáticos quando as métricas caem abaixo das metas:

```
Faithfulness < 0.80 → 🔴 Alerta: possível degradação de qualidade
Answer Relevancy < 0.75 → 🟡 Atenção: respostas possivelmente irrelevantes
Context Recall < 0.70 → 🟡 Atenção: retriever pode estar perdendo documentos
```

No dashboard LangFuse, é possível criar gráficos de evolução das métricas ao longo do tempo, identificando quando uma atualização do corpus ou do modelo causou degradação.

---

## §6 — DeepEval: Testes Unitários para Pipelines LLM

### 6.1 Motivação

O RAGAS avalia a **qualidade média** do pipeline em um dataset. O DeepEval complementa essa abordagem com **testes unitários** — assertions que devem passar para cada caso individual, integradas ao pytest e ao CI/CD.

A analogia com desenvolvimento de software é direta: RAGAS é como um relatório de cobertura de testes, DeepEval é como um teste unitário que falha e bloqueia o deploy.

### 6.2 Estrutura de um Teste DeepEval

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
)

def test_faithfulness_juridico():
    """Pipeline RAG jurídico não deve alucinar sobre legislação."""
    case = LLMTestCase(
        input="Qual o prazo para apelação criminal?",
        actual_output=pipeline_rag.run("Qual o prazo para apelação criminal?"),
        retrieval_context=retriever.get_contexts("Qual o prazo para apelação criminal?"),
    )
    metrica = FaithfulnessMetric(threshold=0.80, model=llm_judge)
    assert_test(case, [metrica])
```

### 6.3 As 5 Métricas Implementadas no LAB4

| Métrica | Threshold | O que testa |
|---|---|---|
| **FaithfulnessMetric** | 0.80 | Resposta fundamentada nos contextos recuperados |
| **AnswerRelevancyMetric** | 0.75 | Resposta pertinente à pergunta |
| **HallucinationMetric** | 0.20 (máx) | Taxa de afirmações sem suporte no contexto |
| **ToxicityMetric** | 0.10 (máx) | Ausência de conteúdo tóxico nas respostas jurídicas |
| **BiasMetric** | 0.15 (máx) | Ausência de viés (étnico, racial, de gênero) em respostas |

> **Por que ToxicityMetric e BiasMetric?** Em sistemas jurídicos com dados de boletins de ocorrência e laudos, o corpus pode conter linguagem discriminatória histórica. É crucial garantir que o modelo não reproduza esses padrões nas respostas.

### 6.4 Integração com CI/CD

```yaml
# .github/workflows/rag-quality-check.yml
name: RAG Quality Gate
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run DeepEval tests
        run: |
          pip install deepeval
          deepeval test run tests/test_pipeline_juridico.py
```

---

## §7 — Análise de Erros: Diagnóstico de Faithfulness Baixo

### 7.1 Taxonomia de Erros de Faithfulness

Quando o Faithfulness fica abaixo de 0.80, há três causas principais:

**Causa 1 — Alucinação Paramétrica**
O modelo usa conhecimento do pré-treinamento em vez do contexto. Solução: reforçar no prompt a instrução de usar apenas o contexto fornecido.

```
❌ Prompt fraco: "Responda à pergunta."
✅ Prompt robusto: "Responda APENAS com base nos trechos abaixo. Se a informação
   não estiver nos trechos, responda 'Informação não encontrada no corpus.'"
```

**Causa 2 — Contexto Insuficiente**
Os chunks recuperados não contêm a informação necessária. Solução: aumentar k, revisar chunking, aplicar Contextual Retrieval (#T09).

**Causa 3 — Janela de Contexto Saturada**
Muitos chunks tornam o contexto longo demais para o modelo processar. Solução: usar reranking para selecionar os top-3 mais relevantes, não top-10.

### 7.2 Template de Diagnóstico

Para cada query com Faithfulness < 0.70:

```
1. Verificar: a resposta cita algo ausente nos contextos?
   → Se sim: Alucinação Paramétrica → ajustar prompt

2. Verificar: os contextos recuperados contêm a resposta correta?
   → Se não: Contexto Insuficiente → ajustar retriever (k, chunking, embedding)

3. Verificar: quantos tokens têm os contextos concatenados?
   → Se > 3000 tokens: Contexto Saturado → adicionar reranking
```

---

## §8 — Resumo: Cadência de Avaliação Recomendada

| Momento | Avaliação | Ferramenta | Frequência |
|---|---|---|---|
| **Desenvolvimento** | Métricas RAGAS no dataset de teste | RAGAS + Pandas | A cada mudança de pipeline |
| **Antes do deploy** | Testes unitários | DeepEval + pytest | CI/CD |
| **Produção** | Monitoração contínua | LangFuse Scores API | Por requisição (amostragem) |
| **Manutenção** | Análise de erros | LangFuse dashboard + RAGAS | Semanal |

> **Insight para o domínio jurídico:** Um pipeline RAG que passa nos testes RAGAS e DeepEval ainda pode degradar quando o corpus é atualizado (nova legislação, novos acórdãos). O monitoramento via LangFuse Scores API é o único mecanismo que detecta essa degradação silenciosa em produção.

---

## Referências Bibliográficas (ABNT)

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217, 2023. Disponível em: <https://arxiv.org/abs/2309.15217>. Acesso em: abr. 2026.

CONFIDENT AI. **DeepEval — The Open-Source LLM Evaluation Framework**. Documentação oficial, 2024. Disponível em: <https://docs.confident-ai.com>. Acesso em: abr. 2026.

CHEN, B. et al. **RAGAS v0.2: Towards Production-Ready RAG Evaluation**. arXiv:2404.14744, 2024.

SAAD-FALCON, J. et al. **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**. arXiv:2311.09476, 2023.

LANGFUSE. **Scores API Documentation**. Disponível em: <https://langfuse.com/docs/scores/custom>. Acesso em: abr. 2026.

KWON, W. et al. **Efficient Memory Management for LLM Serving with PagedAttention**. *ACM SOSP*, 2023.

LEWIS, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *NeurIPS*, v. 33, 2020.
