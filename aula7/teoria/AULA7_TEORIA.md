# Aula 7 — Query Enhancement
## Multi-Query RAG, RAG-Fusion e Step-Back Prompting
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Norma:** ABNT NBR 6023:2018 (referências) | NBR 10520:2023 (citações)  
**Proporção:** 30% teoria / 70% prática | **Carga:** 5h

---

## Sumário

1. [O Problema Fundamental: Vocabulary Mismatch](#1)
2. [Multi-Query RAG — Perspectivas Múltiplas](#2)
3. [Deduplicação e Threshold de Similaridade](#3)
4. [Step-Back Prompting — Abstração para Recuperação](#4)
5. [Query Decomposition — Perguntas Compostas](#5)
6. [RAG-Fusion — Fusão por Reciprocal Rank Fusion](#6)
7. [Análise de Trade-off: Recall vs Latência vs Custo](#7)
8. [Guia de Decisão por Cenário](#8)

---

## 1. O Problema Fundamental: Vocabulary Mismatch {#1}

### 1.1 Por Que Queries Falham?

Um pipeline RAG, por mais sofisticado que seja em sua infraestrutura de busca vetorial, enfrenta um desafio anterior à própria busca: **a distância semântica entre a linguagem do usuário e a linguagem dos documentos indexados**.

Este fenômeno é conhecido na literatura como *vocabulary mismatch* (ROBERTSON; ZARAGOZA, 2009) e é especialmente crítico no domínio jurídico, onde a mesma situação factual pode ser descrita com terminologias completamente distintas dependendo da fonte:

| Intenção do Usuário | Linguagem do Usuário | Linguagem do Documento |
|---|---|---|
| Encerrar contrato de trabalho | "demitir funcionário" | "rescisão contratual" / "dispensa sem justa causa" |
| Violência doméstica | "bater na esposa" | "lesão corporal no âmbito doméstico" / "violência de gênero" |
| Mandado de prisão | "ordem de prender" | "decreto prisional" / "prisão preventiva decretada" |
| Busca e apreensão | "revistar a casa" | "busca domiciliar" / "diligência de busca" |
| Testemunho falso | "mentir na audiência" | "falso testemunho" / "perjúrio" |
| Roubo a banco | "assalto bancário" | "roubo majorado" / "subtração mediante violência à instituição financeira" |
| Fiança | "pagar para sair da cadeia" | "liberdade provisória mediante fiança" / "arbitramento de fiança" |
| Júri | "tribunal popular" | "Tribunal do Júri" / "Conselho de Sentença" |
| Prazo recursal | "prazo para recorrer" | "prazo preclusivo" / "interposição de recurso" |
| Herança | "bens do morto" | "espólio" / "massa hereditária" / "inventário" |

**Resultado prático:** uma query com embeddings semânticamente próximos ao texto do usuário pode estar a uma distância de coseno considerável dos documentos técnicos, resultando em baixo recall mesmo com um índice perfeito.

> *"O maior inimigo do retrieval não é a falta de documentos relevantes — é a falta de vocabulário compartilhado entre quem pergunta e quem escreveu."*  
> — Adaptado de MA et al. (2023)

### 1.2 A Família de Soluções: Query Enhancement

A solução não é reformular o usuário (impossível em produção), mas **enriquecer a query antes do retrieval**. A Tabela 1 sintetiza as três técnicas desta aula:

**Tabela 1 — Comparação das técnicas de Query Enhancement**

| Técnica | Estratégia | Quando usar | Ganho típico |
|---|---|---|---|
| **Multi-Query RAG** | Gerar N variações semânticas | Ambiguidade lexical; usuários leigos | +15-22pp Recall |
| **Step-Back Prompting** | Abstrair para conceito mais geral | Queries excessivamente específicas | +12-18pp Answer Relevancy |
| **RAG-Fusion** | Sub-queries + RRF de rankings | Queries complexas multi-facetadas | +20-25pp Context Recall |

---

## 2. Multi-Query RAG — Perspectivas Múltiplas {#2}

### 2.1 Conceito e Motivação

O Multi-Query RAG parte de uma premissa simples: **se o usuário formulou a query de uma forma, existem N outras formas igualmente válidas de fazer a mesma pergunta**, cada uma potencialmente levando a documentos distintos no espaço vetorial.

A técnica, formalizada por MA et al. (2023), consiste em:

```
query_original
    ↓ LLM (geração de variações)
[query_1, query_2, query_3, query_4]
    ↓ retrieval paralelo
[docs_1, docs_2, docs_3, docs_4]
    ↓ deduplicação
docs_únicos
    ↓ geração
resposta_final
```

**Por que funciona?** O espaço de embeddings não é uniforme. Diferentes formulações da mesma pergunta criam vetores ligeiramente distintos que, em um corpus de alta densidade, podem acertar *vizinhos* diferentes — todos relevantes, mas nenhum único vetor os captura todos.

### 2.2 Implementação com LangChain MultiQueryRetriever

O LangChain oferece a classe `MultiQueryRetriever` que encapsula este fluxo. O componente central é o **prompt de geração de variações**, que instrui o LLM a criar perspectivas distintas:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# LLM local via vLLM
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.7  # temperatura mais alta gera maior diversidade
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)
```

O prompt padrão do LangChain instrui o LLM a gerar 3 variações. Para o domínio jurídico, **recomenda-se customizar o prompt** para especificar que as variações devem incluir terminologia técnica, jurisprudencial e coloquial:

```python
from langchain.prompts import PromptTemplate

PROMPT_MULTI_QUERY_JURIDICO = PromptTemplate(
    input_variables=["question"],
    template="""Você é um assistente jurídico especializado.
Sua tarefa é gerar 4 versões alternativas da pergunta abaixo para recuperação de documentos jurídicos.
Produza variações que cubram: (1) terminologia técnica jurídica, (2) linguagem coloquial,
(3) perspectiva doutrinária e (4) perspectiva processual.
Retorne APENAS as 4 perguntas, uma por linha, sem numeração.

Pergunta original: {question}

Versões alternativas:"""
)
```

### 2.3 Escolha do N: Quantas Variações Gerar?

**N=1** (sem variações): equivale ao Naive RAG. Recall máximo = recall da query original.

**N=3 a 5**: faixa ótima empírica para a maioria dos domínios. Ganho marginal de recall decresce a partir de N=4 (MA et al., 2023).

**N≥8**: risco de *query drift* — as variações começam a se afastar da intenção original, recuperando documentos tangencialmente relacionados que reduzem a precisão sem aumentar significativamente o recall.

> **Regra prática:** Use N=4 para produção. Aumente para N=6 apenas se as queries do seu domínio apresentarem alta variância lexical e o corpus for de alta densidade.

---

## 3. Deduplicação e Threshold de Similaridade {#3}

### 3.1 Por Que Deduplicar?

Com N=4 e k=5 documentos por query, o Multi-Query recupera até 20 documentos. Sem deduplicação, o prompt de geração pode incluir o mesmo chunk repetido 4 vezes, desperdiçando tokens e confundindo o LLM.

O LangChain `MultiQueryRetriever` implementa deduplicação por **igualdade exata** (mesmo `page_content`). Para casos em que o mesmo conteúdo foi indexado com pequenas diferenças (ex: chunking com overlap), esta abordagem falha.

### 3.2 Deduplicação por Similaridade Semântica

A solução robusta é calcular a similaridade de coseno entre os documentos recuperados e remover aqueles acima de um threshold:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_by_similarity(docs, embeddings_model, threshold=0.85):
    """
    Remove documentos semanticamente duplicados.
    threshold=0.85: documentos com similaridade > 85% são considerados duplicatas.
    """
    if not docs:
        return docs
    
    # Gerar embeddings para todos os docs
    texts = [doc.page_content for doc in docs]
    embeddings = embeddings_model.encode(texts, normalize_embeddings=True)
    
    # Calcular matriz de similaridade
    sim_matrix = cosine_similarity(embeddings)
    
    # Selecionar documentos únicos (greedy)
    selected_indices = [0]
    for i in range(1, len(docs)):
        # Verificar se o doc i é muito similar a algum já selecionado
        max_sim = max(sim_matrix[i][j] for j in selected_indices)
        if max_sim < threshold:
            selected_indices.append(i)
    
    return [docs[i] for i in selected_indices]
```

**Escolha do threshold:**
- `0.95`: muito permissivo — remove apenas duplicatas quase idênticas
- `0.85`: balanceado — remove paráfrases próximas (recomendado)
- `0.75`: agressivo — pode remover documentos genuinamente distintos

---

## 4. Step-Back Prompting — Abstração para Recuperação {#4}

### 4.1 O Problema das Queries Hiper-específicas

Considere a query: *"O réu João da Silva, julgado em 2019 na 3ª Vara Criminal de São Paulo pelo artigo 157 §2º do CP, pode apelar com base em erro na dosimetria?"*

Esta query contém informações de alta especificidade (nome, data, vara, artigo) que provavelmente **não existem no corpus exatamente nesta forma**. O embedding desta query se localiza em uma região densa de especificidades que nenhum documento cobre completamente.

O Step-Back Prompting (ZHENG et al., 2024) propõe uma solução contraintuitiva: **pergunte primeiro pelo princípio geral**, depois use o contexto recuperado para responder a pergunta específica.

### 4.2 O Processo de Abstração

```
Query original (específica)
    ↓ Step-Back Prompt (LLM)
Query abstrata (principiológica)
    ↓ retrieval
Documentos sobre o princípio geral
    ↓ geração com contexto geral + query original
Resposta específica fundamentada em princípios gerais
```

**Tabela 2 — Exemplos de transformação jurídica com Step-Back**

| Query Original | Query Step-Back | Por que melhora |
|---|---|---|
| "João pode apelar por erro na dosimetria?" | "Quais são os fundamentos jurídicos para apelação criminal no Brasil?" | Recupera doutrina sobre recursos criminais |
| "Posso prender preventivamente por tráfico de droga sem flagrante?" | "Quais são os requisitos legais para a prisão preventiva?" | Recupera artigos do CPP sobre prisão preventiva |
| "A Súmula 231 do STJ se aplica ao réu primário?" | "Como as súmulas do STJ são aplicadas na dosimetria da pena?" | Recupera jurisprudência sobre súmulas e dosimetria |
| "Laudo do IML de 03/03/2022 atesta lesão grave?" | "Como os laudos médico-legais são utilizados como prova no processo penal?" | Recupera normas sobre prova pericial |
| "Delegado pode apreender celular sem mandado judicial?" | "Quais são os limites constitucionais da busca e apreensão policial?" | Recupera fundamentos constitucionais sobre inviolabilidade |

### 4.3 Implementação do Prompt de Abstração

```python
STEP_BACK_PROMPT = """Você é um jurista especializado em recuperação de informações jurídicas.
Dada a pergunta abaixo, formule UMA pergunta mais geral que capture os princípios jurídicos
subjacentes. A pergunta geral deve ser respondível por documentos doutrinários ou normativos,
sem depender de detalhes específicos do caso.

Pergunta original: {query}

Pergunta geral (Step-Back):"""

def step_back_query(query: str, llm) -> str:
    """Transforma uma query específica em uma abstração principiológica."""
    response = llm.invoke(STEP_BACK_PROMPT.format(query=query))
    return response.content.strip()
```

### 4.4 Quando Step-Back NÃO Ajuda

O Step-Back pode **prejudicar** o recall nas seguintes situações:

- **Queries já abstratas:** "Quais são os princípios do processo penal?" → Step-Back pode sobre-generalizar
- **Buscas por entidade específica:** "Qual é o teor da Súmula Vinculante 11?" → abstração perde o alvo
- **Corpus técnico denso:** quando os documentos já estão no nível de princípios gerais

> **Regra prática:** Use Step-Back apenas quando a query contém nomes próprios, datas, números de processo, artigos específicos ou outros identificadores únicos que provavelmente não estão no corpus na mesma forma.

---

## 5. Query Decomposition — Perguntas Compostas {#5}

### 5.1 O Problema das Queries Multi-facetadas

Queries complexas como *"Quais são as diferenças entre prisão preventiva e prisão temporária, quando cada uma se aplica e quais são os direitos do preso em cada caso?"* são, na verdade, **três perguntas independentes** combinadas em uma.

Um único retrieval não consegue cobrir adequadamente todos os aspectos. A solução é a **Query Decomposition**: decompor a pergunta composta em sub-perguntas atômicas.

### 5.2 Implementação

```python
DECOMPOSITION_PROMPT = """Decomponha a pergunta abaixo em sub-perguntas independentes e
mais simples. Cada sub-pergunta deve ser autocontida e respondível de forma independente.
Retorne APENAS as sub-perguntas, uma por linha.

Pergunta: {question}

Sub-perguntas:"""

def decompose_query(question: str, llm) -> list[str]:
    response = llm.invoke(DECOMPOSITION_PROMPT.format(question=question))
    sub_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return sub_queries
```

**Exemplo de decomposição jurídica:**

Query: *"Quais são as diferenças entre prisão preventiva e temporária, quando cada uma se aplica e quais são os direitos do preso?"*

Decomposição:
1. *"O que é prisão preventiva e quais são seus requisitos legais?"*
2. *"O que é prisão temporária e quais são seus requisitos legais?"*
3. *"Quais são as diferenças entre prisão preventiva e prisão temporária?"*
4. *"Quais são os direitos do preso provisório?"*

---

## 6. RAG-Fusion — Fusão por Reciprocal Rank Fusion {#6}

### 6.1 Arquitetura do RAG-Fusion

O RAG-Fusion (RACKAUCKAS, 2024) combina a geração de múltiplas sub-queries com um algoritmo de fusão de rankings (RRF) para produzir um ranking unificado e robusto.

```
query_original
    ↓ LLM (geração de sub-queries)
[sub_query_1, sub_query_2, sub_query_3]
    ↓ retrieval paralelo (asyncio)
[ranking_1, ranking_2, ranking_3]
    ↓ RRF (Reciprocal Rank Fusion)
ranking_fundido (top-K documentos)
    ↓ geração
resposta_final
```

A diferença do Multi-Query RAG é a etapa de **fusão de rankings por RRF** em vez de simples deduplicação. Isso permite que documentos consistentemente relevantes em múltiplos rankings recebam pontuação amplificada.

### 6.2 Reciprocal Rank Fusion (RRF)

O RRF foi proposto por CORMACK et al. (2009) como um método de fusão de rankings que é simultaneamente simples, robusto e não requer calibração de pesos.

**Fórmula:**

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Onde:
- `d` = documento
- `R` = conjunto de rankings (um por sub-query)
- `rank_r(d)` = posição do documento `d` no ranking `r` (começa em 1)
- `k` = constante de suavização (tipicamente `k=60`)

**Por que k=60?** A constante `k` evita que documentos na posição 1 dominem completamente o ranking final. Com k=60, a diferença entre rank=1 (score=1/61≈0.0164) e rank=2 (score=1/62≈0.0161) é pequena, dando mais peso à consistência entre rankings do que à posição absoluta.

**Exemplo numérico com 3 rankings:**

| Documento | Rank (Q1) | Rank (Q2) | Rank (Q3) | RRF Score |
|---|---|---|---|---|
| Doc A | 1 | 3 | 2 | 1/61 + 1/63 + 1/62 ≈ 0.0487 |
| Doc B | 2 | 1 | 1 | 1/62 + 1/61 + 1/61 ≈ 0.0490 |
| Doc C | 3 | 2 | 5 | 1/63 + 1/62 + 1/65 ≈ 0.0474 |
| Doc D | 5 | 4 | 3 | 1/65 + 1/64 + 1/63 ≈ 0.0472 |

**Obs:** Doc B supera Doc A no ranking final porque aparece em 1º lugar em 2 rankings. Doc A aparece em 1º lugar em apenas 1 ranking. Documentos **consistentemente relevantes** ganham sobre documentos **excepcionalmente relevantes** em apenas um ranking.

### 6.3 Implementação com asyncio

O paralelismo real é crucial para que o RAG-Fusion não triplique a latência:

```python
import asyncio
from langchain_openai import ChatOpenAI
from opensearchpy import AsyncOpenSearch

async def retrieve_for_query(query: str, opensearch_client, index_name: str, 
                              embeddings_model, k: int = 5) -> list:
    """Executa retrieval para uma única query — função assíncrona."""
    query_embedding = embeddings_model.encode(query, normalize_embeddings=True).tolist()
    
    response = await opensearch_client.search(
        index=index_name,
        body={
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            }
        }
    )
    return response["hits"]["hits"]

async def rag_fusion_retrieve(original_query: str, llm, opensearch_client, 
                               index_name: str, embeddings_model, n: int = 3) -> list:
    """Pipeline completo RAG-Fusion com paralelismo real."""
    
    # 1. Gerar sub-queries
    sub_queries = await generate_sub_queries_async(original_query, llm, n)
    sub_queries.append(original_query)  # incluir a query original
    
    # 2. Retrieval paralelo (asyncio.gather = paralelismo real)
    retrieval_tasks = [
        retrieve_for_query(q, opensearch_client, index_name, embeddings_model)
        for q in sub_queries
    ]
    all_rankings = await asyncio.gather(*retrieval_tasks)
    
    # 3. RRF
    fused_docs = reciprocal_rank_fusion(all_rankings, k=60)
    
    return fused_docs

def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list:
    """
    Implementação do RRF.
    rankings: lista de rankings, cada ranking é uma lista de hits do OpenSearch.
    Retorna lista de dicts {doc_id, score, content} ordenada por score RRF.
    """
    doc_scores = {}
    doc_content = {}
    
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            doc_id = hit["_id"]
            doc_content[doc_id] = hit["_source"].get("content", "")
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            
            # Fórmula RRF: 1 / (k + rank)
            doc_scores[doc_id] += 1.0 / (k + rank)
    
    # Ordenar por score RRF (decrescente)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {"doc_id": doc_id, "rrf_score": score, "content": doc_content[doc_id]}
        for doc_id, score in sorted_docs
    ]
```

---

## 7. Análise de Trade-off: Recall vs Latência vs Custo {#7}

### 7.1 O Impacto de N no Pipeline

Ao aumentar N (número de queries/sub-queries), três grandezas são afetadas:

**Recall:** cresce com N, mas com **retorno marginal decrescente**. Entre N=1 e N=3, o ganho é substancial (~15-20pp). Entre N=3 e N=5, o ganho é modesto (~5-8pp). Acima de N=5, o ganho é desprezível (< 2pp).

**Latência:**
- *Com asyncio:* a latência é dominada pela query mais lenta. Em teoria, N=3 e N=1 têm latência similar se executados em paralelo. Na prática, a sobrecarga de coordenação das N chamadas de geração de sub-queries adiciona ~200-500ms.
- *Sem asyncio:* latência cresce linearmente com N.

**Custo:**
- Cada sub-query consome tokens na geração (LLM) e no retrieval (embeddings)
- Se estiver usando LLM via API (não local), N queries = N× tokens de geração + N× tokens de resposta

### 7.2 Análise Econômica (modelo de custo por 1.000 queries)

Para um sistema em produção com API externa (exemplo: GPT-4o-mini a $0.15/1M tokens input, $0.60/1M tokens output):

| Abordagem | Tokens por query | Custo/1.000 queries | Recall típico |
|---|---|---|---|
| Naive RAG (N=1) | ~1.500 tokens total | ~$0.12 | Baseline |
| Multi-Query N=3 | ~4.500 tokens total | ~$0.36 | +18pp |
| Multi-Query N=5 | ~7.500 tokens total | ~$0.60 | +22pp |
| RAG-Fusion N=3 | ~5.000 tokens total | ~$0.40 | +22pp |

> **Para vLLM local (Llama 3.1 8B):** o custo é apenas computacional (GPU). Com uma GPU A100, o custo por query é essencialmente o custo de energia elétrica, tornado o uso de N=5 muito mais viável do que com APIs externas.

### 7.3 Recomendações por Cenário

| Cenário | Abordagem Recomendada | Justificativa |
|---|---|---|
| Chatbot público (alto volume, baixo custo) | Naive RAG ou Multi-Query N=3 | Custo precisa ser controlado |
| Assistente jurídico interno (baixo volume) | RAG-Fusion N=3 | Qualidade > custo |
| Investigação policial (queries complexas) | RAG-Fusion N=5 + Step-Back | Recall máximo necessário |
| Busca legislativa simples (usuário técnico) | Naive RAG | Vocabulário já é preciso |
| Pesquisa doutrinária (usuário leigo) | Multi-Query N=4 | Cobertura de terminologia |

---

## 8. Guia de Decisão por Cenário {#8}

```
O usuário usa linguagem técnica jurídica?
├─ SIM → Naive RAG ou Hybrid Search (Aula 4) pode ser suficiente
└─ NÃO → Aplicar Query Enhancement
         │
         A query é excessivamente específica (nome, data, artigo)?
         ├─ SIM → Step-Back Prompting (+ Multi-Query opcional)
         └─ NÃO → A query tem múltiplas facetas?
                  ├─ SIM → Query Decomposition + RAG-Fusion
                  └─ NÃO → Multi-Query RAG (N=3 a 4)
```

### 8.1 Perguntas de Fixação

1. Explique, com um exemplo do domínio jurídico, por que o problema de vocabulary mismatch não pode ser resolvido apenas melhorando o modelo de embedding.

2. Um assistente de IA deve responder à pergunta: *"Quais foram os fundamentos usados pelo STF no julgamento da ADI 1234?"* Qual técnica de Query Enhancement você aplicaria e por quê?

3. Considerando um sistema com orçamento limitado de GPU, como você determinaria o valor ótimo de N para um pipeline de Multi-Query RAG em produção? Descreva o experimento.

4. Calcule o RRF score para um documento que aparece na posição 2 em Q1, posição 5 em Q2, e não aparece nos top-10 de Q3. Use k=60 e considere apenas os documentos nos top-10. Compare com um documento que aparece em 4º lugar em todos os três rankings.

5. Quais seriam as limitações de usar RAG-Fusion com N=10 em um sistema judicial de alta disponibilidade? Proponha uma solução de fallback para mitigar a latência.

---

## Referências Bibliográficas

CORMACK, G. V.; CLARKE, C. L. A.; BUETTCHER, S. **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods**. In: *Proceedings of the 32nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*, 2009. p. 758–759.

LANGCHAIN. **MultiQueryRetriever**. LangChain Documentation, 2024. Disponível em: <https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever>. Acesso em: abr. 2026.

MA, X. et al. **Query Rewriting for Retrieval-Augmented Large Language Models**. In: *International Conference on Learning Representations (ICLR)*, Vienna, 2023. arXiv:2305.14283. Disponível em: <https://arxiv.org/abs/2305.14283>. Acesso em: abr. 2026.

OPENSEARCH PROJECT. **Hybrid Search Documentation**. Disponível em: <https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/>. Acesso em: abr. 2026.

RACKAUCKAS, Z. **RAG-Fusion: a New Take on Retrieval-Augmented Generation**. arXiv:2402.03367, 2024. Disponível em: <https://arxiv.org/abs/2402.03367>. Acesso em: abr. 2026.

ROBERTSON, S.; ZARAGOZA, H. **The Probabilistic Relevance Framework: BM25 and Beyond**. *Foundations and Trends in Information Retrieval*, v. 3, n. 4, p. 333-389, 2009.

TANG, Y.; YANG, Y. **MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries**. arXiv:2401.15391, 2024. Disponível em: <https://arxiv.org/abs/2401.15391>. Acesso em: abr. 2026.

ZHENG, H. S. et al. **Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**. In: *International Conference on Learning Representations (ICLR)*, Vienna, 2024. arXiv:2310.06117. Disponível em: <https://arxiv.org/abs/2310.06117>. Acesso em: abr. 2026.
