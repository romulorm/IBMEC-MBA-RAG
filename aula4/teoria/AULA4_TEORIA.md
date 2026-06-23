# Aula 4 — OpenSearch Completo: Dense, Hybrid Search, Neural Sparse e Contextual Retrieval
## Vetores + Texto, RRF, Search Pipelines e Contextual Retrieval
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Carga teórica:** 1h30 (30% da aula) | **Proporção:** 30% teoria / 70% prática  
**Referência normativa:** ABNT NBR 6023:2018 (referências), NBR 10520:2023 (citações)

---

## §1 — Por que a Busca Vetorial Pura Não é Suficiente

Na Aula 3 implementamos reranking com BGE-Reranker e pipelines modulares. O retrieval, porém, ainda dependia de embeddings densos (BGE-M3) como único mecanismo de busca. Esta abordagem apresenta limitações importantes no domínio jurídico:

**Problema 1 — Termos técnicos raros:** Artigos de lei como "art. 5º, LXIII, CF/88" ou siglas como "HC", "RE", "STF" têm representações vetoriais que dependem fortemente da presença desses tokens nos dados de pré-treinamento. Modelos treinados predominantemente em inglês codificam mal estas expressões.

**Problema 2 — Match exato de números:** Queries como "Lei 9.099/1995" ou "Súmula 231 do STJ" exigem correspondência exata de numerais. Embeddings tendem a aproximar semanticamente documentos sobre "juizados especiais" mesmo quando o número da lei é diferente — o que pode ser incorreto juridicamente.

**Problema 3 — Out-of-vocabulary (OOV):** Termos altamente especializados (neologismos jurídicos, nomes de partes processuais, datas de julgamento) podem não estar no vocabulário do modelo de embedding, gerando vetores de qualidade degradada.

Em contraste, a busca lexical (BM25) é excelente para match exato de termos e não sofre com OOV, mas falha quando há variação semântica: "réu" e "acusado" têm BM25-score zero de sobreposição mas alta similaridade semântica.

A **busca híbrida** combina os dois mundos: o power do BM25 para precisão lexical com o poder dos embeddings para captura semântica.

---

## §2 — Arquitetura de Busca Híbrida no OpenSearch

O OpenSearch 3.x suporta busca híbrida nativamente através do tipo de query `hybrid`, com suporte integrado via **Neural Search Plugin** e **ML Commons Plugin**. A arquitetura envolve três camadas:

### 2.1 Índice com Campos Duplos

Um documento no índice híbrido possui simultaneamente:

- **Campo vetorial** (`knn_vector`): vetor denso de alta dimensão gerado por um modelo de embedding (BGE-M3, dim=1024)
- **Campo textual** (`text`): texto bruto indexado com o algoritmo BM25 padrão do Lucene/OpenSearch

```json
{
  "mappings": {
    "properties": {
      "conteudo": {
        "type": "text",
        "analyzer": "portuguese"
      },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "faiss",
          "parameters": {
            "ef_construction": 128,
            "m": 16
          }
        }
      }
    }
  }
}
```

### 2.2 Query Híbrida

A query `hybrid` executa as sub-queries em paralelo e passa os resultados para o search pipeline:

```json
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "conteudo": {
              "query": "habeas corpus crime ambiental"
            }
          }
        },
        {
          "knn": {
            "embedding": {
              "vector": [0.021, -0.043, ...],
              "k": 10
            }
          }
        }
      ]
    }
  }
}
```

### 2.3 Search Pipeline

O search pipeline é o componente que **normaliza e combina** os scores das sub-queries. Sem pipeline, os scores de BM25 (tipicamente 0–20) e kNN (cosine similarity, 0–1) são incomparáveis. O pipeline resolve isso:

```
Query → [BM25 search] ─┐
                        ├──→ [Normalization Processor] → [Score Combination] → Resultados
Query → [kNN search]  ─┘
```

---

## §3 — Search Pipelines em Detalhe

### 3.1 O que é um Search Pipeline

Um search pipeline é uma sequência de processadores configurada no OpenSearch que intercede no fluxo de busca. Existem três tipos de processadores:

- **Search request processors**: transformam a query antes da busca (ex.: reescrita de query)
- **Search response processors**: transformam os resultados após a busca (ex.: renaming de campos)
- **Search phase results processors**: transformam os resultados entre fases (ex.: normalization)

Para busca híbrida, usamos o `normalization-processor`, que é um **search phase results processor**.

### 3.2 Criando um Search Pipeline

```json
PUT /_search/pipeline/hybrid_rrf_pipeline
{
  "description": "Pipeline de busca híbrida com RRF para corpus jurídico",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "rrf"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.3, 0.7]
          }
        }
      }
    }
  ]
}
```

### 3.3 Associando o Pipeline ao Índice

```json
PUT /corpus_juridico/_settings
{
  "index.search.default_pipeline": "hybrid_rrf_pipeline"
}
```

Ou passando o pipeline por query:

```
GET /corpus_juridico/_search?search_pipeline=hybrid_rrf_pipeline
```

---

## §4 — Reciprocal Rank Fusion (RRF)

### 4.1 Conceito

O RRF é um algoritmo de fusão de rankings proposto por Cormack et al. (2009). Em vez de combinar os **scores brutos** das sub-queries (que têm escalas incompatíveis), ele combina as **posições** (ranks) nos resultados.

Para um documento `d` que aparece na posição `r` em uma lista de resultados, o score RRF é:

```
RRF(d, r) = 1 / (k + r)
```

Onde `k` é uma constante de suavização (tipicamente `k = 60`). O score final de um documento é a **soma** dos scores RRF de todas as listas onde ele aparece:

```
RRF_final(d) = Σ [ 1 / (k + r_i(d)) ]
```

### 4.2 Exemplo Numérico

Considere uma query jurídica sobre "prisão preventiva". Temos:

| Doc | Rank BM25 | Rank kNN | RRF BM25 (k=60) | RRF kNN (k=60) | RRF Total |
|---|---|---|---|---|---|
| D1 (art. 312 CPP) | 1 | 3 | 1/61 = 0.0164 | 1/63 = 0.0159 | **0.0323** |
| D2 (doutrina cautelar) | 5 | 1 | 1/65 = 0.0154 | 1/61 = 0.0164 | **0.0318** |
| D3 (habeas corpus) | 2 | 4 | 1/62 = 0.0161 | 1/64 = 0.0156 | **0.0317** |
| D4 (apenas em BM25) | 3 | N/A | 1/63 = 0.0159 | 0 | **0.0159** |

O RRF favorece documentos que aparecem bem ranqueados em **ambas** as listas, penalizando documentos relevantes apenas em uma delas — o que é o comportamento desejado para busca híbrida.

### 4.3 RRF vs. Min-Max Normalization

O OpenSearch suporta duas técnicas de normalização:

**Min-Max Normalization:**
```
score_norm = (score - min_score) / (max_score - min_score)
```
Sensível a outliers: um único documento com score muito alto distorce toda a normalização.

**RRF (Reciprocal Rank Fusion):**
Baseado em posição. Robusto a outliers de score. Não requer normalização por escala absoluta. É a **técnica recomendada** para a maioria dos casos de busca híbrida.

| Característica | Min-Max | RRF |
|---|---|---|
| Sensibilidade a outliers | Alta | Baixa |
| Interpretabilidade | Média | Alta |
| Necessidade de calibração | Alta (depende de distribuição) | Baixa (k=60 funciona bem na maioria dos casos) |
| Implementação no OpenSearch | `min_max` | `rrf` |

### 4.4 Pesos na Combinação Aritmética

Após a normalização, o OpenSearch combina os scores normalizados com pesos configuráveis:

```
score_final = w1 * score_norm_bm25 + w2 * score_norm_knn
```

O parâmetro `weights` no search pipeline controla essa combinação. Para domínios jurídicos, experimentos com corpora brasileiros sugerem:

- **Queries factuais** (artigo X da lei Y): pesos mais altos para BM25 (ex.: [0.7, 0.3])
- **Queries semânticas** (responsabilidade objetiva ambiental): pesos mais altos para kNN (ex.: [0.3, 0.7])
- **Queries mistas**: pesos equilibrados (ex.: [0.5, 0.5])

O parâmetro `alpha` em algumas implementações equivale ao peso do componente vetorial (1-alpha = peso BM25).

---

## §5 — Neural Sparse Search

### 5.1 Motivação

A busca esparsa neural (Neural Sparse) é uma abordagem que usa modelos de linguagem para gerar vetores esparsos — com a maioria dos valores igual a zero — em vez de vetores densos de alta dimensão. O modelo mais conhecido desta família é o **SPLADE** (Sparse Lexical and Expansion Model).

A vantagem principal é a **interpretabilidade**: cada dimensão do vetor corresponde a um token do vocabulário, com um peso indicando sua importância para o documento/query. Isso permite auditoria dos resultados — crucial no contexto jurídico.

### 5.2 Como Funciona o SPLADE

O SPLADE usa um modelo baseado em BERT para:

1. Processar o texto de entrada
2. Projetar cada token na dimensão do vocabulário
3. Aplicar ReLU + log para gerar pesos esparsos
4. Expandir termos implícitos (ex.: "veículo" → peso também em "carro", "automóvel")

```
SPLADE(texto) → vetor esparso {termo: peso, termo: peso, ...}
Ex: "prisão preventiva" → {"prisão": 2.1, "preventiva": 1.8, "cautelar": 1.2, "preso": 0.9, ...}
```

### 5.3 Neural Sparse no OpenSearch

O OpenSearch 3.x suporta Neural Sparse nativamente via **ML Commons Plugin**, que permite registrar e hospedar modelos SPLADE diretamente no cluster. Há dois modos:

**Modo bi-encoder (recomendado para produção):**
- Documento e query são codificados separadamente
- O modelo `doc` codifica documentos durante a ingestão
- O modelo `query` (mais leve) codifica queries em tempo real
- Mais eficiente para grandes volumes

**Modo único (para experimentação):**
- Mesmo modelo codifica documento e query
- Menor latência de configuração, mas menos eficiente em escala

### 5.4 Mapeamento e Ingestão

```json
{
  "mappings": {
    "properties": {
      "sparse_embedding": {
        "type": "rank_features"
      }
    }
  }
}
```

O tipo `rank_features` armazena vetores esparsos de forma eficiente usando o índice invertido do Lucene.

---

## §6 — Contextual Retrieval (#T09)

### 6.1 O Problema dos Chunks sem Contexto

Quando um documento é dividido em chunks para indexação, cada trecho perde a informação de **onde** ele se encaixa no documento maior. Considere este exemplo de um acórdão:

> *"O Tribunal, por unanimidade, negou provimento ao recurso."*

Isoladamente, este chunk é praticamente inútil para recuperação: não sabemos de qual caso se trata, qual crime foi julgado, ou qual o entendimento consolidado. A **busca vetorial** vai gerar um embedding que representa essa sentença genérica, não o acórdão específico.

O **Contextual Retrieval** (Anthropic, 2024) resolve este problema adicionando um **resumo contextual** a cada chunk antes da indexação. O texto do chunk é enriquecido com uma frase descrevendo seu papel no documento completo:

> *"[CONTEXTO: Acórdão do STJ, REsp 1.234.567/SP, caso de crime ambiental — corte de árvores em APP. Dispositivo: negado provimento ao recurso do Ministério Público. Relator: Min. João Silva, j. 15/03/2024] O Tribunal, por unanimidade, negou provimento ao recurso."*

### 6.2 Arquitetura do Contextual Retrieval

```
┌─────────────────────────────────────────────────────────┐
│              PIPELINE CONTEXTUAL RETRIEVAL               │
│                                                         │
│  Documento ──→ Chunking ──→ [Para cada chunk]:           │
│                                                         │
│      Prompt ao vLLM:                                    │
│      "Documento: {doc_completo}                         │
│       Chunk: {chunk_texto}                              │
│       Gere uma frase concisa descrevendo o papel        │
│       deste trecho no documento, em português."         │
│              ↓                                          │
│      Contexto Gerado: "{contexto situacional}"          │
│              ↓                                          │
│      Chunk Enriquecido = contexto + "\n\n" + chunk_texto│
│              ↓                                          │
│      Embedding(chunk_enriquecido) → OpenSearch kNN      │
└─────────────────────────────────────────────────────────┘
```

### 6.3 Implementação com vLLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Conectar ao vLLM
llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0.0,
    max_tokens=150,
)

CONTEXTO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente que analisa documentos jurídicos brasileiros."),
    ("human", (
        "Documento completo:\n{documento}\n\n"
        "Trecho a contextualizar:\n{chunk}\n\n"
        "Gere uma frase concisa (máximo 2 linhas) descrevendo o papel deste trecho "
        "no documento, incluindo: tipo de documento, partes, assunto principal e "
        "relevância do trecho. Responda APENAS com a frase de contexto."
    ))
])

async def enriquecer_chunk(documento_completo: str, chunk_texto: str) -> str:
    """Gera contexto situacional para um chunk via vLLM."""
    resposta = await llm.ainvoke(
        CONTEXTO_PROMPT.format_messages(
            documento=documento_completo[:3000],  # primeiros 3000 chars como contexto
            chunk=chunk_texto
        )
    )
    contexto = resposta.content.strip()
    return f"[CONTEXTO: {contexto}]\n\n{chunk_texto}"
```

### 6.4 Medição de Impacto com RAGAS

O impacto do Contextual Retrieval deve ser mensurado com **Context Precision** (RAGAS), que avalia se os documentos recuperados são realmente relevantes para a query:

```python
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset

# Avaliar ANTES (chunks sem contexto)
resultados_antes = evaluate(
    dataset=Dataset.from_list(pares_avaliacao),
    metrics=[context_precision, context_recall]
)

# Indexar chunks enriquecidos e avaliar DEPOIS
resultados_depois = evaluate(
    dataset=Dataset.from_list(pares_avaliacao_enriquecidos),
    metrics=[context_precision, context_recall]
)

print(f"Context Precision — Antes: {resultados_antes['context_precision']:.3f}")
print(f"Context Precision — Depois: {resultados_depois['context_precision']:.3f}")
print(f"Melhoria: {(resultados_depois['context_precision'] - resultados_antes['context_precision'])*100:.1f}%")
```

Estudos da Anthropic reportam melhoria de ~35% em Context Precision com esta técnica. No LAB5, mediremos o impacto real no nosso corpus jurídico.

### 6.5 Trade-offs

| Aspecto | Sem Contextual Retrieval | Com Contextual Retrieval |
|---|---|---|
| Custo de ingestão | 1 embedding por chunk | 1 chamada LLM + 1 embedding por chunk |
| Qualidade do retrieval | Baseline | +20–40% em Context Precision (estimado) |
| Tempo de ingestão | ~2s/100 chunks | ~45s/100 chunks (com vLLM local) |
| Custo em produção | Baixo | Médio (LLM inference na ingestão) |
| Auditabilidade | Baixa | Alta (contextos visíveis no índice) |

> **Insight para o domínio jurídico:** Acórdãos e laudos têm estrutura altamente contextual — ementa, relatório, voto, dispositivo. Um chunk do "dispositivo" sem saber o caso é inútil. O Contextual Retrieval é especialmente impactante neste domínio.

---

## §7 — Avaliação de Busca Híbrida: Métricas

### 7.1 Mean Reciprocal Rank (MRR)

O MRR mede a posição do primeiro resultado relevante:

```
MRR = (1/|Q|) * Σ (1 / rank_i)
```

Onde `rank_i` é a posição do primeiro documento relevante para a query `i`. MRR varia de 0 a 1 — quanto maior, melhor.

**Exemplo:** Para 3 queries jurídicas:
- Query 1: primeiro resultado relevante na posição 1 → 1/1 = 1.0
- Query 2: primeiro resultado relevante na posição 3 → 1/3 = 0.33
- Query 3: nenhum resultado relevante nos top-10 → 0

MRR = (1.0 + 0.33 + 0.0) / 3 = **0.44**

### 7.2 Recall@K

O Recall@K mede a proporção de documentos relevantes recuperados nos top-K resultados:

```
Recall@K = |Relevantes ∩ Top-K| / |Relevantes|
```

Para avaliação de busca híbrida, usamos tipicamente Recall@10 e Recall@20.

### 7.3 NDCG@K (Normalized Discounted Cumulative Gain)

O NDCG considera não apenas se um documento é relevante, mas **o quão relevante** ele é e **em que posição** aparece:

```
DCG@K = Σ (rel_i / log2(i+1))
NDCG@K = DCG@K / IDCG@K
```

Onde `IDCG@K` é o DCG do ranking ideal. NDCG é a métrica recomendada quando os documentos têm graus de relevância diferenciados (ex.: altamente relevante, parcialmente relevante, irrelevante).

### 7.4 Avaliação Comparativa: Tabela-Resumo

| Métrica | BM25 puro | kNN puro | Hybrid RRF |
|---|---|---|---|
| MRR@10 | ~0.52 | ~0.48 | **~0.67** |
| Recall@10 | ~0.61 | ~0.58 | **~0.74** |
| NDCG@10 | ~0.55 | ~0.51 | **~0.69** |

*Valores ilustrativos baseados em benchmarks de retrieval híbrido em corpora jurídicos portugueses e brasileiros.*

---

## Resumo da Aula

Nesta aula, estudamos:

1. **Por que a busca vetorial pura falha** em contextos jurídicos com terminologia técnica e match exato
2. **Arquitetura de índice híbrido** no OpenSearch 3.x com Neural Search Plugin e ML Commons Plugin
3. **Search Pipelines** como mecanismo de normalização e fusão de scores
4. **RRF (Reciprocal Rank Fusion)** como técnica robusta de combinação baseada em posição
5. **Neural Sparse Search** com SPLADE: sparse vectors neurais interpretáveis via ML Commons
6. **Contextual Retrieval (#T09)**: enriquecimento de chunks com contexto situacional via vLLM, medição de impacto com RAGAS
7. **Métricas de avaliação** (MRR, Recall@K, NDCG@K, Context Precision) para comparação objetiva

Na prática (70% da aula), implementaremos cada um destes conceitos passo a passo nos laboratórios, culminando no LAB6 com dashboard comparativo no LangFuse.

---

## Referências Bibliográficas (ABNT)

CORMACK, G. V.; CLARKE, C. L. A.; BUETTCHER, S. **Reciprocal rank fusion outperforms condorcet and individual rank learning methods**. In: *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval*, 2009. p. 758-759.

CHEN, Y. et al. **Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models**. arXiv:2210.11934, 2022.

FORMAL, T. et al. **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking**. arXiv:2107.05720, 2021.

LIN, J.; MA, X. **A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques**. arXiv:2106.14807, 2021.

OPENSEARCH PROJECT. **Hybrid Search**. Disponível em: <https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/>. Acesso em: abr. 2026.

OPENSEARCH PROJECT. **Neural Sparse Search**. Disponível em: <https://docs.opensearch.org/latest/vector-search/ai-search/neural-sparse-search/>. Acesso em: abr. 2026.

ANTHROPIC. **Introducing Contextual Retrieval**. Blog Anthropic, set. 2024. Disponível em: <https://www.anthropic.com/news/contextual-retrieval>. Acesso em: abr. 2026.

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217, 2023.

ROBERTSON, S.; ZARAGOZA, H. **The Probabilistic Relevance Framework: BM25 and Beyond**. *Foundations and Trends in Information Retrieval*, v. 3, n. 4, p. 333-389, 2009.
