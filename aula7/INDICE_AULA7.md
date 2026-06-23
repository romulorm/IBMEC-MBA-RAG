# Índice — Aula 7: Query Enhancement
## Multi-Query RAG, RAG-Fusion e Step-Back Prompting
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 7 de 12 | **Carga:** 5h | **Proporção:** 30% teoria / 70% prática  
**Pré-requisito:** Aulas 1–6 concluídas (pipeline RAG funcional com OpenSearch + métricas RAGAS disponíveis)  
**Stack:** LangChain MultiQueryRetriever · vLLM · OpenSearch · LangFuse · RAGAS · Python asyncio · Langflow

---

## Estrutura de Arquivos

```
aula7/
│
├── INDICE_AULA7.md                                      ← Este arquivo
├── AVALIACAO_AULA7.md                                   ← Rubricas e critérios (professor)
│
├── teoria/
│   └── AULA7_TEORIA.md                                  ← Material teórico completo (8 seções)
│
├── labs/
│   ├── LAB1_Multi_Query_RAG.ipynb                       ← LangChain MultiQueryRetriever + deduplicação
│   ├── LAB2_Step_Back_Prompting.ipynb                   ← Step-Back com vLLM + comparação 5 queries
│   ├── LAB3_RAG_Fusion_Completo.ipynb                   ← Sub-queries paralelas + RRF + geração
│   ├── LAB4_Benchmark_N_Queries.ipynb                   ← Recall, latência e custo N=1/3/5
│   ├── LAB5_LangFuse_Custo_Traces.ipynb                 ← Visualização de custo real por abordagem
│   └── LAB6_Langflow_RAG_Fusion.ipynb                   ← RAG-Fusion visual no Langflow
│
├── exemplos/
│   ├── EXEMPLO1_Multi_Query_Minimo.ipynb                ← Multi-Query em 5 células (referência rápida)
│   └── EXEMPLO2_RAG_Fusion_Basico.ipynb                 ← RAG-Fusion mínimo sem asyncio (referência)
│
└── datasets/
    └── corpus_query_enhancement.json                    ← 15 docs + 20 queries com variações semânticas
```

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|---|---|---|---|---|
| **1. Motivação — Vocabulary Mismatch** | 20 min | Teoria | O problema fundamental: usuário diz "rescisão", doc diz "término antecipado" | `teoria/AULA7_TEORIA.md §1` |
| **2. Multi-Query RAG** | 25 min | Teoria | Geração de N perspectivas, deduplicação por similaridade, threshold ótimo | `teoria/AULA7_TEORIA.md §2–3` |
| **3. Demo — EXEMPLO1** | 10 min | Demo | Multi-Query mínimo em 5 células | `exemplos/EXEMPLO1_Multi_Query_Minimo.ipynb` |
| **4. LAB 1 — Multi-Query** | 40 min | Prática | MultiQueryRetriever + 4 variações + deduplicação com threshold coseno | `labs/LAB1_Multi_Query_RAG.ipynb` |
| **5. Step-Back Prompting** | 20 min | Teoria | Abstração de queries específicas para princípios gerais, tabela de transformações | `teoria/AULA7_TEORIA.md §4` |
| **6. LAB 2 — Step-Back** | 35 min | Prática | Implementação com vLLM, comparação em 5 perguntas conceituais jurídicas | `labs/LAB2_Step_Back_Prompting.ipynb` |
| **7. RAG-Fusion e RRF** | 20 min | Teoria | Sub-queries → retrieval paralelo → RRF matemático → fusão de rankings | `teoria/AULA7_TEORIA.md §5–6` |
| **8. Demo — EXEMPLO2** | 10 min | Demo | RAG-Fusion básico sem asyncio | `exemplos/EXEMPLO2_RAG_Fusion_Basico.ipynb` |
| **9. LAB 3 — RAG-Fusion** | 45 min | Prática | asyncio paralelismo + OpenSearch + RRF + geração Llama 3.1 | `labs/LAB3_RAG_Fusion_Completo.ipynb` |
| **10. Trade-off e Análise Econômica** | 15 min | Teoria | Recall vs latência vs custo — como calibrar N, análise de custo por 1000 queries | `teoria/AULA7_TEORIA.md §7` |
| **11. LAB 4 — Benchmark** | 30 min | Prática | Medir recall, latência e tokens para N=1, N=3, N=5 com gráficos | `labs/LAB4_Benchmark_N_Queries.ipynb` |
| **12. LAB 5 — LangFuse** | 20 min | Prática | Comparar traces das 3 abordagens, visualizar custo real | `labs/LAB5_LangFuse_Custo_Traces.ipynb` |
| **13. LAB 6 — Langflow** | 20 min | Prática | Construir RAG-Fusion como fluxo paralelo visual | `labs/LAB6_Langflow_RAG_Fusion.ipynb` |

---

## Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Explicar** o problema de vocabulary mismatch com exemplos reais do domínio jurídico e investigativo
2. **Implementar Multi-Query RAG** com LangChain MultiQueryRetriever gerando 4 variações semânticas por query
3. **Aplicar Step-Back Prompting** para transformar queries específicas em abstrações que aumentam o recall
4. **Construir um pipeline RAG-Fusion** completo com asyncio (paralelismo real) + RRF + geração com Llama 3.1
5. **Calcular o custo** de cada abordagem por 1.000 queries e justificar a escolha pelo trade-off recall/latência/custo
6. **Comparar pipelines** objetivamente com benchmark N=1/3/5 e visualizar os resultados no LangFuse

---

## Stack Tecnológico

| Componente | Ferramenta | Papel no Pipeline |
|---|---|---|
| Query Expansion | **LangChain MultiQueryRetriever** | Geração de N variações semânticas + deduplicação |
| Step-Back | **vLLM (Llama 3.1 8B)** via prompt engineering | Abstração de queries para princípios gerais |
| Paralelismo | **Python asyncio** | Execução simultânea de N sub-queries |
| Vector Store | **OpenSearch 3.x kNN** | Retrieval para cada variação de query |
| Score Fusion | **RRF (Reciprocal Rank Fusion)** | Fusão de rankings das sub-queries |
| LLM | **Llama 3.1 8B Instruct** | Geração de variações + resposta final |
| Servidor LLM | **vLLM** (PagedAttention) | API OpenAI-compatible em GPU |
| Observabilidade | **LangFuse** | Rastreamento de custo e latência por abordagem |
| Avaliação | **RAGAS** | Recall e qualidade comparativa |
| No-code | **Langflow** | Visualização do pipeline RAG-Fusion |

---

## Fichas de Técnicas RAG — Esta Aula

### #T10 — Multi-Query RAG

| Campo | Valor |
|---|---|
| **ID Oficial** | #T10 |
| **Introduzida em** | Aula 7 |
| **Problema que resolve** | Vocabulary mismatch: uma única formulação da query não captura todas as variações semânticas relevantes |
| **Ferramentas** | LangChain MultiQueryRetriever + vLLM (geração de variações) |
| **Quando usar** | Queries com terminologia ambígua ou informal; usuários leigos buscando em corpus técnico-jurídico |
| **Limitação principal** | N queries = N×latência e N×custo (se LLM externo); risco de ruído com N muito alto |
| **Baseline típico** | +18pp Recall vs Naive RAG com N=4 |

### #T11 — Step-Back Prompting

| Campo | Valor |
|---|---|
| **ID Oficial** | #T11 |
| **Introduzida em** | Aula 7 |
| **Problema que resolve** | Queries muito específicas que falham por falta de contexto geral no corpus indexado |
| **Ferramentas** | vLLM (Llama 3.1) com prompt de abstração |
| **Quando usar** | Queries sobre casos específicos que precisam de contexto doutrinário; perguntas sobre procedimentos que requerem conhecimento de princípios |
| **Limitação principal** | Pode sobre-abstrair e perder especificidade; não ajuda em queries já abstratas |
| **Baseline típico** | +15pp Answer Relevancy em queries conceituais |

### #T12 — RAG-Fusion

| Campo | Valor |
|---|---|
| **ID Oficial** | #T12 |
| **Introduzida em** | Aula 7 |
| **Problema que resolve** | Combina Multi-Query com RRF para maximizar tanto recall quanto precisão de ranking |
| **Ferramentas** | Python asyncio + OpenSearch + RRF manual |
| **Quando usar** | Sistemas de produção jurídicos que precisam de alta cobertura; investigações com múltiplas perspectivas legais |
| **Limitação principal** | Custo multiplicado por N; latência pode ser o gargalo mesmo com asyncio |
| **Baseline típico** | +22pp Context Recall vs Naive RAG com N=3 |

---

## Avaliação

Ver `AVALIACAO_AULA7.md` para rubricas completas.

| Entregável | Peso | Lab |
|---|---|---|
| Multi-Query funcionando com 4 variações e deduplicação documentada | 25% | LAB1 |
| Step-Back com comparação em 5 queries jurídicas (tabela antes/depois) | 20% | LAB2 |
| RAG-Fusion completo com asyncio + RRF + geração funcionando | 25% | LAB3 |
| Benchmark documentado: recall, latência e custo para N=1/3/5 | 20% | LAB4 |
| Análise de trade-off escrita com recomendação justificada | 10% | LAB4 + LAB5 |

---

## Referências Bibliográficas (ABNT)

MA, X. et al. **Query Rewriting for Retrieval-Augmented Large Language Models**. In: *International Conference on Learning Representations (ICLR)*, Vienna, 2023. arXiv:2305.14283.

RACKAUCKAS, Z. **RAG-Fusion: a New Take on Retrieval-Augmented Generation**. arXiv:2402.03367, 2024.

CORMACK, G. V.; CLARKE, C. L. A.; BUETTCHER, S. **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods**. In: *Proceedings of SIGIR*, 2009. p. 758–759.

ZHENG, H. S. et al. **Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**. In: *ICLR*, Vienna, 2024. arXiv:2310.06117.

TANG, Y.; YANG, Y. **MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries**. arXiv:2401.15391, 2024.

LANGCHAIN. **MultiQueryRetriever**. Documentação oficial, 2024. Disponível em: <https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever>. Acesso em: abr. 2026.

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217, 2023.
