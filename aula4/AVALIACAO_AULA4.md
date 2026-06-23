# Avaliação — Aula 4: OpenSearch Completo — Dense, Hybrid Search, Neural Sparse e Contextual Retrieval
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

---

## Critérios Gerais

| Critério | Peso |
|---|---|
| Funcionalidade técnica (código executa sem erros) | 40% |
| Análise crítica e comparativa dos resultados | 30% |
| Qualidade da implementação e comentários | 20% |
| Aplicação ao domínio jurídico/segurança pública | 10% |

---

## Rubricas por Laboratório

### LAB1 — Índice Híbrido no OpenSearch 3.x (20%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Índice criado no OpenSearch 3.x com Neural Search Plugin e ML Commons habilitados. Campo `embedding` (kNN, dim=1024, HNSW) e campo `conteudo` (text, BM25). Mapeamento correto com `space_type: cosinesimil`. Ingestão de ao menos 10 documentos com vetores BGE-M3 validados. |
| **Bom (70–89%)** | Índice criado corretamente mas com pequenas falhas no mapeamento (ex.: dimensão errada, espaço L2 em vez de cosine). Documentos ingeridos mas sem validação dos vetores. |
| **Regular (50–69%)** | Índice criado mas sem campo kNN ou sem campo text simultâneos. Apenas um modo de busca disponível. |
| **Insuficiente (<50%)** | Índice não criado ou com erros críticos que impedem qualquer busca. |

### LAB2 — Search Pipeline com RRF (20%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Search pipeline criado com `normalization-processor` (RRF) e ativado no índice. Consulta híbrida executada com `hybrid` query mostrando scores fundidos. Comparação entre RRF e arithmetic normalization documentada. |
| **Bom (70–89%)** | Pipeline criado e funcional mas sem comparação entre métodos de fusão. Apenas um método implementado. |
| **Regular (50–69%)** | Pipeline criado com erros parciais. Scores fundidos mas sem evidência de RRF corretamente aplicado. |
| **Insuficiente (<50%)** | Pipeline não criado ou busca híbrida não executada. |

### LAB3 — Análise Comparativa Busca Híbrida (30%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Comparação sistemática entre: (a) kNN puro, (b) BM25 puro, (c) hybrid com RRF. Métricas MRR@10 e Recall@10 calculadas para ao menos 5 queries jurídicas com relevância anotada. Análise qualitativa de ao menos 2 casos onde hybrid supera ou falha em relação aos modos únicos. |
| **Bom (70–89%)** | Comparação entre ao menos 2 modos com métricas calculadas. Análise qualitativa superficial. |
| **Regular (50–69%)** | Apenas comparação visual/qualitativa sem métricas objetivas. |
| **Insuficiente (<50%)** | Sem comparação entre modos ou análise ausente. |

### LAB4 — Neural Sparse Search (10%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Modelo de sparse encoding carregado via OpenSearch ML. Índice com campo `sparse_embedding` criado. Ingestão com `neural_sparse` processor e consulta com `neural_sparse` query funcional. |
| **Bom (70–89%)** | Implementação funcional mas com limitações (ex.: modelo local em vez de OpenSearch ML, ou pipeline de ingestão manual). |
| **Regular (50–69%)** | Implementação parcial: índice criado mas ingestão ou consulta não funcional. |
| **Insuficiente (<50%)** | Lab não executado ou sem evidência de neural sparse search. |

### LAB5 — Contextual Retrieval (#T09) (20%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Prompt de contextualização aplicado aos 100 chunks via vLLM. Chunks enriquecidos reindexados no OpenSearch 3.x. RAGAS executado antes e depois — Context Precision e Context Recall calculados. Delta positivo comprovado e registrado no LangFuse. |
| **Bom (70–89%)** | Enriquecimento funcional mas RAGAS executado apenas antes ou apenas depois, sem comparação completa. Ou Delta presente mas não registrado no LangFuse. |
| **Regular (50–69%)** | Contextualização implementada mas re-indexação incompleta ou RAGAS com erros. |
| **Insuficiente (<50%)** | Lab não executado ou sem evidência de Contextual Retrieval. |

### LAB6 — Dashboard LangFuse Comparativo (10%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Benchmark executado para BM25, Dense kNN e Hybrid RRF. MRR@5 e latência calculados para todas as queries. Dashboard Matplotlib gerado com gráficos de barras por estratégia e por query. Scores registrados no LangFuse e trace visível. CSVs exportados. |
| **Bom (70–89%)** | Benchmark funcional mas com apenas 2 estratégias comparadas, ou dashboard sem gráfico detalhado por query. |
| **Regular (50–69%)** | Apenas tabela de texto comparando estratégias, sem gráficos ou sem LangFuse. |
| **Insuficiente (<50%)** | Lab não executado. |

---

## Entregáveis

- Notebooks executados (cells com output visível) para os 6 Labs
- Ao menos um print/screenshot do painel do OpenSearch Dashboards mostrando o índice criado (bônus)
- Comentários explicando as decisões técnicas em cada lab

**Prazo:** Conforme calendário do MBA  
**Formato de entrega:** Google Colab + link compartilhado
