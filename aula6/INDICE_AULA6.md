# Índice — Aula 6: Indexação Avançada
## Hierarchical Indexing, RAPTOR e HyDE
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 6 de 12 | **Carga:** 5h | **Proporção:** 25% teoria / 75% prática  
**Pré-requisito:** Aulas 1–5 concluídas (baseline RAGAS da Aula 2 necessário para comparação)  
**Stack:** LlamaIndex · RAPTOR · UMAP · scikit-learn GMM · RAGAS · vLLM · FAISS · OpenSearch  
**Referência:** ABNT NBR 6023:2018 | NBR 10520:2023

---

## Estrutura de Arquivos

```
aula6/
├── INDICE_AULA6.md                           ← este arquivo
├── AVALIACAO_AULA6.md
├── teoria/
│   └── AULA6_TEORIA.md                       ← 30% do conteúdo (~12 seções)
├── exemplos/
│   ├── EXEMPLO1_Parent_Child_Minimo.ipynb    ← Parent-Child em 3 células
│   └── EXEMPLO2_HyDE_Minimo.ipynb            ← HyDE em 4 células
├── labs/
│   ├── LAB1_LlamaIndex_Parent_Child.ipynb    ← Implementação completa
│   ├── LAB2_RAPTOR_Clustering_Arvore.ipynb   ← UMAP + GMM + sumarização recursiva
│   ├── LAB3_HyDE_Gap_Semantico.ipynb         ← Pipeline HyDE + visualização geométrica
│   └── LAB4_RAGAS_Comparacao_Tecnicas.ipynb  ← Dashboard comparativo
└── datasets/
    └── corpus_indexacao_avancada.json        ← 10 docs jurídicos + 7 queries
```

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|-------|---------|------|----------|---------|
| 1 | 30 min | Teoria | Motivação: falhas do chunking plano; panorama das 3 técnicas | AULA6_TEORIA.md §1-2 |
| 2 | 15 min | Demo | Exemplo 1: Parent-Child em 3 células | EXEMPLO1 |
| 3 | 45 min | Lab | LAB 1: Parent-Child completo com comparação vs flat | LAB1 |
| 4 | 20 min | Teoria | RAPTOR: algoritmo, UMAP, GMM, árvore | AULA6_TEORIA.md §3 |
| 5 | 50 min | Lab | LAB 2: RAPTOR + visualização 2D | LAB2 |
| 6 | 15 min | Teoria | HyDE: geometria do gap semântico | AULA6_TEORIA.md §4 |
| 7 | 15 min | Demo | Exemplo 2: HyDE mínimo | EXEMPLO2 |
| 8 | 40 min | Lab | LAB 3: HyDE completo com visualização | LAB3 |
| 9 | 45 min | Lab | LAB 4: Comparação RAGAS + dashboard | LAB4 |
| 10 | 25 min | Síntese | Guia de decisão, discussão, próximos passos | AULA6_TEORIA.md §5-6 |

---

## Objetivos de Aprendizagem

1. Implementar Parent-Child Retriever com LlamaIndex e identificar quando ele supera o chunking plano
2. Construir um índice RAPTOR com UMAP + GMM + sumarização recursiva sobre corpus jurídico
3. Aplicar HyDE e medir a redução do gap semântico com métricas de similaridade coseno
4. Visualizar geometricamente a diferença entre embedding de query e embedding de documento hipotético
5. Comparar os 3 pipelines com o baseline Naive RAG usando o framework RAGAS
6. Selecionar a estratégia de indexação adequada para diferentes cenários jurídicos

---

## Stack Tecnológico

| Componente | Ferramenta | Papel na Aula |
|---|---|---|
| Hierarquia de índices | LlamaIndex (HierarchicalNodeParser, AutoMergingRetriever) | Lab 1 — Parent-Child |
| Clustering dimensional | UMAP + scikit-learn GaussianMixture | Lab 2 — RAPTOR |
| Sumarização recursiva | vLLM (Llama 3.1 8B) via LangChain | Lab 2 — RAPTOR |
| Embedding de hipotéticos | sentence-transformers (BGE-M3) | Lab 3 — HyDE |
| Busca vetorial | FAISS (fallback) / OpenSearch kNN | Labs 1, 3 |
| Avaliação | RAGAS (Context Precision, Recall, Faithfulness, Answer Relevancy) | Lab 4 |
| Visualização | matplotlib + seaborn | Labs 2, 3, 4 |
| Runtime | Python 3.11+ / Google Colab | Todos |

---

## Fichas de Técnicas RAG — Esta Aula

### #T07 — Hierarchical Indexing (Parent-Child Retriever)

| Campo | Valor |
|---|---|
| ID Oficial | #T07 |
| Introduzida em | Aula 6 |
| Problema que resolve | Granularidade dupla: busca precisa com contexto rico |
| Ferramentas | LlamaIndex: HierarchicalNodeParser + AutoMergingRetriever |
| Quando usar | Documentos com estrutura hierárquica clara (leis, acórdãos) |
| Limitação principal | Chunk size ratio deve ser ≥ 1:4 (filho:pai) |
| Baseline típico | +16pp Context Precision vs Naive RAG |

### #T08 — RAPTOR

| Campo | Valor |
|---|---|
| ID Oficial | #T08 |
| Introduzida em | Aula 6 |
| Problema que resolve | Queries de síntese sobre corpus extensos (>100 docs) |
| Ferramentas | UMAP + GaussianMixture + LLM (sumarização) |
| Quando usar | Repositórios grandes; análise de tendências e padrões |
| Limitação principal | Custo computacional na indexação; corpus pequeno degrada qualidade |
| Baseline típico | +18pp Context Recall vs Naive RAG |

### #T05 — HyDE (Hypothetical Document Embeddings)

| Campo | Valor |
|---|---|
| ID Oficial | #T05 |
| Introduzida em | Aula 6 |
| Problema que resolve | Gap semântico entre queries coloquiais e corpus técnico |
| Ferramentas | LLM (geração) + SentenceTransformer (embedding do hipotético) |
| Quando usar | Usuários leigos; linguagem informal vs corpus técnico-jurídico |
| Limitação principal | Latência extra (+500-2000ms/query); falha com entidades específicas |
| Baseline típico | +13pp Answer Relevancy vs Naive RAG |

---

## Avaliação

| # | Entregável | Peso | Ferramenta |
|---|---|---|---|
| E1 | Parent-Child implementado e comparado com flat | 25% | RAGAS + print do AutoMerging log |
| E2 | Visualização RAPTOR com UMAP 2D (arquivo PNG) | 25% | matplotlib |
| E3 | Pipeline HyDE funcional com gap semântico visualizado | 25% | Gráfico UMAP + delta score |
| E4 | Dashboard RAGAS comparando 4 técnicas | 25% | relatorio_final_aula6.json |

**Aprovação:** E4 concluído + melhoria de pelo menos uma técnica > 5% vs Naive RAG.

---

## Referências Bibliográficas (ABNT)

SARTHI, P. et al. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**. In: *ICLR*, Vienna, 2024. arXiv:2401.18059.

GAO, L. et al. **Precise Zero-Shot Dense Retrieval without Relevance Labels**. In: *ACL*, Toronto, 2023. arXiv:2212.10496.

LLAMAINDEX. **Auto-Merging Retriever**. LlamaIndex Documentation, 2024. Disponível em: <https://docs.llamaindex.ai>. Acesso em: abr. 2026.

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. In: *EACL*, Malta, 2024. arXiv:2309.15217.

MCINNES, L.; HEALY, J. **UMAP: Uniform Manifold Approximation and Projection**. arXiv:1802.03426, 2018.
