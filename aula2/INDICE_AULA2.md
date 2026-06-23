# Índice — Aula 2: Ingestão, Chunking e Naive RAG
## A Base de Tudo
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 2 de 12 | **Carga:** 5h | **Proporção:** 25% teoria / 75% prática
**Pré-requisito:** Aula 1 concluída (ambiente Ollama + OpenSearch operacional) | **Stack:** Docling · LangChain · NLTK · BGE-M3 (via Ollama) · OpenSearch · Ollama (llama3.2:3b)

---

## Estrutura de Arquivos

```
aula2/
│
├── INDICE_AULA2.md                              ← Este arquivo
├── AVALIACAO_AULA2.md                           ← Rubricas e critérios (professor)
│
├── teoria/
│   └── AULA2_TEORIA.md                          ← Material teórico completo (11 seções)
│
├── labs/                                        ← 5 Laboratórios práticos (25% teoria / 75% prática)
│   ├── LAB1_Docling_Ingestao_Avancada.ipynb    ← PDF complexo → DoclingDocument → LangChain
│   ├── LAB2_Comparacao_Chunking.ipynb          ← 5 estratégias no mesmo texto
│   ├── LAB3_Analise_Qualitativa_Chunks.ipynb   ← Métricas de qualidade de chunks
│   ├── LAB4_Naive_RAG_Pipeline_Completo.ipynb  ← Pipeline ponta-a-ponta (Docling+BGE-M3+OS+Ollama)
│   ├── LAB5_Registro_Baseline.ipynb            ← 5 queries + avaliação 5D + exportação
│   └── docs/                                   ← Documentação auxiliar
│       ├── README_AULA2.txt                    ← Orientações gerais da aula
│       ├── README_LAB5.md                      ← Guia detalhado do LAB5
│       └── README_LABS_3_4.md                  ← Guia dos LABs 3 e 4
│
├── exemplos/
│   ├── EXEMPLO1_Estrategias_Chunking.ipynb     ← Chunking comparativo (referência rápida)
│   ├── EXEMPLO2_Docling_Ingestao.ipynb         ← Docling básico (referência rápida)
│   └── EXEMPLO3_Naive_RAG_Completo.ipynb       ← Pipeline Naive RAG completo com FAISS (referência de estudo)
│
└── datasets/
    ├── Manual_DPCA_atualizado.pdf              ← PDF DIGITAL (texto extraível) — usado por LAB1, EXEMPLO2 e LAB4/EXEMPLO3 (modo Docling)
    ├── Laudo.pdf                                ← PDF ESCANEADO (imagem de texto) — caso OCR no LAB1 / EXEMPLO2
    ├── corpus_juridico_sample.json             ← 8 docs + 5 queries de avaliação
    └── (demais PDFs auxiliares: STF_Decisao_Monocratica, IRDR_48, TCU, Transpetro, PACC_PCDF)
```

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|---|---|---|---|---|
| **1. Abertura** | 15 min | Teoria | Por que chunking define tudo — diagrama e motivação | `teoria/AULA2_TEORIA.md §1` |
| **2. Estratégias de Chunking** | 45 min | Teoria | 5 estratégias com diagramas e tabelas comparativas | `teoria/AULA2_TEORIA.md §2–6` |
| **3. LAB 1 — Docling** | 60 min | Prática | PDF complexo → Markdown estruturado → LangChain Docs | `labs/LAB1_Docling_Ingestao_Avancada.ipynb` |
| **4. LAB 2 — Chunking** | 45 min | Prática | Aplicar 5 estratégias no mesmo acórdão e comparar | `labs/LAB2_Comparacao_Chunking.ipynb` |
| **5. LAB 3 — Qualidade** | 30 min | Prática | Coerência, órfãos, artigos fracionados, UMAP | `labs/LAB3_Analise_Qualitativa_Chunks.ipynb` |
| **6. Naive RAG — Teoria** | 20 min | Teoria | Arquitetura, stack, BGE-M3, Ollama, limitações | `teoria/AULA2_TEORIA.md §8–9` |
| **7. LAB 4 — Pipeline** | 60 min | Prática | Pipeline completo Docling+BGE-M3+OpenSearch+Ollama | `labs/LAB4_Naive_RAG_Pipeline_Completo.ipynb` |
| **8. LAB 5 — Baseline** | 45 min | Prática | 5 queries, avaliação 5D, exportação CSV/Excel | `labs/LAB5_Registro_Baseline.ipynb` |

---

## Objetivos de Aprendizagem (conforme ementa)

Ao final desta aula, o aluno será capaz de:

1. Compreender por que a estratégia de chunking é a **decisão mais impactante** num pipeline RAG
2. Aplicar e comparar **5 estratégias**: fixed-size, recursive, semantic, sentence-window, document-aware
3. Usar **Docling** para processar PDFs com layout complexo (tabelas, imagens, fórmulas) em Markdown
4. Implementar um **pipeline Naive RAG end-to-end**: ingestão → chunking → embedding → indexação → retrieval → geração
5. Estabelecer o **baseline de qualidade** para comparações nas aulas subsequentes

---

## Stack Tecnológico

| Componente | Ferramenta | Papel no Pipeline |
|---|---|---|
| Ingestão | **Docling** (IBM Research ≥2.0) | PDF/DOCX → Markdown estruturado com tabelas; OCR em escaneados (caso `Laudo.pdf`) |
| Datasets reais | **`Manual_DPCA_atualizado.pdf` + `Laudo.pdf`** (em `aula2/datasets/`) | PDFs do domínio Segurança Pública usados nos exemplos/laboratórios Docling |
| Chunking básico | **LangChain TextSplitters** | Fixed-size, Recursive, Semantic, Header-based |
| Chunking avançado | **LangChain `Document` + NLTK** (`sent_tokenize`) | Sentence-Window implementado em Python puro, alinhado com o framework do curso |
| Embeddings | **BGE-M3** servido por **Ollama** (`ollama pull bge-m3`, dim=1024) | Vetorização multilíngue multi-granularidade |
| Vector Store | **OpenSearch kNN** (Podman/Docker — provisionado na Aula 1) | Índice vetorial com busca kNN eficiente |
| Vector Store (fallback) | **FAISS** | Alternativa local quando OpenSearch indisponível |
| LLM | **Llama 3.2 3B Instruct** (padrão da Aula 1) | Geração de respostas jurídicas |
| Servidor LLM | **Ollama** (`http://localhost:11434`) | API REST compatível com OpenAI — Windows/macOS/Linux |
| Orquestração | **LangChain LCEL** | Pipeline RAG modular e composável |

---

## Fichas de Técnicas RAG — Esta Aula

### Ficha T01 — Naive RAG

| Campo | Conteúdo |
|---|---|
| **ID** | #T01 |
| **Categoria** | RAG Básico |
| **Subtítulo** | Arquitetura fundacional |
| **Descrição** | Pipeline linear: documentos → chunking → embeddings → vector store → similarity search → prompt → LLM. Ponto de partida e baseline de comparação de todo o curso. |
| **Aplicabilidades** | Chatbots de FAQ corporativo; suporte ao cliente com base estática; prototipagem de assistentes internos; Q&A sobre documentação técnica |
| **Vantagens** | Simples, rápido de implementar, custo computacional baixo, fácil de debugar |
| **Limitações** | Sem refinamento de relevância; contexto truncado em docs longos; sensível à formulação da query |
| **Lab** | LAB4 — implementação completa + LAB5 — baseline |
| **Referência** | Lewis et al. (2020). NeurIPS. |

---

## Avaliação

Ver `AVALIACAO_AULA2.md` para rubricas completas.

| Entregável | Peso | Lab |
|---|---|---|
| Pipeline Naive RAG funcional e indexando corpus | 40% | LAB1 + LAB4 |
| Comparação de chunking com análise qualitativa | 35% | LAB2 + LAB3 |
| Baseline de 5 queries registrado | 25% | LAB5 |

---

## Referências Bibliográficas (ABNT)

LEWIS, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *NeurIPS*, v. 33, 2020.

GAO, Y. et al. **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997, 2023.

IBM RESEARCH. **Docling**. Disponível em: <https://docling.readthedocs.io>. Acesso em: abr. 2026.

LANGCHAIN. **Text Splitters**. Disponível em: <https://python.langchain.com/docs>. Acesso em: abr. 2026.

OLLAMA. **Ollama — Get up and running with large language models locally**. Disponível em: <https://ollama.com/> e <https://github.com/ollama/ollama>. Acesso em: maio 2026.

BAAI. **BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings**. arXiv:2402.03216, 2024.
