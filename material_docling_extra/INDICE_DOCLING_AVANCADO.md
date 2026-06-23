# Índice — Aula 5: Docling e Ingestão Inteligente de Documentos
## Estratégias Avançadas de Processamento Documental para Sistemas RAG Jurídicos
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 5 de 12 | **Carga:** 5h | **Proporção:** 30% teoria / 70% prática  
**Pré-requisito:** Aulas 1–4 concluídas (RAG básico, chunking, embeddings, buscas híbridas)  
**Stack:** Docling ≥2.0 · EasyOCR · Apache Tika · BGE-M3 · LangChain · FAISS · OpenSearch · vLLM

---

## Estrutura de Arquivos

```
aula5/
│
├── INDICE_AULA5.md                                        ← Este arquivo
├── AVALIACAO_AULA5.md                                     ← Rubricas e critérios (professor)
│
├── teoria/
│   └── AULA5_TEORIA.md                                    ← Material teórico (10 seções)
│
├── labs/
│   ├── LAB1_Docling_Instalacao_Configuracao.ipynb         ← Instalação, conversão básica, HybridChunker
│   ├── LAB2_Docling_PDFs_Complexos_OCR.ipynb              ← OCR para laudos escaneados
│   ├── LAB3_Docling_Tabelas_Formularios.ipynb             ← Tabelas → DataFrame → texto natural
│   ├── LAB4_Pipeline_Ingestao_Escala.ipynb                ← Paralelismo + cache + BGE-M3 + FAISS
│   └── LAB5_Ingestao_RAG_Completo.ipynb                   ← Pipeline completo + LangChain + vLLM
│
├── exemplos/
│   ├── EXEMPLO1_Docling_Conversao_Minima.ipynb            ← Template mínimo de conversão
│   └── EXEMPLO2_Chunking_Comparado.ipynb                  ← RecursiveChar vs. Jurídico vs. Hybrid
│
└── datasets/
    └── corpus_docling_aula5.json                          ← 10 docs jurídicos + 8 perguntas de teste
```

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|---|---|---|---|---|
| **1. Revisão + Motivação** | 15 min | Teoria | Por que extração de documentos importa para RAG jurídico | `teoria/AULA5_TEORIA.md §1` |
| **2. Arquitetura Docling** | 25 min | Teoria | Pipeline DocLayNet, backends, exportadores | `teoria/AULA5_TEORIA.md §2` |
| **3. LAB 1 — Instalação e Conversão** | 45 min | Prática | Instalar Docling, converter PDF nativo, explorar DoclingDocument, HybridChunker | `labs/LAB1_*.ipynb` |
| **4. OCR — Teoria** | 15 min | Teoria | EasyOCR vs. Tesseract, detecção automática de OCR, pipeline decisão | `teoria/AULA5_TEORIA.md §3` |
| **5. LAB 2 — PDFs Escaneados** | 50 min | Prática | Criar laudo escaneado simulado, configurar OCR, `precisa_ocr()`, avaliação qualidade | `labs/LAB2_*.ipynb` |
| **6. Tabelas — Teoria** | 10 min | Teoria | Desafios de tabelas jurídicas, chunk que não corta tabelas | `teoria/AULA5_TEORIA.md §4` |
| **7. LAB 3 — Tabelas e Formulários** | 50 min | Prática | Relatório com tabelas, extração, DataFrame, texto natural para RAG | `labs/LAB3_*.ipynb` |
| **8. Pipeline em Escala — Teoria** | 15 min | Teoria | ThreadPool, cache MD5, estratégia de ingestão incremental | `teoria/AULA5_TEORIA.md §6` |
| **9. LAB 4 — Ingestão em Escala** | 55 min | Prática | Corpus 5 docs, paralelismo, cache, BGE-M3 batch, FAISS | `labs/LAB4_*.ipynb` |
| **10. LAB 5 — RAG Completo** | 60 min | Prática | OpenSearch ou FAISS, LangChain, vLLM, avaliação de ingestão | `labs/LAB5_*.ipynb` |

---

## Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Configurar** o Docling 2.x no Google Colab com suporte a OCR (EasyOCR) e extração de tabelas
2. **Detectar automaticamente** se um documento PDF requer OCR e selecionar o pipeline adequado
3. **Extrair tabelas** de PDFs complexos e convertê-las para DataFrame pandas e para texto natural adequado ao RAG
4. **Implementar pipeline de ingestão paralelo** usando `ThreadPoolExecutor` com cache MD5 em disco
5. **Construir um pipeline RAG completo** integrando Docling, HybridChunker, BGE-M3, FAISS/OpenSearch e LangChain
6. **Avaliar qualitativamente** a eficácia da ingestão usando consultas jurídicas reais

---

## Stack Tecnológico

| Componente | Ferramenta | Papel no Pipeline |
|---|---|---|
| Motor de extração | **Docling ≥2.0** (IBM Research) | Conversão de PDFs nativos e escaneados para estrutura semântica |
| OCR | **EasyOCR** (fallback: **Tesseract**) | Reconhecimento de texto em imagens/documentos escaneados |
| Chunking | **HybridChunker** (nativo Docling) | Divisão semântica respeitando hierarquia do documento |
| Chunking (legislação) | **RecursiveCharacterTextSplitter** | Separação por Art., §, Inciso para legislação |
| Embeddings | **BGE-M3** (BAAI, dim=1024) | Vetorização multilíngue de chunks |
| Vector Store | **OpenSearch kNN** | Produção: busca híbrida kNN + BM25 |
| Vector Store (fallback) | **FAISS IndexFlatIP** | Desenvolvimento/Colab: busca vetorial local |
| LLM | **Llama 3.1 8B Instruct** | Geração de respostas no pipeline RAG |
| Servidor LLM | **vLLM** | API OpenAI-compatible em localhost:8000 |
| Orquestração | **LangChain LCEL** | Pipeline `retriever | prompt | llm | parser` |
| Detecção de PDF | **pdfplumber** | Heurística para identificar PDFs que precisam de OCR |

---

## Fichas de Técnicas RAG — Esta Aula

### Ficha T11 — Docling Pipeline

| Campo | Conteúdo |
|---|---|
| **ID** | #T11 |
| **Categoria** | Ingestão Inteligente |
| **Subtítulo** | Extração semântica de documentos com IBM Docling |
| **Descrição** | Docling converte documentos (PDF, DOCX, HTML) para um modelo de documento unificado (`DoclingDocument`) que preserva a estrutura hierárquica (headings, parágrafos, tabelas, figuras). Suporta OCR integrado (EasyOCR/Tesseract) para documentos escaneados e exporta para Markdown, JSON e outros formatos. Inclui `HybridChunker` nativo para divisão semântica. |
| **Aplicabilidades** | Acórdãos de tribunais, laudos periciais, legislação, relatórios de inteligência, formulários de BO, documentos administrativos |
| **Vantagens** | Preserva layout e hierarquia; suporte nativo a OCR; chunking semântico integrado; pipeline configurável |
| **Limitações** | Mais lento que parsers simples; requer GPU para OCR rápido; documentos protegidos podem falhar |
| **Lab** | LAB1 (básico) + LAB2 (OCR) + LAB3 (tabelas) + LAB4 (escala) + LAB5 (integração) |
| **Referência** | AUER et al. arXiv:2408.09869, 2024. |

---

## Avaliação

Ver `AVALIACAO_AULA5.md` para rubricas completas.

| Entregável | Peso | Lab |
|---|---|---|
| Conversão Docling (nativo + OCR) | 25 pts | LAB1 + LAB2 |
| Extração e análise de tabelas | 20 pts | LAB3 |
| Pipeline de ingestão em escala (paralelismo + cache + FAISS) | 30 pts | LAB4 |
| Pipeline RAG completo integrado | 25 pts | LAB5 |

---

## Referências Bibliográficas (ABNT)

AUER, Peter et al. **Docling Technical Report**. arXiv:2408.09869, 2024. IBM Research Europe. Disponível em: <https://arxiv.org/abs/2408.09869>. Acesso em: abr. 2026.

BAEK, J. et al. **Character Region Awareness for Text Detection (CRAFT)**. IEEE CVPR, 2019. (EasyOCR engine). Disponível em: <https://github.com/JaidedAI/EasyOCR>. Acesso em: abr. 2026.

CHEN, J. et al. **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**. arXiv:2309.07597, 2024.

IBM RESEARCH. **Docling Documentation**. Disponível em: <https://ds4sd.github.io/docling/>. Acesso em: abr. 2026.

JOHNSON, Jeff; DOUZE, Matthijs; JÉGOU, Hervé. **Billion-scale Similarity Search with GPUs**. *IEEE Transactions on Big Data*, v. 7, n. 3, p. 535-547, 2021.

LANGCHAIN. **Docling Document Loader**. Disponível em: <https://python.langchain.com/docs/integrations/document_loaders/docling/>. Acesso em: abr. 2026.

LEWIS, Patrick et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *Advances in Neural Information Processing Systems*, v. 33, p. 9459-9474, 2020.

SMITH, Ray. **An Overview of the Tesseract OCR Engine**. In: *Ninth International Conference on Document Analysis and Recognition (ICDAR)*. IEEE, 2007.

BRASIL. Lei nº 13.709, de 14 de agosto de 2018. **Lei Geral de Proteção de Dados Pessoais (LGPD)**. Brasília, DF: Presidência da República, 2018.

BRASIL. Lei nº 11.343, de 23 de agosto de 2006. **Lei de Drogas**. Brasília, DF: Presidência da República, 2006.
