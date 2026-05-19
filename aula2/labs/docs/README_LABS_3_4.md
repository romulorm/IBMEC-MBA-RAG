# Labs 3 & 4 — Análise Qualitativa e Pipeline Naive RAG Completo

## 📚 Resumo dos Notebooks Criados

### LAB 3: Análise Qualitativa de Chunks (LAB3_Analise_Qualitativa_Chunks.ipynb)
**Objetivo**: Avaliar a qualidade de chunks ANTES de indexá-los

#### Conteúdo:
- **24 células**: 2 markdown teóricos + 12 code + 10 markdown executivos
- **Corpus**: 3 documentos jurídicos variados (acórdão, lei, relatório)
- **4 Critérios de Qualidade**:
  1. **Coerência Semântica** (peso 35%): Similaridade entre sentenças vizinhas
  2. **Completude Informacional** (peso 15%): Responde perguntas completas
  3. **Independência Contextual** (peso 30%): Não precisa de chunks adjacentes
  4. **Adequação ao Modelo** (peso 20%): Distribuição de tamanhos apropriada

#### Métricas Implementadas:
- Cosine similarity entre embeddings (NLTK + sentence-transformers)
- Detecção de chunks órfãos (pronomes e referências anafóricas)
- Análise estatística de tamanhos (CV, boxplot, histograma)
- Score agregado 0-10 com recomendação automática
- Visualização UMAP dos embeddings

#### Stack:
- LangChain (splitters, documents)
- Embeddings: BGE-M3 via Ollama (`langchain-ollama`) — herdado da Aula 1; fallback HuggingFaceEmbeddings("BAAI/bge-m3")
- Scikit-learn (cosine_similarity)
- Pandas, Matplotlib, Seaborn, UMAP

#### Saídas:
- DataFrame com 4 critérios por chunk
- Gráficos: histogramas, boxplots, spider chart, UMAP
- Recomendação da melhor estratégia (FIXED/RECURSIVE/SEMANTIC)

---

### LAB 4: Pipeline Naive RAG Completo (LAB4_Naive_RAG_Pipeline_Completo.ipynb)
**Objetivo**: Implementar RAG ponta-a-ponta com stack obrigatório

#### Conteúdo:
- **12 células**: Implementação clara e funcional
- **Corpus**: 5 documentos jurídicos realistas em PDF (usando ReportLab)
- **Pipeline Completo**:
  1. **Ingestão (Docling)**: PDFs → Markdown estruturado
  2. **Chunking (LangChain)**: Respeitando artigos, incisos, alíneas
  3. **Embeddings (BGE-M3 via Ollama)**: 1024 dimensões, multilíngue; fallback HuggingFace
  4. **Indexação (OpenSearch da Aula 1 — fallback FAISS local)**: k-NN search
  5. **Retrieval**: Top-5 chunks por similaridade
  6. **Geração (Ollama llama3.2:3b)**: LLM local da Aula 1, com contexto jurídico
  7. **Citações**: [Fonte N] automáticas

#### Datasets utilizados (PDFs reais em `aula2/datasets/`):
1. **`Manual_DPCA_atualizado.pdf`** — PDF DIGITAL com estrutura hierárquica (texto extraível, sem OCR)
2. **`Laudo.pdf`** — PDF ESCANEADO (imagem de texto, exige `do_ocr=True`)
3. (Corpus inline opcional no LAB4/EXEMPLO3 — fragmentos da Lei 11.343/2006, Acórdão HC, Relatório de Inteligência — usado quando `USE_DOCLING_REAL=False`)

#### Stack Obrigatório (alinhado com Aula 1):
- **Framework**: LangChain (LCEL = LangChain Expression Language) + langchain-ollama
- **Embedding**: BGE-M3 via Ollama (`ollama pull bge-m3`, dim=1024); fallback BAAI/bge-m3 via HuggingFace
- **Vectorstore**: OpenSearch kNN (Podman/Docker da Aula 1) com fallback FAISS local
- **LLM**: `llama3.2:3b` (padrão) ou `llama3.1:8b` (opcional) — servido pelo Ollama da Aula 1 em http://localhost:11434
- **Ingestão**: Docling + ReportLab
- **Python**: 3.11+ (venv_rag da Aula 1)
- **Ambiente**: Local (Jupyter/VS Code com o kernel "MBA RAG (Python 3.11)")

#### Funcionalidades:
- RAG chain declarativa (LCEL)
- Prompt template jurídico (especialista, citations, honestidade)
- Debugging integrado (retrieval + scores + latência)
- Persistência (FAISS save/load)
- Extensibilidade (adicionar novo documento)

---

## 🎓 Estrutura Pedagógica (25% Teoria / 75% Prática)

### LAB 3 - Análise Qualitativa
| Tipo | Células | Proporção |
|------|---------|-----------|
| Markdown Teórico | 5 | 20% |
| Code Prático | 12 | 80% |
| **Total** | **17** | **100%** |

**Comentários em Português**: Cada célula de código tem 15-30 linhas de comentários explicando cada operação.

### LAB 4 - Pipeline Completo
| Tipo | Células | Proporção |
|------|---------|-----------|
| Markdown Teórico | 3 | 25% |
| Code Prático | 9 | 75% |
| **Total** | **12** | **100%** |

---

## 📋 Checklist de Funcionalidades

### LAB 3
- ✅ 3 estratégias de chunking (Fixed, Recursive, Semantic)
- ✅ 4 critérios de qualidade com métricas
- ✅ Corpus jurídico variado
- ✅ Análise estatística completa
- ✅ Visualizações (histogramas, boxplot, UMAP)
- ✅ Recomendação automática
- ✅ Análise específica para direito (artigos cortados)
- ✅ Exercício prático para aluno

### LAB 4
- ✅ Ingestão dos 2 PDFs reais do dataset (`Manual_DPCA_atualizado.pdf` + `Laudo.pdf`) via Docling, com fallback de corpus inline para execução rápida
- ✅ Ingestão com Docling
- ✅ Chunking jurídico customizado (artigos, incisos, alíneas)
- ✅ Embeddings BGE-M3 (1024d, multilíngue)
- ✅ Indexação FAISS com fallback
- ✅ Configuração do LLM via Ollama (com caminho portátil ChatOpenAI → /v1)
- ✅ Prompt template jurídico especialista
- ✅ RAG chain LCEL ponta-a-ponta
- ✅ Debugging integrado (retrieval + scores + latência)
- ✅ Persistência (save/load)
- ✅ Exercício de extensão
- ✅ Referências ABNT

---

## 🚀 Como Usar

### No Google Colab:

1. **Abrir LAB3**:
   ```
   Copia o arquivo LAB3_Analise_Qualitativa_Chunks.ipynb
   Upload para Google Colab
   Execute célula por célula
   ```

2. **Aprender avaliação de chunks**:
   - Entender 4 critérios de qualidade
   - Calcular métricas objetivas
   - Ver exemplos de chunks bons vs ruins
   - Usar UMAP para visualização

3. **Abrir LAB4**:
   ```
   Copia o arquivo LAB4_Naive_RAG_Pipeline_Completo.ipynb
   Upload para Google Colab
   Execute célula por célula
   ```

4. **Implementar RAG completo**:
   - Criar PDFs jurídicos
   - Ingerir com Docling
   - Chunking jurídico
   - Gerar embeddings
   - Indexar com FAISS
   - Confirmar Ollama da Aula 1 (`ollama list` mostra llama3.2:3b e bge-m3)
   - Executar RAG queries
   - Testar persistência

---

## 📊 Especificações Técnicas

### Metadata Colab (Obrigatória)
Ambos os notebooks incluem:
```json
{
  "colab": {
    "provenance": [],
    "toc_visible": true
  },
  "kernelspec": {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
  },
  "language_info": {
    "name": "python",
    "version": "3.11.0"
  }
}
```

### JSON Validação
- ✅ nbformat: 4
- ✅ nbformat_minor: 5
- ✅ Estrutura válida (cells, metadata, etc.)
- ✅ Sem dependência de vLLM — Ollama como servidor LLM padrão (vLLM mencionado apenas para portabilidade em produção)
- ✅ Comentários em português em 100% das células code

---

## 💡 Stack Justificado

| Componente | Escolha | Por quê |
|------------|---------|--------|
| **Chunking** | LangChain | API unificada, splitters jurídicos |
| **Embeddings** | BGE-M3 (via Ollama) | 1024d, multilíngue, mesmo servidor que o LLM |
| **Vectorstore** | OpenSearch (Aula 1) + FAISS fallback | Persistente em produção; local e rápido para dev |
| **LLM** | Ollama llama3.2:3b | Local, sem CUDA obrigatório, Windows/macOS/Linux |
| **Framework** | LangChain LCEL | Composição declarativa, rastreável |
| **Ingestão** | Docling | Entende tabelas, estrutura de PDFs |
| **Ambiente** | venv_rag (Aula 1) | Reaproveita ambiente já provisionado |

---

## 📚 Referências ABNT

LEWIS, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *arXiv preprint arXiv:2005.11401*, 2020.

GAO, Y. et al. **Retrieval-Augmented Generation for Large Language Models: A Survey**. *arXiv preprint arXiv:2312.10997*, 2023.

BAI, X. et al. **Bge M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings**. *arXiv preprint arXiv:2402.03216*, 2024.

KWIATKOWSKI, T. et al. **Natural Questions: A Benchmark for Question Answering Research**. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019.

---

## 🎯 Próximas Etapas (Labs 5+)

1. **Lab 5**: Query Expansion & Rewriting
2. **Lab 6**: Re-ranking (ColBERT, MonoT5)
3. **Lab 7**: Agentic RAG (routing, decision trees)
4. **Lab 8**: Evaluation (Hit@5, MRR, NDCG)
5. **Lab 9**: Production Deployment

---

**Última atualização**: Abril 2026  
**Status**: ✅ Pronto para uso em sala de aula  
**Carga horária recomendada**: 4 horas (2h por notebook)
