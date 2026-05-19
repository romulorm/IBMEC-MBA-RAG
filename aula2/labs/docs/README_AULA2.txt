================================================================================
MBA - RAG & CAG APLICADOS A DIREITO E SEGURANÇA PÚBLICA
AULA 2: INGESTÃO AVANÇADA DE DOCUMENTOS E ESTRATÉGIAS DE CHUNKING
================================================================================

DATA DE CRIAÇÃO: 16 de abril de 2026
ATUALIZAÇÃO: maio de 2026 — migração de vLLM para Ollama (infra da Aula 1)
LINGUAGEM: Python 3.11+
AMBIENTE: Local (venv_rag da Aula 1) — Jupyter/VS Code; Colab apenas como alternativa

================================================================================
NOTEBOOKS CRIADOS (2)
================================================================================

1. LAB1_Docling_Ingestao_Avancada.ipynb
   ───────────────────────────────────────
   Objetivo: Processar PDFs jurídicos com tabelas aninhadas usando Docling
   
   Conteúdo:
   ✓ 23 células (12 Markdown + 11 Code)
   ✓ 780 linhas de código com 164 comentários (21% cobertura)
   ✓ nbformat=4, nbformat_minor=5
   
   Estrutura:
   1. Instalação de dependências (Docling, LangChain, reportlab — fallback)
   2. Teoria: Problema que Docling resolve (PyPDF2 vs Docling)
   3. Carga dos 2 PDFs reais do dataset (`Manual_DPCA_atualizado.pdf` digital + `Laudo.pdf` escaneado/OCR), com fallback ReportLab
   4. Extração com PyPDF2 (baseline ilegível)
   5. Inicialização e configuração do Docling
   6. Conversão de PDF simples (sem tabela)
   7. Conversão de PDF complexo (com tabela estruturada)
   8. Inspeção do DoclingDocument Object
   9. Comparação quantitativa (tempo, caracteres, tabelas)
   10. Pipeline Docling → LangChain Documents
   11. Configuração para OCR em PDFs escaneados
   12. Processamento em lote com tratamento de erros
   13. Exercício e referências ABNT
   
   Stack:
   • Docling (processamento estruturado de PDFs)
   • PyPDF2 (baseline para comparação)
   • LangChain (integração com Documents)
   • reportlab (apenas para fallback se PDFs do dataset não estiverem disponíveis)
   • Datasets: aula2/datasets/Manual_DPCA_atualizado.pdf + Laudo.pdf
   • pandas/matplotlib (análise e visualização)

2. LAB2_Comparacao_Chunking.ipynb
   ───────────────────────────────
   Objetivo: Comparar 5 estratégias de chunking no mesmo acórdão jurídico
   
   Conteúdo:
   ✓ 21 células (11 Markdown + 10 Code)
   ✓ 653 linhas de código com 113 comentários (17% cobertura)
   ✓ nbformat=4, nbformat_minor=5
   
   Estrutura:
   1. Instalação de dependências
   2. Texto de referência: acórdão jurídico (~2000 chars)
   3. Estratégia 1: Fixed-Size Character Chunking
   4. Estratégia 2: Recursive Character Splitting
   5. Estratégia 3: Semantic Chunking (BGE-M3 via Ollama)
   6. Estratégia 4: Sentence-Window Chunking (LangChain + NLTK — implementação Python pura)
   7. Estratégia 5: Document-Aware Header-Based
   8. Comparativo final (tabela + 4 gráficos)
   9. Visualização de fronteiras de chunk
   10. Exercício: escolha de estratégia por cenário jurídico
   11. Referências e próximos passos
   
   5 Estratégias Comparadas:
   
   | # | Estratégia        | Simples | Rápido | Semântica | Uso Ideal            |
   |---|-------------------|---------|--------|-----------|----------------------|
   | 1 | Fixed-Size        | Sim     | Sim    | Não       | Full-text, BM25      |
   | 2 | Recursive         | Sim     | Sim    | Parcial   | RAG produção         |
   | 3 | Semantic          | Não     | Não    | Sim       | Busca semântica      |
   | 4 | Sentence-Window   | Não     | Sim    | Sim       | QA com contexto      |
   | 5 | Header-Based      | Sim     | Sim    | Sim*      | Documentos jurídicos |
   
   Stack:
   • LangChain (CharacterTextSplitter, RecursiveCharacterTextSplitter,
     MarkdownHeaderTextSplitter, Document)
   • LangChain Experimental (SemanticChunker)
   • NLTK (sent_tokenize) — base do sentence-window chunking
   • Embeddings: BGE-M3 via Ollama da Aula 1 (langchain-ollama), com fallback
     HuggingFaceEmbeddings(BAAI/bge-m3) automático quando o Ollama está fora
   • pandas/numpy/matplotlib (análise comparativa)

================================================================================
REQUISITOS PRÉ-AULA (Aula 1)
================================================================================

✓ Python 3.11+ (venv_rag da Aula 1)
✓ Ollama em http://localhost:11434 com `llama3.2:3b` e `bge-m3` instalados
✓ OpenSearch em http://localhost:9200 (opcional — há fallback FAISS)
✓ Conhecimento: RAG básico, estrutura de embedding, LLMs

Dependências que serão instaladas (a maioria já vem do LAB1 da Aula 1):
• docling
• langchain + langchain-community + langchain-text-splitters
• langchain-ollama (cliente nativo do Ollama)
• langchain-openai (caminho portátil via /v1 do Ollama)
• langchain-experimental (semantic chunking)
• nltk (sent_tokenize — base do sentence-window)
• sentence-transformers (fallback de BGE-M3)
• faiss-cpu + opensearch-py
• reportlab (apenas fallback se os PDFs do dataset não existirem)
• pandas, numpy, matplotlib, seaborn, umap-learn

================================================================================
ESTILO E PEDAGOGIA
================================================================================

Proporção: 25% Teoria / 75% Prática

✓ Cada célula de código tem comentários explicando CADA linha relevante
✓ Células Markdown com diagramas ASCII detalhados
✓ Comparações lado-a-lado (PyPDF2 vs Docling, múltiplas estratégias)
✓ Visualizações: gráficos, tabelas, dados brutos
✓ Público: Python intermediário (não é iniciante)

Exemplo de comentário:
──────────────────────
```python
# Criar splitter recursivo
splitter = RecursiveCharacterTextSplitter(
    separators=separadores_juridicos,  # Quebrar em ordem: parágrafo > linha > sentença > palavra
    chunk_size=800,                     # Tamanho máximo de caracteres
    chunk_overlap=100,                  # Sobreposição entre chunks (contexto)
    length_function=len                 # Função para medir tamanho
)
```

================================================================================
DIAGRAMA DO PIPELINE
================================================================================

LAB1 - PROCESSAMENTO DOCLING:
────────────────────────────

    [PDF nativo ou escaneado]
              │
    ┌─────────▼────────────┐
    │ DocumentConverter    │
    │ PipelineOptions:     │
    │ • do_ocr=False       │
    │ • table_structure=T  │
    │ • figure_detection=F │
    └─────────┬────────────┘
              │
    ┌─────────▼──────────────────┐
    │ DoclingDocument Object     │
    │ • children (elementos)     │
    │ • tables (estruturadas)    │
    │ • metadata                 │
    └─────────┬──────────────────┘
              │
    ┌─────────▼───────────────┐
    │ Markdown Export         │
    │ (headers # ## ###)      │
    │ (tabelas formatadas)    │
    └─────────┬───────────────┘
              │
    ┌─────────▼────────────────────────────┐
    │ LangChain Documents                  │
    │ • MarkdownHeaderTextSplitter         │
    │ • RecursiveCharacterTextSplitter    │
    │ • Metadados: fonte, seção, tipo     │
    └──────────────────────────────────────┘


LAB2 - COMPARAÇÃO DE CHUNKING:
─────────────────────────────

    [TEXTO ACORDAO ÚNICO (~2000 chars)]
              │
    ┌─────────┼──────────────────┬──────────────┬──────────────┐
    │         │                  │              │              │
    ▼         ▼                  ▼              ▼              ▼
 Fixed-Size Recursive        Semantic      Sentence-Window  Header-Based
 chunk_size separadores      embeddings    SentenceWindow   Markdown
    800     jurídicos        Ollama bge-m3 (LangChain+NLTK) Headers
              │                  │              │              │
              └──────────────────┴──────────────┴──────────────┘
                           │
                ┌──────────▼───────────┐
                │ ANÁLISE COMPARATIVA  │
                │ • n_chunks           │
                │ • tamanho_médio      │
                │ • cortes_ruins       │
                │ • tempo              │
                │ • overhead           │
                └──────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
     TABELA           GRÁFICOS           RECOMENDAÇÕES
    (pandas)        (matplotlib)        (por caso)

================================================================================
MÉTRICAS ESPERADAS
================================================================================

LAB1 (Docling):
───────────────
✓ Tempo de conversão PDF simples: ~2-3s
✓ Tempo de conversão PDF complexo: ~3-5s
✓ Tabela detectada e estruturada: 100% sucesso
✓ Markdown exportado: bem formatado com headers
✓ Elementos detectados: 4-8 por página (dependendo do PDF)

LAB2 (Chunking):
────────────────
✓ Fixed-Size (800): ~8 chunks, cortes ruins ~10-20%
✓ Recursive (800): ~6 chunks, cortes ruins ~5%
✓ Semantic (85%): ~5 chunks, tempo ~2-3s
✓ Sentence-Window (w=3): ~8-10 nodes, tempo <1s
✓ Header-Based: ~4 chunks, tempo <100ms

================================================================================
INTEGRAÇÃO COM STACK OBRIGATÓRIO
================================================================================

Docling ✓ (LAB1)
  → Processamento estruturado de PDFs
  → Extração de tabelas
  → Markdown export

LangChain ✓ (LAB1 + LAB2)
  → Document primitivo
  → TextSplitters (5 tipos)
  → Integração com embeddings

NLTK ✓ (LAB2 — sentence-window)
  → sent_tokenize() para extrair sentenças
  → integrado em função Python pura que devolve langchain.schema.Document
  → substitui o SentenceWindowNodeParser do LlamaIndex sem adicionar dependência extra

BGE-M3 ✓ (LAB2/LAB4 - Embeddings + Semantic Chunking)
  → OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")
  → Fallback: HuggingFaceEmbeddings("BAAI/bge-m3")
  → Mesmo modelo (e mesmo espaço vetorial) usado em todas as estratégias

OpenSearch ✓ (LAB4 — Aula 1 já provisionou)
  → Indexação de Documents (vector field "embedding" dim=1024)
  → Vector search; combinado com BM25 nas Aulas 3-4

Ollama ✓ (servidor LLM da Aula 1, usado a partir do LAB4)
  → http://localhost:11434  •  modelo padrão: llama3.2:3b
  → Geração com Documents recuperados; portabilidade para vLLM em produção

================================================================================
PRÓXIMOS PASSOS (AULA 3)
================================================================================

1. Indexação em OpenSearch
   • Ingerir Documents de LAB1
   • Criar índices com metadata
   • Vector search + BM25 combinados

2. Retriever Hybrid
   • Implementar ensemble retriever
   • Avaliar NDCG, MAP, MRR

3. Integração com Ollama (da Aula 1) — já consolidada no LAB4 desta Aula 2
   • Endpoint local de LLM em http://localhost:11434
   • Prompting com context window (limit padrão 4096 tokens em llama3.2:3b)

4. Evaluation
   • Métricas de RAG
   • Benchmark em dataset jurídico

================================================================================
REFERÊNCIAS ABNT
================================================================================

ABNT NBR ISO/IEC 27001:2022
  Gestão de segurança da informação

ABNT NBR 10520:2023
  Informação e documentação - Apresentação de citações em documentos

ABNT NBR ISO/IEC 9126:2003
  Avaliação de qualidade de software

Documentação Técnica:
  • Docling: https://github.com/DS4SD/docling
  • LangChain: https://python.langchain.com
  • NLTK: https://www.nltk.org/api/nltk.tokenize.punkt.html
  • Ollama: https://ollama.com/ · https://github.com/ollama/ollama
  • RFC 7763: Markdown specification

================================================================================
CHECKLIST DE VALIDAÇÃO
================================================================================

[✓] LAB1_Docling_Ingestao_Avancada.ipynb
    [✓] JSON válido (nbformat=4.5)
    [✓] 23 células bem estruturadas
    [✓] Comentários em CADA linha de código relevante
    [✓] Diagramas ASCII (pipeline Docling)
    [✓] Datasets reais carregados (Manual_DPCA_atualizado.pdf + Laudo.pdf), com fallback ReportLab
    [✓] Comparação PyPDF2 vs Docling
    [✓] OCR configurado (não executado)
    [✓] Processamento em lote
    [✓] Exercício para aluno
    [✓] Referências ABNT

[✓] LAB2_Comparacao_Chunking.ipynb
    [✓] JSON válido (nbformat=4.5)
    [✓] 21 células bem estruturadas
    [✓] 5 estratégias completas
    [✓] Texto de referência único (mesma entrada)
    [✓] Análise comparativa quantitativa
    [✓] 4 gráficos matplotlib
    [✓] Tabela pandas com recomendações
    [✓] Visualização de fronteiras
    [✓] Exercício com 3 cenários jurídicos
    [✓] Referências ABNT

[✓] Compatibilidade com o ambiente da Aula 1 (Ollama + OpenSearch)
    [✓] Notebooks rodam sob o venv_rag e o kernel "MBA RAG (Python 3.11)"
    [✓] Sem dependências de GPU obrigatória (Ollama detecta hardware automaticamente)
    [✓] Sem vLLM como dependência — Ollama é o servidor LLM padrão (vLLM apenas citado para portabilidade em produção)
    [✓] Python 3.11+

================================================================================
FIM DO README
================================================================================
