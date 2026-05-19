# Instruções de Execução — Labs 3 & 4 (Aula 2)

> **IMPORTANTE — Pré-requisito:** todo o pipeline da Aula 2 roda sobre a infraestrutura provisionada na **Aula 1** (Ollama + OpenSearch + venv `venv_rag`). Antes de abrir qualquer notebook, valide o ambiente: `ollama serve` rodando, `ollama list` mostra `llama3.2:3b` e `bge-m3`, e `curl http://localhost:9200` retorna OK (se for usar OpenSearch — há fallback FAISS).

## Quick Start (5 minutos)

### 1. Preparar Ambiente

Os labs assumem o ambiente da Aula 1 (Python 3.11+, venv `venv_rag`, Ollama em `localhost:11434`, OpenSearch em `localhost:9200`). Se já executou o `LAB1_Setup_Ambiente_Completo` da Aula 1, a maioria das dependências já está instalada. Para os pacotes específicos da Aula 2, execute:

```bash
# Com o venv_rag ativo:
pip install -q langchain>=0.3 langchain-community>=0.3 langchain-text-splitters>=0.3 \
               langchain-ollama>=0.2 langchain-openai>=0.2 \
               sentence-transformers>=3.0 faiss-cpu>=1.8 opensearch-py>=2.7 \
               docling>=2.0 pandas matplotlib seaborn umap-learn python-dotenv
```

### 2. Validar a Infraestrutura da Aula 1

```bash
# Ollama OK e modelos do curso instalados?
curl -s http://localhost:11434/api/tags | python -m json.tool
ollama pull llama3.2:3b     # se faltar
ollama pull bge-m3          # se faltar

# OpenSearch OK? (opcional — há fallback FAISS)
curl -s http://localhost:9200/_cluster/health
```

### 2b. Datasets PDF da Aula 2

Os labs e exemplos que tocam Docling usam **dois PDFs reais** em `aula2/datasets/`:

| Arquivo | Tipo | Onde |
|---|---|---|
| `Manual_DPCA_atualizado.pdf` | PDF digital (texto extraível, sem OCR) | LAB1 (PDF simples) · EXEMPLO2 · LAB4/EXEMPLO3 com `USE_DOCLING_REAL=True` |
| `Laudo.pdf` | PDF escaneado (imagem de texto — exige `do_ocr=True`) | LAB1 (PDF OCR) · EXEMPLO2 · LAB4/EXEMPLO3 com `USE_DOCLING_REAL=True` |

Os notebooks têm fallback automático para PDFs sintéticos via ReportLab caso esses arquivos não estejam no `datasets/`.

### 3. LAB 3: Análise Qualitativa

```
1. Abrir LAB3_Analise_Qualitativa_Chunks.ipynb no VS Code (kernel venv_rag)
2. Executar células sequencialmente (do topo para baixo)
3. Observar:
   - 4 critérios de qualidade sendo calculados
   - Gráficos de distribuição (histograma, boxplot, UMAP)
   - Score final de 0-10 para cada estratégia
   - Recomendação automática
```

### 4. LAB 4: Pipeline RAG Completo (Ollama + BGE-M3)

```
1. Abrir LAB4_Naive_RAG_Pipeline_Completo.ipynb
2. Executar célula por célula
3. Na etapa 7 (LLM), o notebook detecta o Ollama da Aula 1 automaticamente.
   - Caminho padrão: ChatOllama(model="llama3.2:3b")
   - Caminho alternativo (USE_OPENAI_COMPAT=True):
       ChatOpenAI(base_url="http://localhost:11434/v1")
   - Se Ollama não estiver acessível: o notebook explica como subir o servidor
4. Vector store: o notebook tenta OpenSearch (Aula 1) e cai para FAISS local automaticamente
5. Testar RAG com queries jurídicas
```

---

## Detalhes de Cada Lab

### LAB 3: Análise Qualitativa de Chunks

#### O que você vai aprender
- Por que avaliar chunks qualitativamente ANTES de indexar
- 4 critérios de qualidade objetivos
- Como medir coerência semântica com embeddings (BGE-M3 via Ollama)
- Heurísticas para detectar chunks "órfãos"
- Análise estatística de distribuição de tamanhos
- Comparação de 3 estratégias de chunking

#### Fluxo Executável
```
[Célula 1] Instalar dependências (NLTK, sentence-transformers, sklearn)
    ↓
[Célula 2] Definir corpus jurídico (acórdão, lei, relatório)
    ↓
[Célula 3] Implementar 3 estratégias de chunking
    ↓
[Célula 4] Calcular coerência semântica (cosine similarity)
    ↓
[Célula 5] Detectar chunks órfãos (análise manual)
    ↓
[Célula 6] Análise de distribuição de tamanhos
    ↓
[Célula 7] Score agregado 0-10
    ↓
[Célula 8] Análise jurídica (artigos cortados)
    ↓
[Célula 9] Visualização UMAP dos embeddings
    ↓
[Célula 10] Relatório final
    ↓
[Célula 11-12] Exercício prático
```

#### Tempo de Execução
- Instalação: 3-5 minutos
- Carregamento BGE-M3 via Ollama: poucos segundos (modelo já em cache local após a Aula 1)
- Chunking: < 1 segundo
- Coerência: ~20 segundos (100 chunks)
- Cálculos estatísticos: < 5 segundos
- UMAP: ~30 segundos
- **TOTAL: ~2 horas** (incluindo leitura do código)

#### Saídas Esperadas
```
Coerência semântica
   FIXED: 0.612 (avg)
   RECURSIVE: 0.738 (melhor!)
   SEMANTIC: 0.654

Chunks órfãos
   FIXED: 15% (ruim)
   RECURSIVE: 2% (ótimo!)
   SEMANTIC: 8%

Score final (0-10)
   FIXED: 6.2
   RECURSIVE: 8.1 ← RECOMENDADO
   SEMANTIC: 7.3

Gráficos salvos em /tmp/
   - analise_tamanhos_chunks.png
   - umap_chunks.png
```

---

### LAB 4: Pipeline Naive RAG Completo

#### O que você vai aprender
- Arquitetura completa de RAG (ingestão até resposta) sobre infra Ollama local
- Ingestão inteligente com Docling
- Chunking jurídico customizado
- Embeddings BGE-M3 (1024 dimensões) via Ollama
- Indexação vetorial em OpenSearch (com fallback FAISS)
- Configuração do LLM via Ollama (`llama3.2:3b` por padrão; `llama3.1:8b` opcional)
- Prompt engineering para juristas
- RAG chain declarativa (LCEL)
- Debugging e rastreamento
- Persistência e extensibilidade

#### Fluxo Executável
```
[Célula 1] Validar Ollama da Aula 1 + instalar dependências da Aula 2
    ↓
[Célula 2] Definir corpus jurídico de teste (Lei 11.343 + Acórdão)
    ↓
[Célula 3] Chunking jurídico customizado
    ↓
[Célula 4] Inicializar embeddings BGE-M3 via Ollama (fallback HuggingFace)
    ↓
[Célula 5] Gerar embeddings dos chunks
    ↓
[Célula 6] Indexar em OpenSearch (fallback FAISS local)
    ↓
[Célula 7] Configurar LLM via Ollama (caminho A: ChatOllama; B: ChatOpenAI/v1)
    ↓
[Célula 8] Definir prompt template jurídico
    ↓
[Célula 9] Executar queries de teste
    ↓
[Célula 10] Persistir o índice (FAISS local ou índice OpenSearch já persistente)
    ↓
[Célula 11] Exercícios: estender corpus, trocar LLM, trocar embedding, medir latência
```

#### Tempo de Execução
```
Instalação:                       3-5 min (na primeira sessão)
Validação Ollama da Aula 1:       < 5 seg
Chunking:                         < 1 seg
Inicializar embeddings (Ollama):  poucos segundos (já em cache)
Gerar embeddings dos chunks:      ~5-15 seg (depende de quantos)
OpenSearch / FAISS:               < 5 seg
RAG queries:                      3-10 seg/query (Ollama em CPU moderna)

TOTAL: ~30-45 min (já com a Aula 1 montada)
```

#### Saídas Esperadas
```
Corpus chunkeado
   Total: 5-12 chunks (dependendo do corpus de exemplo)
   Tamanho médio: ~700 chars

Embeddings via Ollama (bge-m3)
   Dimensão: 1024
   Backend: ollama:bge-m3 OU huggingface:BAAI/bge-m3 (fallback)

Índice
   OpenSearch: índice "mba-aula2-naive-rag" criado em http://localhost:9200
   ou FAISS local em ~/mba-rag/aula2_artifacts/faiss_index

RAG queries de teste
   Query 1: Pena para tráfico? → Resposta com [Fonte N]
   Query 2: Prisão preventiva? → Resposta com citações
   Query 3: Estatísticas 2025? → Resposta do corpus

Índice persistido (para o LAB5)
```

---

## Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'langchain_ollama'"
**Solução**:
```bash
pip install -q langchain-ollama>=0.2
```

### Problema: Embeddings BGE-M3 via Ollama estão lentos no primeiro uso
**Causa**: cold start do `bge-m3` (carregamento do modelo em memória).
**Solução**: a partir da 2ª chamada o modelo fica em RAM até o Ollama descarregar. Para preaquecer:
```bash
curl -s http://localhost:11434/api/embeddings \
  -d '{"model":"bge-m3","prompt":"warmup"}' > /dev/null
```

### Problema: "Ollama não responde em http://localhost:11434"
**Solução (Windows)**: abra o app Ollama (ícone de llama na bandeja do sistema). Confirme com `curl http://localhost:11434/api/tags`.

**Solução (macOS)**: `ollama serve &` ou abra o app Ollama pelo Launchpad.

**Solução (Linux)**: `sudo systemctl start ollama` (ou `ollama serve &`).

### Problema: "Modelo `llama3.2:3b` não encontrado"
**Solução**:
```bash
ollama pull llama3.2:3b
# Para hardware melhor:
ollama pull llama3.1:8b   # ~5 GB, exige 16 GB RAM
```

### Problema: "Modelo `bge-m3` não encontrado"
**Solução A — usar Ollama**:
```bash
ollama pull bge-m3
```
**Solução B — fallback HuggingFace** (o notebook detecta automaticamente):
```python
# o notebook já cai para HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# se o Ollama não tiver o modelo. Sem ação manual necessária.
```

### Problema: "OpenSearch indisponível"
**Solução**: o notebook cai automaticamente para FAISS local. Para subir o OpenSearch (Podman/Docker da Aula 1):
```bash
cd ~/mba-rag/infra/opensearch
podman-compose up -d   # ou: docker compose up -d
```

### Problema: "Docling not available"
**Solução**:
```bash
pip install -q docling
# A primeira execução baixa os modelos do Docling (~500 MB)
```

### Problema: Resposta do Ollama muito lenta em CPU
**Sintoma**: cada query do LAB4/LAB5 demora 30 s+.
**Solução**:
- Trocar para `llama3.2:3b` se estiver usando `llama3.1:8b`.
- Reduzir `num_predict` para 256 nas células do LAB4/LAB5.
- Verificar se o Ollama está usando GPU: `ollama ps` (se houver GPU compatível).

---

## Extensões Sugeridas

### Após LAB 3
- Adicionar novo critério de qualidade (ex: quantidade de termos jurídicos)
- Comparar 5 estratégias de chunking (não apenas 3)
- Testar embeddings diferentes (`bge-m3` vs `nomic-embed-text` — ambos via Ollama)
- Aplicar análise em documento jurídico real (seu PDF, ingerido pelo Docling)

### Após LAB 4
- Adicionar re-ranking BGE-Reranker (Aula 3)
- Implementar query expansion (Aula 7)
- Trocar para `llama3.1:8b` e comparar qualidade vs latência
- Implementar logging estruturado (e enviar traces para a instância LangFuse da Aula 1)
- Criar API FastAPI/Flask para RAG (preview da Aula 12)

---

## Configurações de Ambiente

### Local (Python 3.11+ — recomendado, alinhado com Aula 1)
```bash
# venv_rag da Aula 1 já criado:
source ~/mba-rag/venv_rag/bin/activate   # Linux/macOS
# OU
.\venv_rag\Scripts\Activate.ps1          # Windows

# Instalar pacotes específicos da Aula 2 (se ainda não fez):
pip install langchain-ollama langchain-text-splitters docling umap-learn

# Abrir VS Code na pasta da Aula 2 e selecionar o kernel "MBA RAG (Python 3.11)"
code .
```

### Google Colab (apenas se necessário — perde a infra local)
```python
# No Colab você NÃO terá o Ollama local da Aula 1.
# Para fins de demonstração, instale o Ollama dentro do Colab:
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > /tmp/ollama.log 2>&1 &
!sleep 5 && ollama pull llama3.2:3b
!sleep 5 && ollama pull bge-m3
```

### Hardware Recomendado
- **CPU**: Intel i5+ ou AMD Ryzen 5+ (Apple Silicon Mx ótimo)
- **RAM**: 16 GB mínimo (32 GB ideal — confortável para `llama3.1:8b`)
- **GPU**: Opcional (Ollama detecta NVIDIA / AMD / Metal automaticamente)
- **Disco**: 10 GB livre para modelos do curso

---

## Próximas Aulas (Roadmap)

| Aula | Tópico | Novas Técnicas |
|-----|--------|-------|
| Aula 2 | Ingestão, Chunking e Naive RAG (esta) | #T01 Naive RAG |
| Aula 3 | Advanced/Modular RAG | #T02, #T03 (LCEL, BGE-Reranker) |
| Aula 4 | OpenSearch Hybrid + Neural Sparse | #T04, #T09 |
| Aula 5 | Avaliação RAGAS/DeepEval/LangFuse | — |
| Aula 7 | Query Enhancement (Multi-Query, RAG-Fusion) | #T06, #T12 |

---

## Referências Rápidas

### Documentação
- LangChain: https://python.langchain.com/
- LangChain Ollama: https://python.langchain.com/docs/integrations/providers/ollama/
- Ollama: https://ollama.com/ · https://github.com/ollama/ollama
- FAISS: https://github.com/facebookresearch/faiss
- OpenSearch kNN: https://opensearch.org/docs/latest/search-plugins/knn/index/
- BGE-M3 (HuggingFace): https://huggingface.co/BAAI/bge-m3
- BGE-M3 (Ollama): https://ollama.com/library/bge-m3
- Llama 3.2 (Ollama): https://ollama.com/library/llama3.2
- Docling: https://docling.readthedocs.io/

### Artigos Seminais
- Lewis et al. (2020): "RAG for Knowledge-Intensive NLP"
- Gao et al. (2023): "RAG Survey"
- Chen et al. (2024): "BGE-M3 Embeddings"

---

**Última atualização**: Maio 2026 (migração vLLM → Ollama; infra alinhada com a Aula 1)
**Status**: Pronto para execução sobre o `venv_rag` + Ollama da Aula 1
**Suporte**: instruções de troubleshooting acima e células de debugging dentro de cada lab
