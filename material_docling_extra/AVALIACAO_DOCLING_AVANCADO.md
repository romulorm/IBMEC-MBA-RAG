# Critérios de Avaliação — Aula 5
## Docling e Ingestão Inteligente de Documentos
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 5 de 12 | **Carga:** 5h | **Peso na nota final:** 8,33% (1 de 12 aulas)

---

## Visão Geral dos Entregáveis

| # | Entregável | Peso | Lab | Ferramenta de Avaliação |
|---|---|---|---|---|
| E1 | Conversão e extração com Docling (PDF nativo + escaneado) | 25 pts | LAB1 + LAB2 | Verificação de output |
| E2 | Extração e análise de tabelas | 20 pts | LAB3 | DataFrame + texto natural |
| E3 | Pipeline de ingestão em escala | 30 pts | LAB4 | Cache + paralelismo + FAISS |
| E4 | Pipeline RAG completo integrado | 25 pts | LAB5 | Consultas jurídicas + métricas |
| **Total** | | **100 pts** | | |

---

## E1 — Conversão e Extração com Docling (25 pontos)

### Rubrica Detalhada

#### 1.1 Conversão de PDF Nativo (12 pontos)

| Indicador | Pontos |
|---|---|
| PDF de acórdão jurídico convertido com sucesso para Markdown | 4 |
| Markdown preserva estrutura: headings detectados (≥ 2) | 3 |
| Chunks gerados pelo HybridChunker (≥ 2 chunks válidos) | 3 |
| Metadados jurídicos extraídos (tribunal, artigos ou leis) | 2 |

**Verificação rápida (professor):**
```python
import json
from pathlib import Path

# Carregar chunks do aluno
with open("chunks_lab1.json") as f:
    chunks = json.load(f)

assert len(chunks) >= 2, "Menos de 2 chunks gerados"
assert all("metadados" in c for c in chunks), "Metadados ausentes"
assert any(c["metadados"].get("tribunal") for c in chunks), "Tribunal não extraído"
print(f"✅ E1.1: {len(chunks)} chunks com metadados")
```

#### 1.2 Processamento de PDF Escaneado com OCR (13 pontos)

| Indicador | Pontos |
|---|---|
| PDF escaneado criado (simulação via PIL + reportlab) | 3 |
| Função `precisa_ocr()` implementada e funcional | 3 |
| OCR executado com EasyOCR ou Tesseract | 4 |
| Extrai ≥ 3 dos 5 termos jurídicos esperados do laudo | 3 |

**Verificação rápida:**
```python
# Verificar que OCR extraiu termos críticos do laudo
termos = ['LAUDO', 'INSTITUTO', 'COCAINA', 'AMOSTRA', 'CRIMINALISTICA']
texto_ocr_upper = texto_ocr.upper()
encontrados = sum(1 for t in termos if t in texto_ocr_upper)
assert encontrados >= 3, f"OCR extraiu apenas {encontrados}/5 termos esperados"
print(f"✅ E1.2: OCR extraiu {encontrados}/5 termos")
```

---

## E2 — Extração e Análise de Tabelas (20 pontos)

#### 2.1 Extração de Tabelas com Docling (10 pontos)

| Indicador | Pontos |
|---|---|
| PDF com tabelas criado (≥ 2 tabelas) | 3 |
| Docling detecta e extrai ≥ 2 tabelas | 4 |
| Pelo menos 1 tabela convertida para DataFrame pandas | 3 |

#### 2.2 Representação Textual para RAG (10 pontos)

| Indicador | Pontos |
|---|---|
| Função `tabela_para_texto_natural()` implementada | 4 |
| Texto gerado preserva todos os dados da tabela | 3 |
| Contexto (legenda) incluído na representação textual | 3 |

**Verificação rápida:**
```python
# Verificar extração de tabelas
assert len(tabelas_extraidas) >= 2, "Menos de 2 tabelas extraídas"
dfs_ok = sum(1 for t in tabelas_extraidas if t.get("dataframe") is not None)
assert dfs_ok >= 1, "Nenhuma tabela convertida para DataFrame"

# Verificar conversão para texto
assert len(textos_tabelas) >= 1, "Conversão para texto não realizada"
assert len(textos_tabelas[0]) > 50, "Texto gerado muito curto"
print(f"✅ E2: {len(tabelas_extraidas)} tabelas, {dfs_ok} DataFrames, {len(textos_tabelas)} textos")
```

---

## E3 — Pipeline de Ingestão em Escala (30 pontos)

#### 3.1 Processamento Paralelo (12 pontos)

| Indicador | Pontos |
|---|---|
| Corpus de ≥ 4 documentos criado | 3 |
| `ThreadPoolExecutor` com ≥ 2 workers implementado | 4 |
| Comparação sequencial vs. paralelo executada | 3 |
| Speedup mensurável (ou evidência de paralelismo) | 2 |

#### 3.2 Sistema de Cache (8 pontos)

| Indicador | Pontos |
|---|---|
| Classe `CacheDocling` implementada com hash MD5 | 3 |
| Cache funcional: segunda execução usa hits | 3 |
| `hit_rate` > 0% na segunda execução | 2 |

#### 3.3 Embeddings e Indexação FAISS (10 pontos)

| Indicador | Pontos |
|---|---|
| Pipeline completo: docs → chunks → embeddings | 4 |
| Embeddings BGE-M3 com `dim=1024` | 3 |
| Índice FAISS criado e salvo em disco | 3 |

**Verificação rápida:**
```python
import faiss, numpy as np

# Verificar índice FAISS
index = faiss.read_index("/tmp/aula5_corpus.faiss")
assert index.ntotal >= 4, f"Índice com poucos vetores: {index.ntotal}"
assert index.d == 1024, f"Dimensão incorreta: {index.d} (esperado 1024)"

# Verificar cache
stats = cache.stats()
assert stats["hits"] > 0, "Cache não teve hits"
print(f"✅ E3: FAISS {index.ntotal} vetores, cache hit_rate={stats['hit_rate']}")
```

---

## E4 — Pipeline RAG Completo (25 pontos)

#### 4.1 Integração Docling + FAISS/OpenSearch + LangChain (15 pontos)

| Indicador | Pontos |
|---|---|
| Corpus indexado (FAISS ou OpenSearch) com ≥ 5 chunks | 4 |
| Função de busca `buscar_chunks()` retorna resultados relevantes | 4 |
| Pipeline RAG completo (retrieval + prompt + LLM) funcional | 4 |
| Fallback FAISS implementado quando OpenSearch ausente | 3 |

#### 4.2 Qualidade das Respostas Jurídicas (10 pontos)

| Indicador | Pontos |
|---|---|
| Pipeline responde às 3 perguntas de teste sem erros | 4 |
| Respostas citam fontes (nome do arquivo) | 3 |
| Avaliação de relevância executada e documentada | 3 |

**Verificação rápida:**
```python
# Teste de consulta
resultado = rag_juridico("Quais os requisitos para prisão preventiva?")
assert "resposta" in resultado, "Pipeline não retornou resposta"
assert len(resultado["resposta"]) > 50, "Resposta muito curta"
assert resultado["chunks_recuperados"] >= 1, "Nenhum chunk recuperado"
print(f"✅ E4: Pipeline RAG funcional ({resultado['chunks_recuperados']} chunks)")
```

---

## Pontuação Total e Conceitos

| Pontos | Conceito | Descrição |
|---|---|---|
| 90–100 | **A — Excepcional** | Pipeline completo, cache, paralelismo, RAG funcional, avaliação rigorosa |
| 75–89 | **B — Proficiente** | Todos entregáveis funcionais, pequenas lacunas em métricas ou cache |
| 60–74 | **C — Satisfatório** | Labs 1-3 funcionais, pipeline de escala parcial ou RAG básico |
| 40–59 | **D — Insuficiente** | Labs 1-2 funcionais, sem pipeline completo |
| 0–39 | **F — Reprovado** | Menos de 2 labs funcionais |

---

## Critérios de Aprovação Obrigatórios (Veto)

Os seguintes critérios são **binários** — se não atendidos, a nota máxima é 40 (D), independentemente da pontuação:

1. **Docling instalado e funcional:** `import docling` sem erro + pelo menos 1 PDF convertido com sucesso
2. **Embeddings BGE-M3 corretos:** dimensão `1024`, vetores normalizados (`np.linalg.norm ≈ 1.0`)
3. **Sem uso de Ollama:** nenhuma referência a `ollama` no código (projeto usa vLLM)

---

## Protocolo do Professor (15 minutos por aluno)

| Minuto | Ação |
|---|---|
| 0–2 | Verificar estrutura de arquivos (`aula5/labs/*.ipynb`, `aula5/datasets/`) |
| 2–5 | Executar célula de checkpoint do Lab 1 (verificação de dependências + conversão básica) |
| 5–7 | Executar célula de checkpoint do Lab 4 (paralelismo + cache + FAISS) |
| 7–10 | Executar célula de checkpoint do Lab 5 (pipeline RAG, 1 consulta real) |
| 10–13 | Fazer 2 perguntas de verificação (ver seção abaixo) |
| 13–15 | Avaliar qualidade do código: comentários, tratamento de erros, fallbacks |

---

## Perguntas de Verificação

### Nível Básico — Conceito C (60–74 pts)

1. Qual é a diferença entre usar `DocumentConverter()` padrão e usar `DocumentConverter` com `do_ocr=True`? Quando usar cada um?
2. Para que serve o `HybridChunker` do Docling? O que significa `merge_peers=True`?
3. Por que as tabelas não devem ser divididas no meio durante o chunking?

### Nível Intermediário — Conceito B (75–89 pts)

1. Como a classe `CacheDocling` detecta se um arquivo foi modificado desde a última conversão? Por que MD5 e não data de modificação?
2. Explique o fluxo completo do pipeline: Docling → HybridChunker → metadados → BGE-M3 → FAISS. Qual etapa é mais custosa computacionalmente?
3. Por que convertemos tabelas para texto natural antes de indexar? Quais as desvantagens dessa abordagem?

### Nível Avançado — Conceito A (90–100 pts)

1. Como você adaptaria o `ChunkerLegislacao` para processar o Código de Processo Penal respeitando a hierarquia: Livro → Título → Capítulo → Art. → §?
2. Em produção com 50.000 documentos, quais gargalos você antecipa no pipeline de ingestão e como os resolveria?
3. Como você integraria o pipeline de ingestão desta aula com o sistema de busca híbrida (OpenSearch kNN + BM25) da Aula 4?

---

## Feedback Padrão por Erro Comum

| Erro | Feedback Construtivo |
|---|---|
| `ModuleNotFoundError: docling` | A instalação do Docling pode falhar silenciosamente no Colab. Verifique `!pip install docling --quiet` e reinicie o runtime. |
| OCR retorna texto vazio | O EasyOCR precisa de `gpu=False` no Colab gratuito. Verifique também se o PDF realmente contém imagem (alguns PDFs protegidos bloqueiam a renderização). |
| HybridChunker gera 0 chunks | O tokenizer `BAAI/bge-m3` pode não estar disponível offline. Teste com `tokenizer="bert-base-uncased"` como fallback temporário. |
| Embeddings com `dim=768` | O BGE-M3 retorna `dim=1024`. Se você obteve 768, provavelmente carregou outro modelo. Verifique `SentenceTransformer('BAAI/bge-m3')` explicitamente. |
| Cache com `hit_rate=0%` | Verifique se está usando o mesmo objeto `CacheDocling` entre as execuções. Criar um novo objeto reinicia os contadores. |
| FAISS index com 0 vetores | `faiss.IndexFlatIP.add()` requer `np.float32`. Faça o cast: `embeddings.astype('float32')`. |

---

## Referências para Avaliação (ABNT)

AUER, Peter et al. **Docling Technical Report**. arXiv:2408.09869, 2024.

CHEN, J. et al. **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**. arXiv:2309.07597, 2024.

JOHNSON, Jeff; DOUZE, Matthijs; JÉGOU, Hervé. **Billion-scale Similarity Search with GPUs**. *IEEE Transactions on Big Data*, v. 7, n. 3, p. 535-547, 2021.

LEWIS, Patrick et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *Advances in Neural Information Processing Systems*, v. 33, p. 9459-9474, 2020.
