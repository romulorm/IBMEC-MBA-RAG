# Critérios de Avaliação — Aula 2
## Ingestão, Chunking e Naive RAG: A Base de Tudo
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 2 de 12 | **Carga:** 5h | **Peso na nota final:** 8% (1 de 12 aulas)

---

## Visão Geral dos Entregáveis

Conforme a ementa, três entregáveis formam a avaliação desta aula:

| # | Entregável | Peso | Ferramenta de Avaliação |
|---|---|---|---|
| E1 | Pipeline Naive RAG funcional e indexando corpus | 40% | Execução em sala + inspeção de código |
| E2 | Comparação de chunking documentada com análise qualitativa | 35% | Relatório + notebook LAB2/LAB3 |
| E3 | Baseline de 5 queries registrado (LAB5) | 25% | Arquivo CSV/Excel exportado |

---

## E1 — Pipeline Naive RAG Funcional (40 pontos)

### Rubrica Detalhada

#### 1.1 Ingestão com Docling (10 pontos)

> **Datasets oficiais da Aula 2** (em `aula2/datasets/`):
> - `Manual_DPCA_atualizado.pdf` — PDF digital com estrutura hierárquica (sem OCR);
> - `Laudo.pdf` — PDF escaneado (imagem de texto, exige `do_ocr=True`).
> Esses são os arquivos esperados nos labs/exemplos que tocam Docling.

| Critério | Indicador de Evidência | Pontos |
|---|---|---|
| Docling instalado e funcional | `DocumentConverter()` executa sem erro | 2 |
| Conversão dos 2 PDFs do dataset (`Manual_DPCA_atualizado.pdf` e `Laudo.pdf`) | Outputs Markdown não vazios para ambos; o aluno habilita `do_ocr=True` no `Laudo.pdf` | 3 |
| Estrutura preservada / OCR funcional | `doc.tables`/headers em `Manual_DPCA` ou texto reconhecido no `Laudo.pdf` (ainda que parcial) | 3 |
| Metadados preservados nos Documents | `metadata` inclui `fonte` (`Manual_DPCA`, `Laudo_Pericial`), `origem` (Docling com/sem OCR) | 2 |

**Como verificar em sala:**
```python
# O professor executa este bloco no notebook do aluno (datasets oficiais)
from pathlib import Path
DATASET = Path("aula2/datasets")

# Caso 1 — PDF digital (sem OCR)
result = converter.convert(str(DATASET / "Manual_DPCA_atualizado.pdf"))
md = result.document.export_to_markdown()
assert len(md) > 1000, "Extração muito curta do Manual_DPCA — problema na conversão"
print(f"✅ Manual_DPCA: {len(md)} chars | Tabelas: {len(result.document.tables)}")

# Caso 2 — PDF escaneado (precisa OCR)
from docling.datamodel.pipeline_options import PipelineOptions
conv_ocr = DocumentConverter(pipeline_options=PipelineOptions(do_ocr=True))
result_ocr = conv_ocr.convert(str(DATASET / "Laudo.pdf"))
md_ocr = result_ocr.document.export_to_markdown()
assert len(md_ocr) > 200, "OCR não retornou texto — verificar se EasyOCR foi baixado"
print(f"✅ Laudo (OCR): {len(md_ocr)} chars")
```

**Pontuação:**
- 10 pontos: todos os critérios atendidos, PDFs com estrutura complexa processados
- 7 pontos: funcional mas sem tabelas estruturadas
- 5 pontos: funcional apenas com PDFs simples
- 2 pontos: instalado mas com erros de execução
- 0 pontos: não entregue ou não funciona

---

#### 1.2 Chunking Configurado (10 pontos)

| Critério | Indicador de Evidência | Pontos |
|---|---|---|
| `RecursiveCharacterTextSplitter` com parâmetros justificados | Comentário explicando escolha de chunk_size | 3 |
| Separadores customizados para texto jurídico | Lista inclui `". "`, `"; "` além de `"\n\n"`, `"\n"` | 2 |
| Metadados enriquecidos nos chunks | `secao`, `chunk_id`, `chunk_total` presentes | 3 |
| Distribuição de chunks documentada | Print/plot mostrando n_chunks por documento | 2 |

**Verificação rápida:**
```python
assert len(chunks) >= 10, "Corpus muito pequeno"
assert all("fonte" in c.metadata for c in chunks), "Faltam metadados"
sizes = [len(c.page_content) for c in chunks]
assert min(sizes) > 50, "Chunks muito pequenos"
assert max(sizes) < 2000, "Chunks muito grandes"
print(f"✅ {len(chunks)} chunks | avg={sum(sizes)//len(sizes)} chars")
```

---

#### 1.3 Embeddings e Indexação (10 pontos)

| Critério | Indicador de Evidência | Pontos |
|---|---|---|
| Modelo BGE-M3 servido pelo **Ollama** local da Aula 1 | `OllamaEmbeddings(model="bge-m3")` apontando para `http://localhost:11434` (ou fallback `HuggingFaceEmbeddings("BAAI/bge-m3")` com justificativa) | 4 |
| Índice criado com dimensão correta (1024 com `bge-m3`; 768 se *fallback* `nomic-embed-text` justificado) | `index.ntotal == len(chunks)` e mapping OpenSearch consistente com `len(embed_query("teste"))` | 3 |
| Índice salvo/persistido | Arquivo FAISS (`.faiss`) gravado **ou** índice OpenSearch consultável via `GET /<indice>/_count` | 3 |

#### 1.3b Geração via Ollama Local (verificação rápida do professor)

| Critério | Indicador de Evidência | Pontos |
|---|---|---|
| Cliente LLM aponta para o Ollama da Aula 1 | `ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")` ou equivalente `ChatOpenAI(base_url="http://localhost:11434/v1")` | — (binário: aprova ou reprova E1) |
| Ollama responde durante a aula | `curl -s http://localhost:11434/api/tags` retorna lista contendo `llama3.2:3b` e `bge-m3` | — (binário) |

> Os critérios 1.3b são **binários de aprovação** — se o pipeline não conseguir chamar o Ollama instalado na Aula 1, o E1 não pode receber a pontuação máxima, mesmo que as células rodem em modo simulado.

---

#### 1.4 Retrieval e Geração Funcional (10 pontos)

| Critério | Indicador de Evidência | Pontos |
|---|---|---|
| `similarity_search` retorna resultados relevantes | Top-1 chunk temático correto para query de teste | 4 |
| Prompt template com instrução anti-alucinação | Prompt inclui "citar fontes", "apenas contexto" | 3 |
| Chain RAG executa sem erro | `rag_chain.invoke(query)` retorna string | 3 |

**Teste de aceitação (o professor executa):**
```python
query_teste = "Qual é a pena para tráfico de drogas?"
docs = retriever.invoke(query_teste)
assert len(docs) >= 1, "Retrieval retornou vazio"
assert any("droga" in d.page_content.lower() or "entorpecente" in d.page_content.lower()
           for d in docs), "Retrieval não retornou documento relevante"
print(f"✅ Retrieval OK — Top doc: {docs[0].metadata.get('fonte')}")
```

---

## E2 — Comparação de Chunking com Análise Qualitativa (35 pontos)

### Rubrica Detalhada

#### 2.1 Aplicação das Estratégias (15 pontos)

| Critério | Indicador | Pontos |
|---|---|---|
| Fixed-Size implementado e executado | Chunks gerados, parâmetros documentados | 3 |
| Recursive implementado e executado | Separadores jurídicos customizados | 3 |
| Semantic implementado e executado | Modelo de embedding carregado, breakpoints identificados | 3 |
| Sentence-Window implementado | Função `sentence_window_chunking()` em LangChain + NLTK (`sent_tokenize`) retornando `langchain.schema.Document` com `metadata["window"]` | 3 |
| Document-Aware (Header-Based) implementado | MarkdownHeaderTextSplitter com metadados hierárquicos | 3 |

**Importante:** Todos os 5 devem ser aplicados no **mesmo documento** para comparação justa.

---

#### 2.2 Análise Quantitativa (10 pontos)

| Métrica Calculada | Pontos |
|---|---|
| Número de chunks por estratégia | 2 |
| Tamanho médio, mínimo e máximo por estratégia | 2 |
| Número de chunks com corte ruim (não termina em `.`) | 2 |
| Coeficiente de variação dos tamanhos | 2 |
| Gráfico comparativo (barchart ou boxplot) | 2 |

---

#### 2.3 Análise Qualitativa (10 pontos)

| Critério | Evidência Esperada | Pontos |
|---|---|---|
| Score de coerência semântica calculado | Média de cosine similarity entre sentenças adjacentes por chunk | 3 |
| Detecção de chunks órfãos | Função `detectar_chunk_orfao()` aplicada e percentual por estratégia | 3 |
| Análise de artigos jurídicos fracionados | Regex identificando artigos/incisos cortados | 2 |
| Conclusão justificada | Texto explicando qual estratégia é melhor para cada tipo de documento | 2 |

**Escala de qualidade para a conclusão:**
- **Excelente (2 pts):** Recomendação diferenciada por tipo de documento (acórdão ≠ legislação ≠ relatório) com dados numéricos suportando a escolha
- **Satisfatório (1 pt):** Recomendação genérica ("recursive é melhor") sem diferenciação por tipo
- **Insuficiente (0 pt):** Sem conclusão ou conclusão contraditória com os dados

---

## E3 — Baseline de 5 Queries (25 pontos)

### Rubrica Detalhada

#### 3.1 Execução das 5 Queries (10 pontos)

| Critério | Pontos |
|---|---|
| As 5 queries padrão foram executadas | 5 |
| Respostas completas registradas (não vazias) | 3 |
| Tempo de resposta medido para cada query | 2 |

---

#### 3.2 Avaliação nas 5 Dimensões (10 pontos)

| Dimensão | Pontos |
|---|---|
| Relevância do contexto recuperado preenchida | 2 |
| Completude da resposta preenchida | 2 |
| Fidelidade / Anti-alucinação preenchida | 2 |
| Qualidade das citações preenchida | 2 |
| Utilidade jurídica preenchida | 2 |

**Nota:** A avaliação deve ser feita pelo aluno. O professor irá revisar se as notas são consistentes com as respostas geradas. Uma nota 5 para faithfulness quando o sistema inventou dados vale 0.

---

#### 3.3 Exportação e Análise (5 pontos)

| Critério | Pontos |
|---|---|
| Arquivo CSV exportado e válido | 2 |
| Gráfico radar (spider chart) gerado | 1 |
| Análise de fraquezas com mapeamento para aulas futuras | 2 |

---

## Pontuação Total e Conceitos

| Pontos | Conceito | Descrição |
|---|---|---|
| 90–100 | **A — Excepcional** | Pipeline completo funcional, análise profunda, baseline exemplar para aulas futuras |
| 75–89 | **B — Proficiente** | Pipeline funcional com pequenos ajustes necessários, análise sólida |
| 60–74 | **C — Satisfatório** | Pipeline parcialmente funcional (≥3 etapas), comparação básica de chunking |
| 40–59 | **D — Insuficiente** | Pipeline incompleto, análise superficial, baseline parcial |
| 0–39 | **F — Reprovado** | Entregáveis não entregues ou totalmente não funcionais |

---

## Critérios de Aprovação Obrigatórios (Veto)

Os seguintes critérios são **binários** — se não atendidos, a nota máxima é C independente dos pontos:

1. **Pipeline executa sem erros:** `rag_chain.invoke("query de teste")` deve retornar resposta (mesmo que em modo simulado sem LLM)
2. **Baseline exportado:** Arquivo CSV/JSON deve existir com dados das 5 queries
3. **Sem plágio de código:** O código deve ser executado e compreendido pelo aluno — o professor pode pedir explicação de qualquer célula

---

## Execução em Sala de Aula — Protocolo do Professor

### Sequência de Verificação (15 min por aluno/grupo)

**Minuto 1–3:** Verificação do ambiente
```bash
# Verificar que o ambiente está ativo
jupyter nbconvert --to script LAB4_Naive_RAG_Pipeline_Completo.ipynb --stdout 2>/dev/null | grep -c "import"
```

**Minuto 4–7:** Teste de aceitação E1 (execução das células críticas)
- Célula de instalação
- Célula de ingestão Docling (1 PDF)
- Célula de chunking (mostrar len(chunks))
- Célula de retrieval (1 query)

**Minuto 8–11:** Inspeção do LAB2/LAB3 (comparação de chunking)
- Pedir para o aluno explicar por que escolheu determinada estratégia
- Verificar se gráfico comparativo existe
- Pedir para mostrar o score de coerência da estratégia escolhida

**Minuto 12–15:** Verificação do E3 (baseline)
- Abrir o CSV exportado
- Perguntar: "Qual query teve pior resultado? Por quê? O que mudaria?"
- Esta pergunta avalia compreensão, não memória

---

## Perguntas de Verificação (Professor usa para validar compreensão)

**Nível Básico (C):**
1. "Por que você escolheu chunk_size=[X]? O que aconteceria se fosse 100?"
2. "O que o Docling faz que o PyPDF2 não faz?"
3. "O que é overlap e por que é importante?"

**Nível Intermediário (B):**
4. "Por que o semantic chunking é mais lento que o recursive?"
5. "O que acontece se você usar um modelo de embedding diferente na indexação e na query?"
6. "Como você saberia se o retrieval está funcionando bem antes de conectar o LLM?"

**Nível Avançado (A):**
7. "Olhando seu baseline, qual das 5 dimensões você priorizaria melhorar primeiro? Por quê?"
8. "Como a escolha de chunk_size afeta o número de tokens que o LLM recebe no contexto?"
9. "Se você tivesse 100.000 acórdãos para indexar, como escalaria este pipeline?"

---

## Feedback Padrão por Erro Comum

| Erro Observado | Feedback Construtivo |
|---|---|
| `chunk_overlap=0` | "Experimente perguntar sobre o conteúdo na fronteira entre dois chunks e veja o que acontece" |
| PyPDF2 no lugar de Docling | "Compare a tabela extraída pelo PyPDF2 vs Docling — o que você perderia em produção?" |
| Sem metadados de fonte | "Como você vai citar a fonte da resposta na peça jurídica que o sistema vai ajudar a redigir?" |
| Modelo de embedding diferente na query | "Execute uma query e veja os scores de similaridade — estão entre 0 e 1 como esperado?" |
| Baseline com todas as notas iguais | "É improvável que todas as 5 queries tenham exatamente o mesmo desempenho — revise com atenção" |
| `base_url` apontando para API paga (OpenAI cloud) sem necessidade | "A Aula 1 disponibilizou o Ollama em `localhost:11434` — por que está pagando inferência? Reconfigure para usar a infra local." |
| `ollama serve` não rodando durante a defesa | "Suba o Ollama (`ollama serve` ou serviço do sistema) e refaça `ollama list` — sem o servidor, nenhuma resposta da Aula 2 é reproduzível." |

---

## Referências para Avaliação

LEWIS, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. NeurIPS, 2020. *(Baseline Naive RAG — critério E3)*

GAO, Y. et al. **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997, 2023. *(Framework de avaliação das limitações)*

LANGCHAIN. **Text Splitters**. <https://python.langchain.com/docs>. *(Referência técnica para E1 e E2)*
