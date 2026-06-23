# Aula 2 — Ingestão, Chunking e Naive RAG: A Base de Tudo
## MBA em RAG & CAG Aplicados a Direito e Segurança Pública
**Carga Horária:** 5h | **Proporção:** 25% teoria / 75% prática | **Pré-requisito:** Aula 1 concluída

---

## Referências Bibliográficas (ABNT)

LEWIS, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *Advances in Neural Information Processing Systems*, v. 33, p. 9459–9474, NeurIPS, 2020.

GAO, Y. et al. **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997, 2023.

IBM RESEARCH. **Docling: An Efficient Document Conversion and Understanding Library**. Disponível em: <https://docling.readthedocs.io>. Acesso em: abr. 2026.

LANGCHAIN. **Text Splitters — LangChain Documentation**. Disponível em: <https://python.langchain.com/docs/modules/data_connection/document_transformers/>. Acesso em: abr. 2026.

OLLAMA. **Ollama — Get up and running with large language models locally**. Disponível em: <https://ollama.com/> e <https://github.com/ollama/ollama>. Acesso em: maio 2026.

OLLAMA. **OpenAI compatibility — Ollama API**. Disponível em: <https://ollama.com/blog/openai-compatibility>. Acesso em: maio 2026.

BAAI. **BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**. arXiv:2402.03216, 2024.

---

## 1. Por Que Chunking Define a Qualidade do Retrieval

### 1.1 O Chunk como Unidade Atômica de Informação

Em qualquer pipeline RAG, o **chunk** é a menor unidade de informação que o sistema pode recuperar. Quando o usuário faz uma pergunta, o sistema não busca nos documentos originais inteiros — ele busca entre os chunks indexados. Essa distinção é fundamental: **a qualidade da resposta é limitada pela qualidade dos chunks**.

```
IMPACTO DO CHUNKING NA QUALIDADE RAG

    Documento Original (10.000 palavras)
              │
              ▼
    ┌─────────────────────┐
    │   Estratégia de     │
    │      Chunking       │◄── Decisão mais impactante do pipeline
    └─────────────────────┘
              │
       ┌──────┴──────┐
       ▼             ▼
  Chunk Ruim     Chunk Bom
  (cortado no    (unidade coerente
   meio de uma    de significado)
   sentença)
       │             │
       ▼             ▼
  Embedding      Embedding
  incoerente     representativo
       │             │
       ▼             ▼
  Retrieval      Retrieval
  irrelevante    preciso
       │             │
       ▼             ▼
  Resposta       Resposta
  incorreta      correta
```

### 1.2 As Três Dimensões do Chunking

Todo problema de chunking envolve equilibrar três dimensões em tensão:

**1. Granularidade (chunk_size)**
- Chunks muito pequenos: alta precisão semântica, mas perdem contexto necessário para resposta
- Chunks muito grandes: preservam contexto, mas diluem a relevância e ultrapassam a janela de contexto do modelo de embedding
- Regra prática: o chunk deve conter *exatamente* a informação necessária para responder uma pergunta específica

**2. Continuidade (chunk_overlap)**
- Sem sobreposição: informações na fronteira entre dois chunks são perdidas
- Com sobreposição: o mesmo conteúdo é indexado múltiplas vezes, garantindo recuperação mesmo quando a informação está "dividida"
- Overhead de armazenamento aceitável para ganho de recall

**3. Coerência Semântica**
- Um chunk ideal captura uma *unidade de significado*: um artigo de lei, um parágrafo de fundamentação, uma descoberta pericial
- Chunks que cruzam fronteiras semânticas produzem embeddings que "tentam" representar dois assuntos ao mesmo tempo

### 1.3 Impacto Mensurável

Estudos empíricos em RAG (GAO et al., 2023) demonstram que a escolha de chunking impacta mais na qualidade final do que a escolha do modelo de embedding ou do algoritmo de busca. Em benchmarks internos de chatbots jurídicos, a transição de fixed-size para recursive chunking aumentou o precision@3 em 18–34% sem qualquer mudança no modelo de linguagem.

---

## 2. Fixed-Size Chunking

### 2.1 Fundamentos Teóricos

O fixed-size chunking é a estratégia mais simples: divide o texto em blocos de tamanho fixo, medido em caracteres ou tokens, com sobreposição opcional. É análogo a cortar um rolo de papel com régua — rápido, previsível, sem inteligência sobre o conteúdo.

```
FIXED-SIZE CHUNKING — DIAGRAMA

Texto original (800 chars):
╔══════════════════════════════════════════════════════╗
║ Art. 5º A prisão preventiva poderá ser decretada... ║
║ ...§ 1º Para efeitos desta lei considera-se...      ║
║ ...§ 2º A decretação da medida dependerá de...      ║
╚══════════════════════════════════════════════════════╝
                         │
          chunk_size=300, chunk_overlap=50
                         │
                         ▼
┌──────────────────────┐
│ Chunk 1 (chars 0-300)│ "Art. 5º A prisão preventiva
│                      │  poderá ser decretada como
│                      │  garantia da ordem pública..."
└──────────────────────┘
         ← overlap=50 →
              ┌──────────────────────┐
              │ Chunk 2 (250-550)    │ "...garantia da ordem pública,
              │                      │  da ordem econômica, por
              │                      │  conveniência da instrução..."
              └──────────────────────┘
                       ← overlap=50 →
                            ┌──────────────────────┐
                            │ Chunk 3 (500-800)    │ "...instrução criminal,
                            │                      │  ou para assegurar a
                            │                      │  aplicação da lei penal"
                            └──────────────────────┘
```

### 2.2 Parâmetros e Calibração

| Parâmetro | Tipo | Descrição | Valor Mínimo | Valor Típico | Valor Máximo |
|---|---|---|---|---|---|
| `chunk_size` | int | Tamanho máximo do chunk | 100 chars | 500–1000 | 4000 chars |
| `chunk_overlap` | int | Caracteres sobrepostos entre chunks adjacentes | 0 | 10–20% do size | 50% do size |
| `length_function` | callable | Como medir o tamanho: `len` (chars) ou tokenizer | `len` | `len` ou `tiktoken` | — |
| `separator` | str | Separador preferido para quebra | `""` | `"\n\n"` | qualquer regex |

**Guia de calibração para textos jurídicos:**

- **Artigos de lei curtos** (< 100 palavras cada): `chunk_size=400, overlap=50`
- **Acórdãos com parágrafos médios**: `chunk_size=800, overlap=150`
- **Relatórios longos com seções densas**: `chunk_size=1200, overlap=200`

### 2.3 Quando Usar e Quando Evitar

| Cenário | Recomendação | Justificativa |
|---|---|---|
| Textos normativos (artigos de lei) | ✅ USE | Linguagem uniforme, artigos delimitados |
| Acórdãos com seções heterogêneas | ⚠️ COM CUIDADO | Pode cortar no meio de fundamentações |
| Laudos periciais narrativos | ❌ EVITE | Perde coerência narrativa do laudo |
| Ingestão em massa (> 10.000 docs) | ✅ USE | Velocidade incomparável |
| PDFs com tabelas e figuras | ❌ EVITE | Corta dados tabulares no meio |

---

## 3. Recursive Character Text Splitting

### 3.1 Fundamentos Teóricos

O `RecursiveCharacterTextSplitter` resolve a principal fraqueza do fixed-size: respeitar a estrutura natural do texto. Ele tenta dividir usando uma hierarquia de separadores, aplicando o próximo nível apenas quando o chunk ainda excede o tamanho máximo.

```
HIERARQUIA DE SEPARADORES E DECISÃO

Nível 1: "\n\n"  (separador de parágrafos)
    │
    ├── Chunk ≤ chunk_size? → USA ESTE CHUNK ✅
    └── Chunk > chunk_size? → desce para nível 2
                │
Nível 2: "\n"   (separador de linhas)
                │
                ├── Chunk ≤ chunk_size? → USA ESTE CHUNK ✅
                └── Chunk > chunk_size? → desce para nível 3
                            │
Nível 3: " "   (separador de palavras)
                            │
                            ├── Chunk ≤ chunk_size? → USA ✅
                            └── Ainda > chunk_size? → nível 4
                                        │
Nível 4: ""    (caractere a caractere — último recurso)
                                        │
                                        └── Garante chunk ≤ size ✅


EXEMPLO COM TEXTO JURÍDICO:

Texto:
"Art. 312. A prisão preventiva poderá ser decretada.\n\n
§ 1º Para efeitos deste artigo, considera-se risco à\n
instrução criminal quando houver indício concreto de\n
que o investigado irá destruir provas.\n\n
§ 2º A decretação da medida..."

chunk_size=200 → tenta "\n\n" primeiro:
  → "Art. 312. ..." (180 chars) ✅ FIT
  → "§ 1º Para efeitos..." (145 chars) ✅ FIT
  → "§ 2º A decretação..." (continuaria...)
```

### 3.2 Separadores Customizados para Direito Brasileiro

O poder do Recursive splitter está na customização dos separadores. Para documentos jurídicos brasileiros:

```python
SEPARADORES_JURIDICOS = [
    "\n\n\n",    # Seções maiores (ementa, relatório, dispositivo)
    "\n\n",      # Parágrafos
    "\n",        # Linhas dentro de parágrafo
    ". ",        # Final de sentença (preserva artigos completos)
    "; ",        # Separador de incisos e alíneas
    ", ",        # Enumerações
    " ",         # Palavras
    "",          # Último recurso
]
```

### 3.3 Comparação Fixed-Size vs Recursive no Mesmo Texto

**Texto de entrada:** "Art. 33 da Lei 11.343/2006 — texto completo com 5 parágrafos (1.200 chars)"

| Métrica | Fixed-Size (800/100) | Recursive (800/100) |
|---|---|---|
| Chunks gerados | 3 | 3 |
| Chunks com corte no meio de sentença | 2 | 0 |
| Média de chars por chunk | 780 | 720 |
| Artigos completos capturados | 1/3 | 3/3 |
| Tempo de processamento | 0.001s | 0.003s |

---

## 4. Semantic Chunking

### 4.1 Fundamentos Teóricos

O semantic chunking abandona a ideia de tamanho fixo e usa **embeddings** para detectar mudanças semânticas no texto. Em vez de perguntar "quantos caracteres cabem aqui?", pergunta "onde o assunto muda?"

O algoritmo:
1. Divide o texto em sentenças individuais
2. Gera um embedding para cada sentença (ou para grupos de k sentenças)
3. Calcula a distância de cosseno entre sentenças adjacentes
4. Identifica *breakpoints* onde a distância ultrapassa um threshold
5. Agrupa sentenças contíguas no mesmo tema em um único chunk

```
SEMANTIC CHUNKING — DETECÇÃO DE BREAKPOINTS

Sentença 1: "O réu foi preso em flagrante..."     embedding → [0.2, 0.8, ...]
Sentença 2: "A prisão ocorreu às 23h15..."        embedding → [0.3, 0.7, ...]  dist=0.05
Sentença 3: "Testemunhas confirmaram o ato..."    embedding → [0.2, 0.9, ...]  dist=0.04

           ════ BREAKPOINT (dist=0.42 > threshold=0.20) ════

Sentença 4: "O advogado requereu habeas corpus..." embedding → [0.9, 0.1, ...]
Sentença 5: "O pedido foi instruído com..."        embedding → [0.8, 0.2, ...]  dist=0.06
Sentença 6: "O tribunal decidiu por..."            embedding → [0.7, 0.3, ...]  dist=0.08

           ════ BREAKPOINT (dist=0.38 > threshold) ════

Sentença 7: "A fundamentação baseia-se no Art..."  embedding → [0.6, 0.4, ...]

RESULTADO:
  Chunk A: Sentenças 1-3 (Fato criminal)
  Chunk B: Sentenças 4-6 (Processo judicial)
  Chunk C: Sentença 7+  (Fundamentação jurídica)
```

### 4.2 Tipos de Threshold

| `breakpoint_threshold_type` | Descrição | Quando usar |
|---|---|---|
| `"percentile"` | Quebra nos N% maiores saltos | Documentos com seções bem definidas |
| `"standard_deviation"` | Quebra quando dist > μ + k×σ | Documentos com variação semântica uniforme |
| `"interquartile"` | Usa IQR para detectar outliers semânticos | Documentos com outliers temáticos |
| `"gradient"` | Maximiza gradiente da distância | Documentos com transições abruptas |

### 4.3 Vantagens e Custos

**Vantagens:**
- Chunks coerentes semanticamente → embeddings mais representativos
- Não requer conhecimento prévio da estrutura do documento
- Funciona bem com narrativas longas (laudos, relatórios de inteligência)

**Custos:**
- 10–50x mais lento que fixed-size (precisa gerar embeddings de todas as sentenças)
- Tamanho dos chunks é variável (dificulta estimativa de uso do context window)
- Resultado varia com o modelo de embedding escolhido

---

## 5. Sentence-Window Chunking

### 5.1 Fundamentos Teóricos — A Inovação do Índice Desacoplado do Contexto

O sentence-window chunking é uma das estratégias mais sofisticadas e pouco compreendidas. Ela resolve um problema fundamental: **frases individuais são as melhores unidades para busca, mas contexto amplo é necessário para resposta**.

A estratégia desacopla **o que é indexado** de **o que é devolvido ao LLM**:

```
SENTENCE-WINDOW — DOIS NÍVEIS

ÍNDICE (o que é buscado):
  Frase 1: "A prisão preventiva requer fundamentação concreta."
  Frase 2: "A gravidade abstrata do delito não é fundamento."  ← indexada individualmente
  Frase 3: "É necessário indicar fatos concretos do caso."

JANELA DE CONTEXTO (o que o LLM recebe quando Frase 2 é recuperada):
  window_size=2 → recupera 2 sentenças antes + 2 depois

  ┌─────────────────────────────────────────────────────┐
  │ [Sentença antes-2] "O acusado foi preso em..."      │
  │ [Sentença antes-1] "A prisão preventiva requer..."  │
  │ [SENTENÇA INDEXADA] "A gravidade abstrata..."       │ ← match da busca
  │ [Sentença depois+1] "É necessário indicar fatos..." │
  │ [Sentença depois+2] "O STJ consolidou que..."       │
  └─────────────────────────────────────────────────────┘

BENEFÍCIO: A busca encontra a frase exata; o LLM recebe contexto suficiente.
```

### 5.2 Implementação com LangChain + NLTK (padrão do curso)

O sentence-window chunking foi originalmente popularizado pelo LlamaIndex (`SentenceWindowNodeParser`). Como o pipeline desta disciplina é construído **inteiramente em LangChain** (LAB1, LAB3, LAB4, LAB5 e exemplos), implementamos o mesmo conceito em **Python puro + LangChain `Document` + NLTK** para `sent_tokenize`. Isso evita uma dependência extra (`llama-index-core`) e mantém o tipo de objeto que circula no pipeline — `langchain.schema.Document` — uniforme em todas as estratégias de chunking.

**Implementação de referência (a usada no LAB2):**

```python
from langchain.schema import Document as LCDocument
from nltk.tokenize import sent_tokenize

def sentence_window_chunking(texto: str, window_size: int = 3):
    """Cada sentença vira um Document; metadata['window'] guarda o contexto vizinho."""
    sentencas = sent_tokenize(texto)
    documentos = []
    for i, sent in enumerate(sentencas):
        ini = max(0, i - window_size)
        fim = min(len(sentencas), i + window_size + 1)
        janela = " ".join(sentencas[ini:fim]).strip()
        documentos.append(LCDocument(
            page_content=sent.strip(),                       # unidade indexável
            metadata={
                "window": janela,                            # contexto para o LLM
                "original_text": sent.strip(),
                "posicao": i,
                "window_size": window_size,
            },
        ))
    return documentos
```

**Parâmetros principais:**

| Parâmetro | Descrição | Valor Padrão | Recomendado (jurídico) |
|---|---|---|---|
| `window_size` | Sentenças antes e depois do match | 3 | 2–4 |
| `metadata["window"]` | Janela de contexto guardada em cada Document | concatenação `" "` | manter |
| `metadata["original_text"]` | Sentença original (espelho de `page_content`) | igual à sentença | manter |

**Por que isso é equivalente ao LlamaIndex aqui:** o `SentenceWindowNodeParser` do LlamaIndex produz "nodes" que precisariam ser convertidos para `Document` ao serem indexados pelo OpenSearch/FAISS via LangChain. A implementação acima já entrega `langchain.schema.Document` direto, sem camada de tradução intermediária — e o `OllamaEmbeddings`/`OpenSearchVectorSearch` do LAB4 consome diretamente. Em produção, basta indexar `page_content` no campo vetorial e devolver `metadata["window"]` como contexto.

### 5.3 Quando Sentence-Window é Superior

- Documentos com sentenças densas em informação (acórdãos, pareceres)
- Queries que buscam fatos específicos (datas, nomes, valores, penas)
- Quando se deseja máxima precisão no retrieval com contexto adequado para geração

---

## 6. Document-Aware Chunking (Header-Based)

### 6.1 Fundamentos Teóricos

O document-aware chunking usa a **estrutura explícita do documento** como guia de divisão. Em vez de dividir por tamanho ou semântica, respeita os headers (H1, H2, H3) e seções do documento. Cada chunk herda os metadados hierárquicos da sua posição no documento.

```
DOCUMENT-AWARE CHUNKING — HERANÇA HIERÁRQUICA

Documento Markdown (output do Docling):

# ACÓRDÃO HC-2025.001234-SP                    ← H1
## EMENTA                                       ← H2
   Texto da ementa...

## RELATÓRIO                                    ← H2
### Das Alegações do Impetrante                 ← H3
   Texto das alegações...

### Da Manifestação do MP                       ← H3
   Texto do MP...

## FUNDAMENTAÇÃO                                ← H2
### Da Prisão Preventiva                        ← H3
   Texto sobre prisão...

## DISPOSITIVO                                  ← H2
   Texto do dispositivo...

CHUNKS GERADOS COM METADADOS:
┌────────────────────────────────────────────────────────────┐
│ Chunk 1:                                                   │
│   metadata: {H1: "ACÓRDÃO HC...", H2: "EMENTA"}           │
│   content:  "Texto da ementa..."                           │
├────────────────────────────────────────────────────────────┤
│ Chunk 2:                                                   │
│   metadata: {H1: "ACÓRDÃO...", H2: "RELATÓRIO",           │
│              H3: "Das Alegações do Impetrante"}            │
│   content:  "Texto das alegações..."                       │
├────────────────────────────────────────────────────────────┤
│ Chunk 3:                                                   │
│   metadata: {H1: "ACÓRDÃO...", H2: "FUNDAMENTAÇÃO",       │
│              H3: "Da Prisão Preventiva"}                   │
│   content:  "Texto sobre prisão..."                        │
└────────────────────────────────────────────────────────────┘

FILTRO EM PRODUÇÃO:
  query="argumentos do MP" + filter={H3: "Da Manifestação do MP"}
  → retrieval cirúrgico, zero ruído de outras seções
```

### 6.2 Dependência do Docling

O document-aware chunking só funciona bem com documentos **já estruturados em Markdown**. Para PDFs jurídicos brutos, é necessário passar pelo Docling primeiro, que detecta headers e converte para Markdown com hierarquia preservada. Sem Docling (usando PyPDF2), o PDF vira texto plano e os headers são perdidos.

### 6.3 Tabela Comparativa das 5 Estratégias

| Critério | Fixed-Size | Recursive | Semantic | Sentence-Window | Doc-Aware |
|---|---|---|---|---|---|
| **Velocidade** | ⚡⚡⚡ | ⚡⚡ | ⚡ | ⚡⚡ | ⚡⚡ |
| **Qualidade semântica** | ★★☆ | ★★★ | ★★★★ | ★★★★ | ★★★★ |
| **Preserva estrutura doc** | ✗ | Parcial | ✗ | ✗ | ✅ Total |
| **Requer embedding** | ✗ | ✗ | ✅ | ✅ | ✗ |
| **Tamanho previsível** | ✅ Fixo | ✅ Max | ✗ Variável | ✗ Variável | ✗ Variável |
| **Overhead armazenamento** | Baixo | Baixo | Médio | Alto | Médio |
| **Melhor para (jurídico)** | Normas, portarias | Acórdãos, peças | Laudos, narrativas | Pareceres densos | PDFs estruturados |
| **Implementação** | LangChain | LangChain | LangChain-Experimental | LangChain + NLTK (impl. Python pura) | LangChain |

---

## 7. Docling — Ingestão de Documentos Complexos

### 7.1 O Problema que Docling Resolve

PDFs jurídicos raramente são "texto simples". Um acórdão real contém:
- Cabeçalho com layout em colunas
- Ementa em caixa alta com formatação especial
- Tabelas de dispositivos condenatórios
- Notas de rodapé com citações
- Hierarquia de seções (EMENTA / RELATÓRIO / FUNDAMENTAÇÃO / DISPOSITIVO)
- Eventual diagrama de fluxo processual

Ferramentas simples como `PyPDF2` extraem apenas a camada de texto, ignorando toda essa estrutura. O resultado é um texto plano e desordenado onde a informação perde seu significado contextual.

```
PYPDF2 vs DOCLING — SAÍDA PARA O MESMO PDF

╔══════════════════════════════════════════════════════════════╗
║  PDF REAL: Acórdão com tabela de penas e notas de rodapé    ║
╚══════════════════════════════════════════════════════════════╝

PyPDF2 output (texto extraído sem estrutura):
─────────────────────────────────────────────
ACÓRDÃO HC 2025001234SP 7ª Câmara Criminal HABEAS CORPUS TRÁFICO
DE DROGAS PRISÃO PREVENTIVA FUNDAMENTAÇÃO INIDÔNEA EXCESSO DE
PRAZO CONSTRANGIMENTO ILEGAL CONFIGURADO ORDEM CONCEDIDA 1 A
prisão preventiva constitui medida cautelar...
[tabela de crimes: FURTO 124 118 131 ROUBO 67 71 58 extraída
como texto linear — ilegível e sem estrutura]

Docling output (Markdown estruturado):
─────────────────────────────────────────────
# ACÓRDÃO HC-2025.001234-SP
## 7ª Câmara Criminal — TJSP

### EMENTA
**HABEAS CORPUS. TRÁFICO DE DROGAS. PRISÃO PREVENTIVA.**
A prisão preventiva constitui medida cautelar...

### RELATÓRIO
Trata-se de habeas corpus impetrado...

### Tabela de Dispositivos
| Crime | Jan | Fev | Mar | Total |
|-------|-----|-----|-----|-------|
| Furto | 124 | 118 | 131 | 373   |
| Roubo | 67  | 71  | 58  | 196   |

> [1] Lewis et al. (2020). NeurIPS...
```

### 7.2 Arquitetura Interna do Docling

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCLING PIPELINE                              │
│                                                                 │
│  Input: PDF / DOCX / PPTX / HTML / Imagem                      │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────┐                                           │
│  │  1. PDF Backend  │  pypdfium2 — extrai página como bitmap   │
│  └────────┬─────────┘                                           │
│           │                                                     │
│    ▼                                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. Layout Analysis Model (DocLayNet)                    │  │
│  │     Detecta: Título / Parágrafo / Tabela / Figura /      │  │
│  │              Lista / Cabeçalho / Rodapé / Fórmula        │  │
│  └────────┬─────────────────────────────────────────────────┘  │
│           │                                                     │
│    ▼                                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. Table Structure Recovery                             │  │
│  │     Reconstrói estrutura linha×coluna das tabelas        │  │
│  │     Output: DataFrame → Markdown table                   │  │
│  └────────┬─────────────────────────────────────────────────┘  │
│           │                                                     │
│    ▼                                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  4. OCR Engine (EasyOCR / Tesseract) — opcional          │  │
│  │     Ativado quando o PDF é imagem (escaneado)            │  │
│  └────────┬─────────────────────────────────────────────────┘  │
│           │                                                     │
│    ▼                                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  5. Reading Order Detection                              │  │
│  │     Ordena os elementos na ordem de leitura correta      │  │
│  │     (PDFs em colunas são reordenados corretamente)       │  │
│  └────────┬─────────────────────────────────────────────────┘  │
│           │                                                     │
│    ▼                                                            │
│  DoclingDocument (objeto rico)                                  │
│    ├── .export_to_markdown()   → Markdown estruturado           │
│    ├── .export_to_dict()       → JSON hierárquico               │
│    ├── .tables                 → Lista de DataFrames            │
│    └── .pictures              → Imagens extraídas              │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Datasets oficiais da Aula 2 (em `aula2/datasets/`)

Os labs e exemplos que exercitam Docling usam **dois PDFs reais** do domínio Segurança Pública, escolhidos para cobrir os dois regimes do Docling:

| Arquivo | Tipo | Pipeline Docling esperado | Onde é usado |
|---|---|---|---|
| `Manual_DPCA_atualizado.pdf` | **PDF digital** (texto extraível, ~1 MB) | `do_ocr=False` — rápido; estrutura hierárquica preservada em Markdown | `LAB1` (PDF "simples"), `EXEMPLO2`, `LAB4`/`EXEMPLO3` quando `USE_DOCLING_REAL=True` |
| `Laudo.pdf` | **PDF escaneado** (imagem de texto, ~28 MB) | `do_ocr=True` — lento (30s–3min/CPU), motor EasyOCR | `LAB1` (PDF "complexo"/OCR), `EXEMPLO2` (caso OCR) |

Esses dois arquivos exercitam, juntos, **todos os caminhos** do pipeline Docling — extração estruturada (Manual_DPCA) e reconstrução por OCR (Laudo). Os notebooks têm *fallback* automático para PDFs sintéticos via ReportLab caso a pasta `datasets/` não esteja disponível.

### 7.4 Docling vs Loaders Simples — Quando a Complexidade se Justifica

| Critério | PyPDF2 | PDFMiner | Docling |
|---|---|---|---|
| Texto simples (PDF digital) | ✅ Bom | ✅ Bom | ✅ Excelente |
| Texto em colunas duplas | ⚠️ Desordenado | ⚠️ Desordenado | ✅ Correto |
| Tabelas | ❌ Texto linear | ❌ Texto linear | ✅ DataFrame |
| Figuras e imagens | ❌ Ignora | ❌ Ignora | ✅ Extrai |
| PDFs escaneados (OCR) | ❌ Vazio | ❌ Vazio | ✅ Com OCR |
| Headers / Seções | ❌ Ignora | ⚠️ Parcial | ✅ Markdown |
| Velocidade (pág/min) | ~600 | ~200 | ~80–120 |
| Consumo de memória | Baixo | Médio | Alto (~500MB modelos) |
| Instalação | `pip install PyPDF2` | `pip install pdfminer.six` | `pip install docling` + download de modelos |

**Decisão prática:**
- Use PyPDF2/PDFMiner quando: PDFs digitais simples, alto volume, sem tabelas
- Use Docling quando: PDFs com tabelas, colunas, OCR necessário, estrutura para chunking hierárquico

---

## 8. Naive RAG — Pipeline Fundacional

### 8.1 Definição e Importância como Baseline

O **Naive RAG** é o pipeline RAG mais simples possível: linear, sem reranking, sem filtragem de metadados, sem expansão de query. Ele é essencial não porque seja o melhor, mas porque é o **baseline** — o ponto de partida contra o qual todas as otimizações futuras serão medidas.

Conforme documentado por Lewis et al. (2020) e detalhado em Gao et al. (2023), o Naive RAG estabeleceu os fundamentos arquiteturais que todas as variantes posteriores refinam. Compreendê-lo profundamente é pré-requisito para entender *por que* técnicas avançadas como reranking, compressão de contexto e query expansion são necessárias.

### 8.2 Arquitetura Completa

```
╔═══════════════════════════════════════════════════════════════════════╗
║              NAIVE RAG — PIPELINE COMPLETO                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐ ║
║  │              FASE 1: INDEXAÇÃO (offline)                        │ ║
║  │                                                                 │ ║
║  │  PDF/DOCX → [Docling] → Markdown estruturado                   │ ║
║  │                │                                               │ ║
║  │                ▼                                               │ ║
║  │  [RecursiveCharacterTextSplitter]                              │ ║
║  │     chunk_size=800, overlap=150                                │ ║
║  │                │                                               │ ║
║  │                ▼                                               │ ║
║  │  LangChain Documents com metadados                             │ ║
║  │     {fonte, tipo, número, data, seção}                         │ ║
║  │                │                                               │ ║
║  │                ▼                                               │ ║
║  │  [BGE-M3 Embeddings]                                           │ ║
║  │     dim=1024, multilíngue                                      │ ║
║  │                │                                               │ ║
║  │                ▼                                               │ ║
║  │  [OpenSearch kNN Index]                                        │ ║
║  │     engine=faiss, space_type=cosinesimil                       │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐ ║
║  │              FASE 2: RETRIEVAL + GERAÇÃO (online)               │ ║
║  │                                                                 │ ║
║  │  Pergunta do usuário                                            │ ║
║  │       │                                                         │ ║
║  │       ▼                                                         │ ║
║  │  [BGE-M3] → vetor da query (dim=1024)                          │ ║
║  │       │                                                         │ ║
║  │       ▼                                                         │ ║
║  │  [OpenSearch kNN] → top-k=5 chunks mais similares              │ ║
║  │       │                                                         │ ║
║  │       ▼                                                         │ ║
║  │  Montagem do contexto (chunks + metadados formatados)          │ ║
║  │       │                                                         │ ║
║  │       ▼                                                         │ ║
║  │  [Prompt Template Jurídico]                                     │ ║
║  │   "Baseado APENAS nos documentos abaixo, responda...           │ ║
║  │    [CONTEXTO: chunk1 + chunk2 + ... + chunk5]                  │ ║
║  │    PERGUNTA: {question}"                                        │ ║
║  │       │                                                         │ ║
║  │       ▼                                                         │ ║
║  │  [Ollama Server — llama3.2:3b]                                 │ ║
║  │   POST http://localhost:11434/api/chat                          │ ║
║  │   (também compatível com /v1/chat/completions estilo OpenAI)    │ ║
║  │       │                                                         │ ║
║  │       ▼                                                         │ ║
║  │  Resposta com citação de fontes                                 │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### 8.3 Stack Tecnológico desta Aula

| Componente | Ferramenta | Versão | Papel |
|---|---|---|---|
| **Ingestão** | Docling (IBM Research) | ≥ 2.0 | PDF → Markdown estruturado |
| **Chunking** | LangChain TextSplitters | ≥ 0.3 | Divisão inteligente de texto |
| **Chunking avançado (sentence-window)** | LangChain `Document` + NLTK `sent_tokenize` | LangChain ≥ 0.3, NLTK ≥ 3.9 | Implementação Python pura — uniforme com o resto do pipeline |
| **Embeddings** | BGE-M3 servido por **Ollama** (`bge-m3`) | Ollama ≥ 0.4 | Vetorização multilíngue dim=1024 |
| **Vector Store** | OpenSearch kNN (Podman/Docker — montado na Aula 1) | 3.x | Índice vetorial para retrieval |
| **LLM** | **Llama 3.2 3B Instruct** (padrão Aula 1; opcional `llama3.1:8b`) | — | Geração de respostas |
| **Servidor LLM** | **Ollama** (`http://localhost:11434`) | ≥ 0.4 | API REST + endpoint OpenAI-compatible `/v1` |
| **Orquestração** | LangChain LCEL | ≥ 0.3 | Pipeline RAG |

### 8.4 BGE-M3 — Por Que Este Modelo de Embedding (e Como Servi-lo via Ollama)

O **BGE-M3** (BAAI General Embedding — Multilingual, Multi-Functionality, Multi-Granularity) foi escolhido por três razões alinhadas ao contexto jurídico brasileiro:

1. **Multilíngue nativo**: treinado em 100+ idiomas, incluindo português. Compreende terminologia jurídica brasileira sem fine-tuning.
2. **Dim=1024**: dimensão maior que modelos menores (384, 768) → representa nuances semânticas com maior fidelidade.
3. **Multi-granularidade**: suporta textos de uma palavra até 8.192 tokens — permite indexar desde incisos até acórdãos completos no mesmo índice.

**Como o BGE-M3 é servido nesta aula:** seguindo a infraestrutura provisionada na Aula 1, o BGE-M3 é executado **localmente via Ollama** (`ollama pull bge-m3`) — não via `sentence-transformers`/HuggingFace. Isso garante:

- Mesmo *backend* de inferência usado pelo LLM (Ollama em `http://localhost:11434`) — uma única porta a monitorar;
- Modelo quantizado (GGUF) com consumo de memória adequado a estações Windows/macOS/Linux sem GPU dedicada;
- API REST estável: `POST /api/embeddings` ou via `langchain_ollama.OllamaEmbeddings`.

```python
# Padrão da Aula 2 — embedding BGE-M3 via Ollama
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="bge-m3",                        # ollama pull bge-m3
    base_url="http://localhost:11434",
)
# embeddings.embed_query("art. 33 da Lei 11.343/2006") → list[float] dim=1024
```

> **Fallback HuggingFace (opcional):** caso o aluno não consiga executar `ollama pull bge-m3` (rede restrita, espaço em disco), o lab inclui *fallback* automático para `HuggingFaceEmbeddings(model_name="BAAI/bge-m3")`. O espaço vetorial é o mesmo modelo de fundação — só muda o servidor de inferência.

### 8.5 Limitações do Naive RAG (Motivação para Aulas Futuras)

Documentar as limitações do Naive RAG é tão importante quanto implementá-lo, pois justifica todas as técnicas avançadas do curso:

| Limitação | Impacto | Solução (aula futura) |
|---|---|---|
| Sem reranking | Top-k pode incluir chunks marginalmente relevantes | Reranking cross-encoder (Aula 3) |
| Sem filtragem por metadados | Busca em todo o corpus, incluindo documentos irrelevantes | Filtros híbridos OpenSearch (Aula 4) |
| Sem compressão de contexto | Chunks redundantes consomem tokens do LLM | Contextual Compression (Aula 3) |
| Sem expansão de query | Query mal formulada → retrieval ruim | HyDE, multi-query (Aula 7) |
| Sem verificação de fatos | LLM pode alucinar mesmo com contexto | Self-RAG, verificação (Aula 8) |
| Sem memória conversacional | Cada query é independente | CAG, memória (Aulas 10-11) |

---

## 9. Ollama como Servidor LLM Local (Infraestrutura da Aula 1)

### 9.1 Por Que Ollama nesta Disciplina

A Aula 1 montou o ambiente do curso usando **Ollama** (`ollama serve` em `http://localhost:11434`) como servidor único de inferência local — tanto para LLMs quanto para modelos de embedding. A Aula 2 prossegue exclusivamente sobre essa infraestrutura. As razões da escolha foram detalhadas no roteiro de instalação da Aula 1 e podem ser resumidas em quatro pontos:

1. **Portabilidade total** — Windows, macOS e Linux com um único instalador. Não exige drivers CUDA, GCC, ambiente Linux dedicado nem GPU NVIDIA. O Ollama detecta GPU (NVIDIA, AMD, Apple Silicon Metal) automaticamente e cai para CPU sem reconfigurar nada.
2. **API REST + compatibilidade OpenAI** — expõe simultaneamente uma API nativa (`/api/generate`, `/api/chat`, `/api/embeddings`) e um endpoint OpenAI-compatível (`/v1/chat/completions`). Permite usar `ChatOpenAI`, `OllamaLLM`, `OllamaEmbeddings` ou `curl` indistintamente.
3. **Aderência ao perfil dos alunos do MBA** — operadores de Direito e Segurança Pública executam o pipeline em notebooks corporativos, frequentemente sem GPU. O Ollama responde em 2–5 s/query em CPUs modernas com `llama3.2:3b`.
4. **Conformidade com LGPD e sigilo funcional** — todos os modelos e dados permanecem na máquina do aluno. Não há *egress* para nuvem.

> **Aviso histórico:** versões anteriores deste material usavam **vLLM** (Kwon et al., 2023) como servidor LLM. O vLLM continua sendo a melhor escolha para servir LLMs em produção sobre Linux + GPU NVIDIA, com *throughput* alto via PagedAttention e *continuous batching*. Para o ambiente didático local desta disciplina, porém, o Ollama oferece a mesma API OpenAI-compatível com instalação em minutos em qualquer SO. A migração foi feita preservando a interface (`base_url`, `ChatOpenAI`), de forma que migrar de Ollama para vLLM em produção exige apenas trocar a `base_url`.

### 9.2 Arquitetura do Servidor Ollama

```
OLLAMA — ARQUITETURA SIMPLIFICADA

  Requests chegando:           Ollama daemon (núcleo):
  ┌──────────────┐            ┌────────────────────────────┐
  │ Query 1      │───────────▶│  HTTP Server (porta 11434) │
  │ Query 2      │───────────▶│            │               │
  │ Query 3      │───────────▶│            ▼               │
  └──────────────┘            │  Model Manager             │
                              │  ┌─────────────────────┐   │
  Endpoints nativos:          │  │ llama.cpp runtime   │   │
  ┌──────────────────┐        │  │ Quantização GGUF    │   │
  │ /api/generate    │        │  │ KV Cache + offload  │   │
  │ /api/chat        │        │  │ CPU / GPU detect    │   │
  │ /api/embeddings  │        │  └─────────────────────┘   │
  │ /api/tags        │        └────────────────────────────┘
  └──────────────────┘
                              Modelos em cache local:
  Endpoints OpenAI-compat.:     ~/.ollama/models/
  ┌──────────────────┐          ├── llama3.2:3b   (2.0 GB)
  │ /v1/chat/...     │          ├── bge-m3         (~570 MB)
  │ /v1/embeddings   │          └── (lazy-load — só
  │ /v1/models       │               carrega o que foi pedido)
  └──────────────────┘
```

**Diferenças práticas em relação ao vLLM** que valem ressaltar para esta disciplina:

| Aspecto | Ollama (Aula 2) | vLLM (produção/legado) |
|---|---|---|
| Foco de uso | Desenvolvimento local, didático | Servir LLM em produção (alto QPS) |
| Backend de inferência | `llama.cpp` (GGUF quantizado) | Engine própria com PagedAttention |
| Plataformas | Windows / macOS / Linux | Linux + GPU NVIDIA |
| Modelos | `ollama pull <nome>` (registry oficial) | HuggingFace path direto |
| Concorrência | Boa para 1–4 usuários | Otimizada para 100+ usuários simultâneos |
| Tempo de instalação | ~5 min | ~30 min (CUDA + dependências) |

### 9.3 Integração com LangChain — Três Caminhos Equivalentes

A Aula 2 adota o **caminho A (`langchain_ollama`)** como padrão, com o **caminho B (`ChatOpenAI` apontando para o endpoint `/v1`)** como alternativa de portabilidade. O caminho C é apenas referência para alunos que queiram chamar a API REST diretamente.

**Caminho A — `langchain_ollama` (RECOMENDADO):**

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:3b",                # ollama pull llama3.2:3b
    base_url="http://localhost:11434",
    temperature=0.1,
    num_predict=1024,                   # equivalente a max_tokens
)
```

**Caminho B — `ChatOpenAI` via endpoint OpenAI-compatível do Ollama:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",  # endpoint OpenAI-compat do Ollama
    api_key="ollama",                       # qualquer string — Ollama ignora
    temperature=0.1,
    max_tokens=1024,
)
```

> Essa segunda forma é particularmente útil para reaproveitar código escrito para OpenAI ou vLLM: troca-se apenas a `base_url` e o `model`. **É o mesmo padrão de portabilidade explorado nas Aulas 3 a 12.**

**Caminho C — chamada HTTP direta (`requests`), apenas para fins didáticos:**

```python
import requests

r = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "O que é peculato?"}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    },
    timeout=120,
)
print(r.json()["message"]["content"])
```

### 9.4 Modelos do Curso Disponíveis via Ollama

| Papel | Modelo Ollama | Tamanho | RAM mínima | Quando usar |
|---|---|---|---|---|
| LLM **padrão** desta aula | `llama3.2:3b` | 2.0 GB | 8 GB | Geração das respostas do LAB4/LAB5 |
| LLM avançado (opcional) | `llama3.1:8b` | 4.9 GB | 16 GB | Melhor raciocínio jurídico — se hardware permitir |
| Embedding **padrão** desta aula | `bge-m3` | ~570 MB | 8 GB | Vetorização do corpus (dim=1024, multilíngue) |
| Embedding leve (fallback) | `nomic-embed-text` | 274 MB | 4 GB | Notebook detecta automaticamente e troca para 768 dims |

**Variáveis de ambiente herdadas do `.env` da Aula 1** (`~/mba-rag/.env`):

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=bge-m3
```

Todos os notebooks da Aula 2 leem essas variáveis e operam *out of the box* sobre o ambiente da Aula 1.

### 9.5 Validação Rápida Antes de Iniciar os Labs

```bash
# 1. Servidor está respondendo?
curl -s http://localhost:11434/api/tags | python -m json.tool

# 2. Modelos necessários estão instalados?
ollama list | grep -E "llama3.2|bge-m3"

# 3. Pull dos modelos (se faltar algum)
ollama pull llama3.2:3b
ollama pull bge-m3

# 4. Teste rápido de geração
ollama run llama3.2:3b "Defina prisão preventiva em 1 frase."

# 5. Teste rápido de embedding (deve imprimir 1024)
curl -s http://localhost:11434/api/embeddings \
  -d '{"model":"bge-m3","prompt":"crime de tráfico"}' | \
  python -c "import json,sys;print(len(json.load(sys.stdin)['embedding']))"
```

---

## 10. Armadilhas Comuns desta Aula (Top 5)

### ❌ Armadilha 1 — Chunk Size Muito Pequeno para Textos Jurídicos

**Sintoma:** O sistema recupera chunks corretos, mas as respostas são incompletas porque falta contexto.

**Causa:** Artigos jurídicos têm sentenças longas e densas. Com `chunk_size=200`, um artigo do CPP fica partido em 4–5 chunks, nenhum contendo a informação completa.

**Diagnóstico:**
```python
# Verifique a distribuição de tamanhos das suas sentenças antes de escolher chunk_size
import numpy as np
tamanhos = [len(s) for s in texto.split(". ")]
print(f"Média: {np.mean(tamanhos):.0f} | P90: {np.percentile(tamanhos, 90):.0f}")
# chunk_size deve ser ≥ P90 para capturar sentenças inteiras
```

**Solução:** Para direito penal brasileiro, `chunk_size ≥ 600`. Para acórdãos completos, `800–1200`.

---

### ❌ Armadilha 2 — Modelo de Embedding Diferente na Indexação e na Query

**Sintoma:** Retrieval retorna documentos completamente sem relação com a query.

**Causa:** Se você indexou com `all-MiniLM-L6-v2` (dim=384) e faz query com `BGE-M3` (dim=1024), os espaços vetoriais são incompatíveis — a distância coseno não tem significado.

**Solução:**
```python
# Defina o modelo como constante e use em AMBOS os lados
EMBEDDING_MODEL = "BAAI/bge-m3"

# Na indexação:
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = OpenSearchVectorSearch.from_documents(docs, embeddings, ...)

# Na query (mesmo objeto/modelo):
retriever = vectorstore.as_retriever(...)  # usa o mesmo embeddings
```

---

### ❌ Armadilha 3 — PyPDF2 em PDFs com Layout Complexo

**Sintoma:** Texto extraído está embaralhado, tabelas viram números aleatórios, parágrafos misturados.

**Causa:** PDFs com colunas duplas ou tabelas — o PyPDF2 extrai caractere por caractere na ordem do PDF interno, que não corresponde à ordem de leitura humana.

**Diagnóstico:**
```python
def diagnosticar_pdf(caminho):
    import PyPDF2
    with open(caminho, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        texto = reader.pages[0].extract_text() or ""
    if len(texto) < 100:
        return "PDF ESCANEADO — use Docling com OCR"
    if texto[:200].count('\n') > 20:
        return "LAYOUT COMPLEXO — use Docling"
    return "PDF SIMPLES — PyPDF2 suficiente"
```

---

### ❌ Armadilha 4 — Não Preservar Metadados de Fonte

**Sintoma:** Sistema responde corretamente mas não consegue citar qual acórdão/artigo embasou a resposta.

**Causa:** Documents criados sem metadados, ou metadados descartados durante o chunking.

**Solução:**
```python
# ERRADO: chunk sem metadados
doc = Document(page_content=texto)

# CORRETO: sempre inclua proveniência
doc = Document(
    page_content=texto,
    metadata={
        "fonte": "TJSP",
        "tipo": "acórdão",
        "numero": "HC-2025.001234-SP",
        "data": "2025-03-15",
        "secao": "FUNDAMENTAÇÃO",
        "pagina": 3
    }
)
# text_splitter.split_documents() PRESERVA os metadados automaticamente
```

---

### ❌ Armadilha 5 — OpenSearch kNN Index não Configurado Corretamente

**Sintoma:** `opensearchpy.exceptions.RequestError: search_phase_execution_exception`

**Causa:** O índice foi criado sem o mapeamento correto de campo vetorial, ou com dimensão diferente do embedding.

**Diagnóstico e Solução:**
```python
# Verifique o mapeamento do índice antes de indexar
from opensearchpy import OpenSearch
client = OpenSearch("http://localhost:9200")

# Inspeciona mapeamento existente
mapping = client.indices.get_mapping(index="juridico-v1")
dims_index = mapping["juridico-v1"]["mappings"]["properties"]["vector"]["dimension"]
dims_model = len(embeddings.embed_query("teste"))

if dims_index != dims_model:
    print(f"PROBLEMA: índice espera {dims_index} dims, modelo gera {dims_model}")
    print("Solução: deletar índice e recriar com dimensão correta")
```

---

## 11. Referências Complementares

- LANGCHAIN DOCS. *RecursiveCharacterTextSplitter*. <https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter>
- NLTK PROJECT. *nltk.tokenize.punkt — Punkt sentence tokenizer*. <https://www.nltk.org/api/nltk.tokenize.punkt.html>
- LANGCHAIN. *Document schema (langchain_core.documents.base.Document)*. <https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html>
- BAAI. *BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity*. <https://huggingface.co/BAAI/bge-m3>
- OPENSEARCH DOCS. *k-NN Search*. <https://opensearch.org/docs/latest/search-plugins/knn/index/>
- OLLAMA DOCS. *OpenAI compatibility*. <https://ollama.com/blog/openai-compatibility>
- OLLAMA DOCS. *API Reference (generate, chat, embeddings)*. <https://github.com/ollama/ollama/blob/main/docs/api.md>
- LANGCHAIN OLLAMA. *langchain-ollama integration*. <https://python.langchain.com/docs/integrations/providers/ollama/>
- DOCLING DOCS. *Getting Started*. <https://docling.readthedocs.io/en/latest/>
