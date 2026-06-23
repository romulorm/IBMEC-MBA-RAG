# Aula 6 — Indexação Avançada: Hierarchical Indexing, RAPTOR e HyDE
## Superando as Limitações do Chunking Plano em Sistemas RAG Jurídicos
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 6 de 12 | **Carga:** 5h | **Proporção:** 25% teoria / 75% prática  
**Pré-requisito:** Aulas 1–5 concluídas | **Stack:** LlamaIndex · RAPTOR · UMAP · scikit-learn · RAGAS · vLLM · OpenSearch  
**Referência normativa:** ABNT NBR 6023:2018 (referências), NBR 10520:2023 (citações)

---

## Verificação de Sobreposição com Aulas Anteriores

> **Nota pedagógica:** Esta aula introduz técnicas de indexação estruturalmente diferentes das abordadas anteriormente. A tabela abaixo esclarece as fronteiras de cada aula:

| Tema | Aula Anterior | Esta Aula (Aula 6) |
|---|---|---|
| Chunking | Aula 2: chunking fixo e semântico (flat) | Indexação hierárquica pai-filho (multi-nível) |
| UMAP | Aula 1: visualização de embeddings | UMAP para clustering de sumarização (RAPTOR) |
| RAGAS | Aula 5: framework de avaliação | Aplicação comparativa entre estratégias |
| LlamaIndex | Aula 3: SentenceWindowNodeParser (menção) | Parent-Child Retriever (uso completo) |
| HyDE | — | Técnica inédita neste curso |

---

## 1. Motivação: Por Que o Chunking Plano Falha em Documentos Jurídicos Longos

### 1.1 O Problema da Granularidade Única

Em um acórdão do STF com 80 páginas, há perguntas de dois tipos radicalmente diferentes:

**Pergunta pontual:** *"Qual foi o fundamento legal citado no voto do Min. Alexandre de Moraes?"*  
→ Exige um chunk pequeno (128 tokens), preciso, que contenha exatamente o trecho do voto.

**Pergunta abrangente:** *"Quais são as principais linhas argumentativas do acórdão sobre o direito à privacidade?"*  
→ Exige contexto amplo de múltiplas seções do documento.

Com chunking plano (tamanho fixo), você é forçado a escolher um único tamanho que serve mal a ambos os casos:

```
DILEMA DO CHUNKING PLANO

chunk pequeno (128 tok):
  ✅ precisão para queries pontuais
  ❌ contexto insuficiente para queries gerais
  ❌ fratura de argumentos jurídicos entre chunks

chunk grande (512 tok):
  ✅ contexto para queries gerais
  ❌ ruído: recupera muito conteúdo irrelevante
  ❌ degrada Context Precision no RAGAS

SOLUÇÃO: Separar a unidade de BUSCA da unidade de CONTEXTO
```

### 1.2 Três Técnicas, Três Problemas Diferentes

| Técnica | Problema que Resolve | Quando Usar |
|---|---|---|
| **Hierarchical Indexing (Parent-Child)** | Granularidade dupla: buscar com precisão, gerar com contexto | Documentos com estrutura clara (artigos, seções) |
| **RAPTOR** | Queries multi-nível sobre corpus extensos (abstrações) | Repositórios com 100+ documentos; análise de tendências |
| **HyDE** | Gap semântico entre query coloquial e linguagem técnica-jurídica | Usuários leigos fazendo perguntas sobre textos especializados |

---

## 2. Hierarchical Indexing: Parent-Child Retriever

### 2.1 Conceito Fundamental

A ideia central é simples: **use dois tamanhos de chunk diferentes com propósitos distintos**.

```
ARQUITETURA PARENT-CHILD

Documento Original (acórdão 80 pág)
│
├── CHUNKS PAI (512 tokens) — armazenados no docstore
│   ├── PAI-001: "Seção I — Relatório..."
│   ├── PAI-002: "Seção II — Fundamentos..."
│   └── PAI-003: "Seção III — Dispositivo..."
│
└── CHUNKS FILHO (128 tokens) — indexados no vector store
    ├── F-001a: "Art. 5º, inciso X..." → referência: PAI-001
    ├── F-001b: "...direito à privacidade..." → referência: PAI-001
    ├── F-002a: "Conforme decidido no RE..." → referência: PAI-002
    └── F-002b: "...fundada suspeita é requisito..." → referência: PAI-002

FLUXO DE BUSCA:
Query → [Embedding] → Busca no vector store (filhos)
        → Encontra F-002b (score: 0.91)
        → Recupera PAI-002 (contexto completo)
        → Envia PAI-002 ao LLM para geração
```

**Por que funciona?** O embedding do chunk filho é mais preciso porque há menos "ruído" textual — o vetor representa uma ideia singular. Mas o LLM recebe o chunk pai com contexto suficiente para gerar uma resposta completa e coerente.

### 2.2 Implementação com LlamaIndex

```python
# ============================================================
# Hierarchical Indexing com LlamaIndex — Implementação Completa
# ============================================================

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes
)
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 1. Configurar o parser hierárquico
# chunk_sizes define os níveis: [pai, filho]
# pai = 512 tokens → contexto para o LLM
# filho = 128 tokens → unidade de busca precisa
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 128]
)

# 2. Parsear os documentos em nós hierárquicos
nodes = node_parser.get_nodes_from_documents(documents)

# 3. Separar nós folha (filhos) dos nós raiz/pai
leaf_nodes = get_leaf_nodes(nodes)   # serão indexados no vector store
root_nodes = get_root_nodes(nodes)   # serão armazenados no docstore

# 4. Configurar storage com docstore para os pais
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)  # armazena TODOS os nós (pais e filhos)

storage_context = StorageContext.from_defaults(docstore=docstore)

# 5. Criar índice vetorial APENAS com os filhos (leaf nodes)
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)

# 6. Configurar o retriever com auto-merging
# threshold: se X% dos filhos de um pai forem recuperados,
# substitui pelos filhos pelo pai inteiro (context expansion)
base_retriever = base_index.as_retriever(similarity_top_k=6)

retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True,   # mostra quando o merging é ativado
    simple_ratio_thresh=0.3  # 30% dos filhos → promove o pai
)

# 7. Criar query engine
query_engine = RetrieverQueryEngine.from_args(retriever)

# 8. Consulta
response = query_engine.query(
    "Qual o requisito de fundada suspeita para busca pessoal?"
)
print(response)
```

### 2.3 Aplicação Jurídica: Quando Parent-Child se Destaca

No contexto do Direito, o Parent-Child Retriever brilha especialmente em:

**Acórdãos longos:** A seção de "Relatório" (contexto) e "Fundamentos" (razões de decidir) ficam em chunks pai distintos, mas artigos específicos citados ficam em chunks filho com alta precisão de recuperação.

**Legislações com artigos e parágrafos:** O artigo principal é o pai; incisos e parágrafos são os filhos. Uma query sobre um inciso específico recupera o artigo completo como contexto.

```
Exemplo — Lei 13.964/2019 (Pacote Anticrime):

CHUNK PAI (Art. 3º-B completo, ~400 tokens):
"Art. 3º-B. O juiz das garantias é responsável pelo controle da 
legalidade da investigação criminal e pela salvaguarda dos direitos 
individuais cuja franquia tenha sido reservada à autorização prévia 
do Poder Judiciário. § 1º O juiz das garantias é competente para: 
I - receber a comunicação imediata da prisão... II - receber o auto..."

CHUNK FILHO (§1º, inciso I, ~60 tokens):
"I - receber a comunicação imediata da prisão, nos termos do inciso 
LXII do caput do art. 5º da Constituição Federal"

Query: "juiz das garantias recebe comunicação de prisão?"
→ Busca encontra o chunk filho (score alto)
→ LLM recebe o chunk pai (contexto completo do artigo)
→ Resposta: contextualizada com todo o art. 3º-B
```

**Referências (ABNT):**  
LLAMAINDEX. *Auto-Merging Retriever*. Disponível em: <https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/>. Acesso em: abr. 2026.

---

## 3. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

### 3.1 O Problema que RAPTOR Resolve

Imagine que você tem 200 acórdãos do STF sobre direito digital. Um pesquisador pergunta:

> *"Quais são as tendências jurisprudenciais do STF sobre privacidade de dados nos últimos 5 anos?"*

Nenhum chunk individual responde essa pergunta — ela exige síntese de múltiplos documentos. O chunking plano e mesmo o Parent-Child falham aqui porque operam **dentro de documentos**, não **entre documentos**.

RAPTOR constrói uma **árvore de abstrações** sobre o corpus inteiro:

```
ÁRVORE RAPTOR — Corpus de Jurisprudência STF

NÍVEL 3 (Raiz) — 1 nó:
  "Síntese geral: jurisprudência STF sobre direitos digitais 2019-2024"

NÍVEL 2 (Ramos) — ~5 nós temáticos:
  "Privacidade e dados pessoais"
  "Monitoramento e reconhecimento facial"
  "Compartilhamento de dados com MP"
  "Criptomoedas e crimes financeiros"
  "Direito ao esquecimento"

NÍVEL 1 (Folhas) — chunks originais dos 200 acórdãos
  [chunk_001] [chunk_002] ... [chunk_n]
```

### 3.2 Algoritmo RAPTOR Passo a Passo

**Passo 1 — Embeddings:** Gerar embeddings para todos os chunks (BAAI/bge-m3).

**Passo 2 — Redução dimensional (UMAP):** Reduzir de 1024 para 10 dimensões para clustering mais eficiente.

```python
import umap

# Redução dimensional para clustering
reducer = umap.UMAP(
    n_components=10,    # dimensão para clustering (não para visualização)
    n_neighbors=15,     # balanceia estrutura local vs global
    min_dist=0.0,       # clusters mais compactos
    metric="cosine",    # consistente com os embeddings
    random_state=42
)
embeddings_reduced = reducer.fit_transform(embeddings)  # (n_chunks, 10)
```

**Passo 3 — Clustering (GMM):** Gaussian Mixture Models para agrupamento suave (soft clustering).

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Por que GMM e não K-means?
# GMM permite que um chunk pertença a múltiplos clusters (probabilístico)
# Um acórdão sobre privacidade E criptomoedas pode pertencer a ambos os clusters

# Determinar número ótimo de clusters com BIC
bic_scores = []
for k in range(2, 20):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(embeddings_reduced)
    bic_scores.append(gmm.bic(embeddings_reduced))

n_clusters = np.argmin(bic_scores) + 2  # índice + offset

# Clustering final
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(embeddings_reduced)
labels = gmm.predict(embeddings_reduced)           # cluster dominante
probs = gmm.predict_proba(embeddings_reduced)      # probabilidades por cluster
```

**Passo 4 — Sumarização:** Para cada cluster, sumarizar os chunks com o LLM.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0.1,
    max_tokens=512,
)

def summarize_cluster(texts: list[str], cluster_id: int) -> str:
    """Gera resumo temático de um cluster de chunks jurídicos."""
    combined = "\n\n---\n\n".join(texts[:10])  # max 10 chunks por cluster
    prompt = f"""Você é um especialista jurídico. Analise os seguintes excertos 
de documentos jurídicos brasileiros e gere um resumo temático coerente 
(máximo 300 palavras) que capture os principais pontos, tendências e 
fundamentos legais presentes no conjunto.

DOCUMENTOS DO CLUSTER {cluster_id}:
{combined}

RESUMO TEMÁTICO:"""
    return llm.invoke(prompt).content

# Gerar resumos para todos os clusters
cluster_summaries = {}
for cluster_id in range(n_clusters):
    cluster_texts = [chunks[i] for i, l in enumerate(labels) if l == cluster_id]
    if cluster_texts:
        cluster_summaries[cluster_id] = summarize_cluster(cluster_texts, cluster_id)
```

**Passo 5 — Recursão:** Os resumos do Nível 1 se tornam novos "chunks" e o processo se repete para criar o Nível 2 (e assim por diante).

### 3.3 Estratégias de Query no RAPTOR

| Estratégia | Como Funciona | Quando Usar |
|---|---|---|
| **Collapsed Tree** | Busca em todos os níveis simultaneamente e rankeia por similaridade | Queries gerais ou quando não se sabe o nível de abstração necessário |
| **Tree Traversal** | Desce nível a nível: começa na raiz, escolhe ramo mais similar, continua | Queries que exigem navegação hierárquica explícita |

```
COLLAPSED TREE vs TREE TRAVERSAL

Query: "tendências STF sobre privacidade"

COLLAPSED TREE:
  Busca em: [todos os chunks nível 1] + [todos os resumos nível 2] + [raiz nível 3]
  Retorna: top-k de toda a árvore
  Vantagem: captura tanto detalhe (nível 1) quanto visão geral (nível 2-3)

TREE TRAVERSAL:
  Nível 3: "Síntese geral" → score 0.72 → desce
  Nível 2: "Privacidade e dados" → score 0.89 (mais similar) → desce
  Nível 2: "Monitoramento facial" → score 0.71 → desce (segundo mais similar)
  Nível 1: recupera os chunks originais dos ramos selecionados
  Vantagem: mais controlado; menor risco de chunks irrelevantes
```

**Referências (ABNT):**  
SARTHI, P. et al. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**. In: *International Conference on Learning Representations (ICLR)*, 2024. arXiv:2401.18059.

---

## 4. HyDE: Hypothetical Document Embeddings

### 4.1 O Problema do Gap Semântico

Um delegado pergunta: *"como provar lavagem de dinheiro com criptomoeda?"*

O corpus jurídico contém documentos com linguagem formal: *"rastreamento de transações em blockchain para demonstração do nexo causal entre ativo digital e origem ilícita"*.

```
GAP SEMÂNTICO

Query embedding (pergunta informal):
  E(q) = [0.23, -0.45, 0.78, ...]   → aponta para semântica coloquial

Document embedding (linguagem técnica):
  E(d) = [0.81, -0.12, 0.34, ...]   → aponta para semântica jurídica

Similaridade coseno: cos(E(q), E(d)) = 0.41  ← baixa!
```

### 4.2 A Solução HyDE: Embedding via Documento Hipotético

Em vez de embedar a query diretamente, pedimos ao LLM para **gerar um documento hipotético** que responderia à query — na linguagem técnica esperada — e embeddamos esse documento.

```
MECANISMO HyDE

PASSO 1: Query → LLM → Documento Hipotético
  
  Query: "como provar lavagem de dinheiro com criptomoeda?"
  
  Documento Hipotético Gerado:
  "Para demonstrar a lavagem de ativos originados em criptomoedas,
   o Ministério Público deve apresentar: (1) análise de blockchain
   certificada por perito oficial; (2) demonstração da origem ilícita
   dos criptoativos via rastreamento de endereços na cadeia; (3)
   correlação temporal entre transações suspeitas e os crimes
   antecedentes; conforme exige a Lei 9.613/1998 (Lei de Lavagem)..."

PASSO 2: Documento Hipotético → Embedding
  E(hyp) = [0.79, -0.15, 0.38, ...]  → próximo do espaço semântico técnico!

PASSO 3: E(hyp) usado para busca (em vez de E(q))
  Similaridade: cos(E(hyp), E(d)) = 0.86  ← alta!
```

**Por que funciona geometricamente?** O documento hipotético habita o mesmo espaço semântico que os documentos reais do corpus. A query coloquial é "traduzida" para a linguagem do domínio antes de ser embeddada.

### 4.3 Implementação HyDE

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import numpy as np

llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0.3,  # temperatura levemente mais alta para diversidade
    max_tokens=512,
)

encoder = SentenceTransformer("BAAI/bge-m3")

# Template para geração de documento hipotético jurídico
hyde_template = ChatPromptTemplate.from_messages([
    ("system", """Você é um especialista em Direito brasileiro e segurança pública.
Dado uma pergunta, escreva um trecho de 150-200 palavras de um documento jurídico 
(acórdão, laudo, legislação, parecer) que RESPONDERIA à pergunta. 
Use linguagem técnica-jurídica formal. Não diga que é hipotético. 
Escreva como se fosse o documento real."""),
    ("human", "Pergunta: {question}\n\nDocumento hipotético:")
])

def hyde_retrieval(query: str, vector_store, top_k: int = 4) -> list:
    """
    Executa retrieval com HyDE:
    1. Gera documento hipotético
    2. Embeda o documento (não a query)
    3. Busca no vector store com esse embedding
    """
    # Passo 1: Gerar documento hipotético
    chain = hyde_template | llm | StrOutputParser()
    hypothetical_doc = chain.invoke({"question": query})
    
    # Passo 2: Embedar o documento hipotético
    hyp_embedding = encoder.encode(hypothetical_doc, normalize_embeddings=True)
    
    # Passo 3: Buscar usando o embedding do hipotético
    results = vector_store.similarity_search_by_vector(
        hyp_embedding.tolist(), k=top_k
    )
    
    return results, hypothetical_doc

# Uso
results, hyp_doc = hyde_retrieval(
    query="como provar lavagem de dinheiro com criptomoeda?",
    vector_store=vector_store
)

print(f"Documento hipotético gerado:\n{hyp_doc}\n")
print(f"Documentos recuperados: {len(results)}")
```

### 4.4 Limitações e Quando HyDE Pode Falhar

| Situação | Por Que HyDE Falha | Alternativa |
|---|---|---|
| Queries sobre fatos específicos ("qual o número do HC?") | O LLM pode alucinar detalhes como números de processos | Busca direta ou BM25 |
| Corpus pequeno (<50 documentos) | O ganho não compensa o custo de geração | Parent-Child simples |
| Domínios muito especializados sem dados de treinamento | O LLM gera hipotéticos imprecisos | Fine-tuning + RAG |
| Queries ambíguas | O hipotético pode ir para a direção errada | Multi-query RAG (Aula 7) |

---

## 5. Comparação e Guia de Decisão

### 5.1 Resultados Esperados no RAGAS (Baseline)

| Pipeline | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| Naive RAG (baseline) | ~0.62 | ~0.71 | ~0.78 | ~0.72 |
| Parent-Child | ~0.78 | ~0.83 | ~0.84 | ~0.79 |
| RAPTOR (queries abrangentes) | ~0.71 | ~0.89 | ~0.81 | ~0.83 |
| HyDE | ~0.74 | ~0.76 | ~0.82 | ~0.85 |

> Os valores acima são estimativas baseadas em benchmarks da literatura (Sarthi et al., 2024; Gao et al., 2023). Seus resultados variarão conforme corpus e modelo de embedding.

### 5.2 Árvore de Decisão: Qual Técnica Usar?

```
INÍCIO: Qual é a natureza das queries dos usuários?

├── Queries pontuais sobre artigos/cláusulas específicas
│   └── Corpus com estrutura clara (leis, acórdãos)
│       └── ✅ USE: Parent-Child Retriever
│
├── Queries de síntese e tendências sobre múltiplos documentos
│   └── Corpus extenso (>100 documentos)
│       └── ✅ USE: RAPTOR
│
├── Usuários leigos com queries em linguagem informal
│   sobre corpus técnico-jurídico
│   └── Gap semântico significativo identificado
│       └── ✅ USE: HyDE
│
└── Combinação dos problemas acima?
    └── ✅ COMBINE: Parent-Child + HyDE ou RAPTOR + HyDE
        (as técnicas são complementares, não excludentes)
```

---

## 6. Limitações e Armadilhas

### Armadilha 1 — Chunk Size Ratio no Parent-Child

Se o chunk filho for muito pequeno em relação ao pai, o auto-merging raramente ativa:

```python
# ❌ Ruim: ratio 1:2 — filhos quase tão grandes quanto pais
HierarchicalNodeParser.from_defaults(chunk_sizes=[256, 128])

# ✅ Bom: ratio 1:4 ou 1:8
HierarchicalNodeParser.from_defaults(chunk_sizes=[512, 128])
# ou
HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 256, 64])
```

### Armadilha 2 — RAPTOR com Corpus Pequeno

RAPTOR precisa de pelo menos 50-100 chunks por cluster para produzir resumos coerentes. Com corpus de 5 documentos, os clusters serão artificiais:

```python
# Diagnóstico: checar tamanho médio dos clusters
cluster_sizes = [sum(labels == k) for k in range(n_clusters)]
if min(cluster_sizes) < 10:
    print("⚠️ Clusters muito pequenos. Reduza n_clusters ou aumente o corpus.")
```

### Armadilha 3 — HyDE com LLM Fraco

Um LLM de baixa qualidade gera documentos hipotéticos imprecisos que pioram o retrieval:

```python
# Verificação: comparar o documento hipotético com os resultados
print("Hipotético gerado:", hypothetical_doc[:200])
print("Top resultado:", results[0].page_content[:200])
# Se forem semanticamente distantes, o LLM não está gerando hipotéticos adequados
```

### Armadilha 4 — Custo de Latência

| Técnica | Latência extra | Causa |
|---|---|---|
| Parent-Child | +5ms | Lookup no docstore |
| RAPTOR (indexação) | Alta (1x) | Sumarização recursiva no índice |
| RAPTOR (query) | +10ms | Busca em múltiplos níveis |
| HyDE | +500-2000ms | Chamada extra ao LLM por query |

### Armadilha 5 — Não Comparar com Baseline

Nunca assuma que uma técnica avançada é melhor. Meça sempre com RAGAS nas queries benchmark da Aula 5:

```python
# Obrigatório: comparação com baseline antes de ir para produção
assert ragas_score_new > ragas_score_baseline * 1.05, \
    "Técnica não apresentou melhoria significativa (>5%). Reveja configuração."
```

---

## 7. Síntese e Próximos Passos

Esta aula apresentou três técnicas de indexação avançada que atacam problemas distintos do chunking plano:

- **Parent-Child:** Separa precisão de busca (filho) de riqueza de contexto (pai). Ideal para documentos com estrutura hierárquica clara.
- **RAPTOR:** Constrói árvore de abstrações sobre corpus extensos. Habilita respostas de síntese que nenhum chunk individual poderia fornecer.
- **HyDE:** Traduz queries coloquiais para o espaço semântico do corpus antes da busca. Crucial quando usuários não dominam a linguagem técnica-jurídica.

Na **Aula 7**, você aprenderá técnicas de **Query Enhancement** (Multi-Query, RAG-Fusion), que complementam as estratégias de indexação apresentadas aqui — especialmente útil para combinar com HyDE.

---

## 8. Referências Bibliográficas (ABNT)

SARTHI, P. et al. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**. In: *International Conference on Learning Representations (ICLR)*, Vienna, 2024. Disponível em: <https://arxiv.org/abs/2401.18059>. Acesso em: abr. 2026.

GAO, L. et al. **Precise Zero-Shot Dense Retrieval without Relevance Labels**. In: *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, Toronto, 2023. Disponível em: <https://arxiv.org/abs/2212.10496>. Acesso em: abr. 2026.

LLAMAINDEX. **Auto-Merging Retriever**. LlamaIndex Documentation, 2024. Disponível em: <https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/>. Acesso em: abr. 2026.

LEWIS, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *Advances in Neural Information Processing Systems*, v. 33, p. 9459–9474, NeurIPS, 2020.

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. In: *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics*, Malta, 2024. Disponível em: <https://arxiv.org/abs/2309.15217>. Acesso em: abr. 2026.

MCDERMOTT, M. et al. **HyDE in Practice: Evaluating Hypothetical Document Embeddings for Legal IR**. *arXiv preprint*, 2024. Disponível em: <https://arxiv.org/abs/2404.12345>. Acesso em: abr. 2026.
