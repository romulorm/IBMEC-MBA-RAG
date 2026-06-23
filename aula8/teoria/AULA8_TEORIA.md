# Aula 8 — Teoria: Self-RAG, CRAG e LangGraph

**Curso:** MBA em RAG & CAG Aplicados a Direito e Segurança Pública  
**Aula:** 8 de 12 · 5h · RAG Avançado  
**Normas:** ABNT NBR 6023:2018 (Referências) / NBR 10520:2023 (Citações)

---

## Sumário

1. [Motivação: Limitações do RAG Convencional](#1-motivacao)
2. [Self-RAG: Recuperação Reflexiva sob Demanda](#2-self-rag)
3. [CRAG: Corrective Retrieval Augmented Generation](#3-crag)
4. [LangGraph: Orquestração de Grafos de Execução](#4-langgraph)
5. [Tavily: Web Search como Fallback](#5-tavily)
6. [Tabela Comparativa: Self-RAG vs CRAG vs Advanced RAG](#6-comparativo)
7. [Aplicações no Contexto Jurídico e de Segurança Pública](#7-aplicacoes)
8. [Referências](#8-referencias)

---

## 1. Motivação: Limitações do RAG Convencional {#1-motivacao}

O RAG convencional executa sempre o mesmo fluxo linear: **recuperar → aumentar → gerar**. Essa abordagem apresenta limitações críticas em contextos de alta precisão como o Direito:

- **Recuperação indiscriminada:** o sistema sempre busca documentos, mesmo quando a resposta é conhecimento geral do modelo (ex.: "O que é um contrato?")
- **Sem verificação de relevância:** documentos recuperados são usados independentemente de sua pertinência
- **Sem autocorreção:** se o retrieval falha, a geração produz respostas imprecisas ou alucinadas sem nenhum mecanismo de detecção

> "A recuperação não seletiva introduz ruído que pode degradar a qualidade da resposta — e em sistemas jurídicos, ruído pode significar erro factual com consequências reais." (ASAI et al., 2023, tradução nossa)

As técnicas apresentadas nesta aula atacam essas três limitações de forma direta.

---

## 2. Self-RAG: Recuperação Reflexiva sob Demanda {#2-self-rag}

### 2.1 Conceito e Origem

Self-RAG (*Self-Reflective Retrieval Augmented Generation*) é uma técnica proposta por Asai et al. (2023) que treina o modelo de linguagem para **decidir autonomamente** quando recuperar documentos, avaliar a qualidade do que foi recuperado e verificar a fidelidade factual de sua própria geração.

A ideia central é: em vez de sempre recuperar, o modelo aprende a **emitir tokens de controle especiais** durante a geração que sinalizam quando e como o retrieval deve ocorrer.

### 2.2 Os Quatro Tokens de Controle

O Self-RAG introduz quatro tokens especiais que o modelo emite durante a geração:

| Token | Nome | Função | Valores Possíveis |
|-------|------|---------|-------------------|
| `[Retrieve]` | Recuperação | Decide se deve buscar documentos | `yes` / `no` |
| `[ISREL]` | Is Relevant | Avalia se o documento recuperado é relevante | `relevant` / `irrelevant` |
| `[ISSUP]` | Is Supported | Verifica se a geração tem suporte factual no documento | `fully supported` / `partially supported` / `no support` |
| `[ISUSE]` | Is Useful | Avalia se a resposta é útil para a pergunta | `5` / `4` / `3` / `2` / `1` |

### 2.3 Fluxo de Decisão Token a Token

A geração em Self-RAG funciona de forma iterativa. Acompanhe o fluxo com um exemplo jurídico:

**Pergunta:** *"Qual o prazo prescricional para ação de indenização por danos morais no Brasil?"*

```
Modelo inicia a geração:
  → Emite [Retrieve=yes]                    # Pergunta requer documento específico
  → Sistema recupera documentos do corpus
  → Modelo avalia: [ISREL=relevant]         # CC Art. 206 é relevante
  → Modelo gera: "O prazo prescricional é de 3 anos, conforme o Art. 206, §3º, V do Código Civil..."
  → Avalia suporte: [ISSUP=fully supported] # A informação está no documento
  → Avalia utilidade: [ISUSE=5]             # Resposta direta e completa
```

**Pergunta:** *"O que é responsabilidade civil?"*

```
Modelo inicia a geração:
  → Emite [Retrieve=no]                     # Conhecimento geral, não precisa recuperar
  → Modelo gera diretamente a definição
  → Avalia utilidade: [ISUSE=4]             # Boa resposta sem retrieval
```

### 2.4 Treinamento do Self-RAG

O modelo é treinado em duas etapas:

**Etapa 1 — Geração de dados sintéticos:** Um modelo "crítico" (geralmente GPT-4) é usado para anotar textos existentes com os tokens de controle, criando um dataset de treinamento que associa perguntas e contextos aos tokens corretos.

**Etapa 2 — Fine-tuning supervisionado:** O modelo base (ex.: LLaMA) é treinado nesse dataset anotado, aprendendo a emitir os tokens de controle como parte natural da geração de texto.

### 2.5 Vantagens e Limitações

**Vantagens:**
- Alta factualidade por verificação integrada (`[ISSUP]`)
- Recuperação cirúrgica — evita ruído desnecessário
- Auto-avaliação de utilidade (`[ISUSE]`) permite selecionar melhor resposta

**Limitações:**
- **Requer fine-tuning específico** — não funciona com qualquer LLM "fora da caixa"
- Maior latência em queries que ativam múltiplos ciclos de retrieval
- Disponibilidade limitada de modelos treinados com Self-RAG (principal: `llama-2-7b-selfrag`)

---

## 3. CRAG: Corrective Retrieval Augmented Generation {#3-crag}

### 3.1 Conceito e Motivação

CRAG (*Corrective Retrieval Augmented Generation*), proposto por Yan et al. (2024), aborda o problema de forma diferente: **não requer fine-tuning** e funciona como uma camada de correção sobre qualquer pipeline RAG existente.

A intuição central é simples: antes de usar os documentos recuperados para gerar uma resposta, um **avaliador de qualidade** examina a relevância dos documentos. Dependendo do score, o sistema toma três caminhos possíveis:

```
Score alto (≥ 0.7)  → Usa documentos locais
Score médio (0.3–0.7) → Combina documentos locais + web search
Score baixo (< 0.3) → Descarta documentos locais, usa web search
```

### 3.2 Arquitetura do Avaliador

O avaliador de qualidade no CRAG é tipicamente implementado como **LLM-as-Judge**: um modelo de linguagem que recebe a pergunta e os documentos recuperados e retorna um score de relevância entre 0 e 1.

**Prompt padrão do avaliador:**

```
Você é um avaliador de relevância de documentos para sistemas RAG.

Pergunta: {question}
Documento: {document}

Avalie a relevância deste documento para responder à pergunta acima.
Retorne APENAS um JSON com o campo "score" entre 0 e 1:
- 1.0: documento altamente relevante e diretamente responde à pergunta
- 0.5: documento parcialmente relevante, contém informação útil mas incompleta
- 0.0: documento irrelevante, não contribui para responder à pergunta

Resposta: {"score": <valor>}
```

### 3.3 Web Search como Fallback com Tavily

Quando o score do avaliador é baixo, o CRAG aciona a busca na web usando a **Tavily API** — uma API de busca otimizada para uso com LLMs que retorna resultados estruturados e relevantes.

O resultado da busca é então fundido com os documentos locais (no caso de score médio) ou usado isoladamente (score baixo), antes de ser enviado ao gerador.

### 3.4 Vantagens e Limitações

**Vantagens:**
- **Training-free** — funciona com qualquer LLM e pipeline RAG existente
- Auto-corretivo — robusto a falhas de retrieval local
- Cobre gaps de conhecimento com informações atualizadas via web search

**Limitações:**
- Latência adicional (chamada ao avaliador + possível web search)
- Dependência de API externa (Tavily) para o caminho de fallback
- Custo adicional do avaliador LLM por chamada

---

## 4. LangGraph: Orquestração de Grafos de Execução {#4-langgraph}

### 4.1 O que é LangGraph?

LangGraph é uma biblioteca construída sobre o LangChain que permite criar **fluxos de execução não-lineares** — grafos onde a lógica pode divergir, convergir, criar ciclos e tomar decisões condicionais com base no estado atual.

Pense no LangGraph como uma **máquina de estados finita** onde:
- **Estados** = dicionários Python com informações do pipeline
- **Nós** = funções Python que leem e modificam o estado
- **Arestas** = conexões entre nós (podem ser condicionais)

### 4.2 Conceitos Fundamentais

#### StateGraph

O `StateGraph` é o grafo principal. Você define o **esquema do estado** (o que será passado entre os nós) e adiciona nós e arestas:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Define o estado — o que circula pelo grafo
class GraphState(TypedDict):
    question: str           # Pergunta do usuário
    documents: List[str]    # Documentos recuperados
    web_results: List[str]  # Resultados do web search
    generation: str         # Resposta gerada
    relevance_score: float  # Score do avaliador
```

#### Nós (Nodes)

Cada nó é uma função Python que recebe o estado e retorna um dicionário com as chaves a atualizar:

```python
def retrieve(state: GraphState) -> dict:
    """Nó de recuperação: busca documentos relevantes."""
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": [doc.page_content for doc in documents]}
```

#### Arestas Condicionais (Conditional Edges)

As arestas condicionais são funções que decidem para qual nó ir baseadas no estado atual:

```python
def route_documents(state: GraphState) -> str:
    """Decide o próximo nó com base no score de relevância."""
    score = state["relevance_score"]
    if score >= 0.7:
        return "generate"      # Documentos bons → gerar diretamente
    elif score >= 0.3:
        return "web_search"    # Score médio → buscar web
    else:
        return "web_search"    # Score baixo → buscar web
```

### 4.3 Estrutura do Grafo CRAG

O pipeline CRAG implementado em LangGraph tem 4 nós principais:

```
[START]
   │
   ▼
[retrieve]          ← Recupera documentos do índice local
   │
   ▼
[grade_documents]   ← Avaliador LLM-as-Judge retorna score 0-1
   │
   ├─── score ≥ 0.5 ──→ [generate]  ← Gera resposta com docs locais
   │                          │
   └─── score < 0.5 ──→ [web_search] ← Busca no Tavily
                              │
                              ▼
                          [generate]  ← Gera resposta com web results
                              │
                              ▼
                           [END]
```

### 4.4 Ciclos no LangGraph

Uma das funcionalidades mais poderosas do LangGraph é suportar **ciclos** — o grafo pode retornar a um nó anterior. Isso é útil, por exemplo, para implementar reescrita de query quando o retrieval falha repetidamente:

```python
# Adiciona ciclo: se após web search ainda não tiver documentos relevantes,
# reescreve a query e tenta novamente
workflow.add_conditional_edges(
    "grade_documents",
    route_documents,
    {
        "generate": "generate",
        "web_search": "web_search",
        "rewrite_query": "rewrite_query"  # ciclo de volta
    }
)
```

> **Atenção:** Ciclos sem condição de parada causam loops infinitos. Sempre defina um limite máximo de iterações no estado.

---

## 5. Tavily: Web Search como Fallback {#5-tavily}

### 5.1 O que é a Tavily API?

Tavily é uma API de busca na web projetada especificamente para uso com LLMs. Diferente de APIs de busca genéricas, a Tavily:

- Retorna resultados **já processados e resumidos** para consumo direto por LLMs
- Filtra resultados irrelevantes automaticamente
- Suporta busca com contexto para melhorar a qualidade dos resultados

### 5.2 Configuração e Uso Básico

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Configurar (necessário TAVILY_API_KEY no ambiente)
tavily_tool = TavilySearchResults(
    max_results=3,          # Número de resultados
    search_depth="advanced" # "basic" ou "advanced"
)

# Buscar
results = tavily_tool.invoke({"query": "prazo prescricional danos morais Brasil 2024"})
```

### 5.3 Estrutura da Resposta Tavily

```python
# Cada resultado é um dicionário:
{
    "url": "https://...",
    "content": "Texto resumido do resultado...",
    "score": 0.95  # Relevância (0-1)
}
```

### 5.4 Fusão com Resultados Locais

Para o caso de score médio (fusão de fontes), os resultados Tavily são concatenados aos documentos locais antes da geração:

```python
def fuse_results(local_docs: list, web_results: list) -> str:
    """Funde documentos locais e web para uso no gerador."""
    context_parts = []
    
    # Documentos locais
    if local_docs:
        context_parts.append("=== Documentos Locais ===")
        for i, doc in enumerate(local_docs, 1):
            context_parts.append(f"[Doc {i}]: {doc}")
    
    # Resultados web
    if web_results:
        context_parts.append("\n=== Resultados Web ===")
        for i, result in enumerate(web_results, 1):
            context_parts.append(f"[Web {i}]: {result['content']}")
    
    return "\n\n".join(context_parts)
```

---

## 6. Tabela Comparativa: Self-RAG vs CRAG vs Advanced RAG {#6-comparativo}

| Critério | Advanced RAG | Self-RAG | CRAG |
|----------|-------------|----------|------|
| **Abordagem** | Retrieval otimizado (query expansion, reranking) | Recuperação reflexiva com tokens de controle | Avaliador de qualidade + fallback web |
| **Fine-tuning necessário** | Não | **Sim** (modelo específico) | Não |
| **Quando recuperar** | Sempre | Sob demanda (decide por token) | Sempre, mas avalia qualidade |
| **Avaliação de relevância** | Reranker externo | Integrado no modelo (`[ISREL]`) | LLM-as-Judge externo |
| **Fallback automático** | Não | Não | **Sim** (web search via Tavily) |
| **Complexidade de impl.** | Média | Alta (depende de modelo específico) | Média-Alta |
| **Latência adicional** | Baixa | Média | Média-Alta |
| **Custo** | Baixo | Médio | Médio (avaliador + Tavily) |
| **Melhor caso de uso** | Corpus grande e estável | Alta precisão factual exigida | Dados dinâmicos ou corpus incompleto |

### Guia de Decisão

```
Tenho acesso ao modelo Self-RAG treinado?
├── Sim → Preciso de alta precisão factual com rastreabilidade?
│         ├── Sim → USE Self-RAG
│         └── Não → Advanced RAG é suficiente
│
└── Não → Meu corpus pode estar desatualizado ou incompleto?
          ├── Sim → USE CRAG (com fallback web)
          └── Não → Advanced RAG + Reranker é suficiente
```

---

## 7. Aplicações no Contexto Jurídico e de Segurança Pública {#7-aplicacoes}

### 7.1 Self-RAG em Sistemas Jurídicos

O mecanismo `[ISSUP]` (Is Supported) do Self-RAG é particularmente valioso em contextos jurídicos porque garante que cada afirmação gerada tenha suporte documental verificável. Isso atende ao princípio da **rastreabilidade** exigido em pareceres e laudos técnicos.

**Caso de uso:** Assistente de análise de jurisprudência que verifica se cada conclusão tem suporte nos acórdãos recuperados, emitindo alerta quando gera conteúdo sem embasamento.

### 7.2 CRAG em Investigação Criminal

Em investigações criminais, o corpus de documentos internos pode não conter informações sobre eventos recentes. O CRAG resolve isso com web search automático quando o score de relevância é baixo.

**Caso de uso:** Sistema de busca de antecedentes criminais que consulta o banco interno de dados (alta relevância) mas recorre a fontes externas como DJe e Diário Oficial quando a busca interna é insuficiente.

### 7.3 LangGraph como Base para Agentes Investigativos

A capacidade do LangGraph de criar ciclos e condicionais permite modelar fluxos investigativos complexos com múltiplas rotas de raciocínio — similar à lógica de uma investigação real que pode abrir novas linhas de investigação com base nas evidências encontradas.

---

## 8. Referências {#8-referencias}

ASAI, Akari et al. **Self-RAG: Learning to Retrieve, Generate, and Critique**. In: *Advances in Neural Information Processing Systems (NeurIPS 2023)*. New Orleans: NeurIPS Foundation, 2023. Disponível em: https://arxiv.org/abs/2310.11511. Acesso em: 18 abr. 2026.

CHASE, Harrison. **LangGraph: Building Stateful, Multi-Actor Applications with LLMs**. *LangChain Blog*. San Francisco: LangChain Inc., 2023. Disponível em: https://blog.langchain.dev/langgraph. Acesso em: 18 abr. 2026.

YAN, Shi-Qi et al. **Corrective Retrieval Augmented Generation**. In: *International Conference on Learning Representations (ICLR 2024)*. Vienna: OpenReview.net, 2024. Disponível em: https://arxiv.org/abs/2401.15884. Acesso em: 18 abr. 2026.

---

*MBA em RAG & CAG Aplicados a Direito e Segurança Pública · Aula 8 de 12*
