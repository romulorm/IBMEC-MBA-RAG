# Aula 8 — Self-RAG, CRAG e LangGraph: RAG Reflexivo e Auto-Corretivo

**Curso:** MBA em RAG & CAG Aplicados a Direito e Segurança Pública  
**Carga Horária:** 5 horas  
**Proporção:** 25% teoria / 75% prática  
**Nível:** RAG Avançado  

---

## Ementa

Técnicas de RAG com capacidade de auto-avaliação e correção automática. Self-RAG com tokens de controle especiais para recuperação reflexiva sob demanda. CRAG com avaliador de qualidade e web search como fallback inteligente. Orquestração de pipelines complexos com LangGraph usando StateGraph, nós e arestas condicionais.

---

## Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. Compreender a arquitetura Self-RAG e seus quatro tokens de controle especiais (`[Retrieve]`, `[ISREL]`, `[ISSUP]`, `[ISUSE]`)
2. Implementar um pipeline CRAG com nó de avaliação e roteamento condicional
3. Construir grafos de execução com LangGraph usando ciclos e condicionais
4. Integrar a API Tavily como fallback de web search quando o retrieval local falha
5. Monitorar pipelines multi-caminho com LangFuse
6. Avaliar com RAGAS se auto-correção melhora a métrica de Faithfulness

---

## Estrutura de Arquivos

```
aula8/
├── INDICE_AULA8.md               ← Este arquivo
├── AVALIACAO_AULA8.md            ← Critérios e rubrica de avaliação
│
├── teoria/
│   └── AULA8_TEORIA.md           ← Teoria completa (Self-RAG, CRAG, LangGraph)
│
├── exemplos/
│   ├── EXEMPLO1_Self_RAG_Minimo.ipynb   ← Self-RAG básico com Ollama
│   └── EXEMPLO2_CRAG_Basico.ipynb       ← CRAG mínimo sem LangGraph
│
├── labs/
│   ├── LAB1_Self_RAG_vLLM.ipynb             ← Self-RAG: tokens de controle via vLLM
│   ├── LAB2_CRAG_LangGraph.ipynb            ← CRAG completo com StateGraph
│   ├── LAB3_Avaliador_LLM_as_Judge.ipynb    ← Avaliador de relevância 0-1
│   ├── LAB4_Tavily_Integracao.ipynb         ← Web search fallback
│   ├── LAB5_LangFuse_Traces_CRAG.ipynb      ← Monitoramento de rotas
│   └── LAB6_RAGAS_CRAG_vs_AdvancedRAG.ipynb ← Comparativo Faithfulness
│
└── datasets/
    ├── corpus_juridico_aula8.json    ← Corpus jurídico para retrieval
    └── queries_teste_aula8.json      ← Queries de teste Self-RAG e CRAG
```

---

## Roteiro da Aula

### Bloco 1 — Teoria (75 min)

| # | Tópico | Duração |
|---|--------|---------|
| 1.1 | Self-RAG: arquitetura e tokens de controle | 20 min |
| 1.2 | CRAG: avaliador de qualidade e roteamento | 15 min |
| 1.3 | LangGraph: StateGraph, nós e arestas condicionais | 25 min |
| 1.4 | Tavily API: web search como fallback | 10 min |
| 1.5 | Tabela comparativa: Self-RAG vs CRAG vs Advanced RAG | 5 min |

### Bloco 2 — Exemplos Guiados (30 min)

| # | Notebook | Duração |
|---|----------|---------|
| 2.1 | EXEMPLO1: Self-RAG mínimo com Ollama | 15 min |
| 2.2 | EXEMPLO2: CRAG básico sem LangGraph | 15 min |

### Bloco 3 — Laboratórios Práticos (175 min)

| # | Lab | Duração |
|---|-----|---------|
| 3.1 | LAB1: Self-RAG com tokens de controle via vLLM | 25 min |
| 3.2 | LAB2: CRAG completo com LangGraph | 35 min |
| 3.3 | LAB3: Avaliador LLM-as-Judge | 25 min |
| 3.4 | LAB4: Integração Tavily | 25 min |
| 3.5 | LAB5: LangFuse — traces CRAG | 30 min |
| 3.6 | LAB6: RAGAS comparativo | 35 min |

---

## Stack Tecnológico

| Ferramenta | Versão | Uso |
|------------|--------|-----|
| LangGraph | ≥ 0.2 | Orquestração de grafo com StateGraph |
| LangChain | ≥ 0.3 | Chains e retrievers |
| vLLM | ≥ 0.4 | Servidor de inferência OpenAI-compatible para Self-RAG |
| Tavily API | - | Web search fallback |
| LangFuse | ≥ 2.0 | Observabilidade e traces |
| RAGAS | ≥ 0.1 | Avaliação Faithfulness |
| Python | 3.11+ | Linguagem base |

---

## Referências Acadêmicas

- ASAI, A. et al. **Self-RAG: Learning to Retrieve, Generate, and Critique**. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.
- YAN, S. et al. **Corrective Retrieval Augmented Generation**. In: *International Conference on Learning Representations (ICLR)*, 2024.
- CHASE, H. **LangGraph: Building Stateful, Multi-Actor Applications**. LangChain Blog, 2023.

---

*MBA em RAG & CAG Aplicados a Direito e Segurança Pública · Aula 8 de 12*
