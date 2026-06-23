# Avaliação — Aula 8: Self-RAG, CRAG e LangGraph

**Curso:** MBA em RAG & CAG Aplicados a Direito e Segurança Pública  
**Aula:** 8 de 12 — Self-RAG, CRAG e LangGraph  

---

## Critérios de Avaliação

### Entrega 1 — Pipeline CRAG Funcional (40 pontos)

O aluno deve entregar um notebook com pipeline CRAG completo usando LangGraph:

| Critério | Pontos |
|----------|--------|
| StateGraph com 4 nós implementados corretamente (`retrieve`, `grade_documents`, `web_search`, `generate`) | 15 |
| Avaliador LLM-as-Judge retornando score 0–1 com threshold em 0.5 | 10 |
| Roteamento condicional funcional (local → web → generate) | 10 |
| Execução sem erros com pelo menos 3 queries de teste | 5 |

### Entrega 2 — Trace LangFuse (30 pontos)

O aluno deve apresentar evidências de monitoramento no LangFuse:

| Critério | Pontos |
|----------|--------|
| Trace completo capturado com spans para cada nó do grafo | 10 |
| Taxa de ativação do web search calculada (≥ 5 queries executadas) | 10 |
| Análise de latência por rota (rota local vs rota web search) | 5 |
| Comparativo de custo por caminho documentado | 5 |

### Entrega 3 — Comparativo RAGAS (30 pontos)

O aluno deve executar avaliação comparativa entre CRAG e Advanced RAG (baseline da Aula 7):

| Critério | Pontos |
|----------|--------|
| Métrica Faithfulness calculada para CRAG | 10 |
| Métrica Faithfulness calculada para Advanced RAG (baseline) | 10 |
| Diferença documentada com análise qualitativa (mínimo 5 linhas) | 10 |

---

## Rubrica de Conceitos

| Pontuação | Conceito | Descrição |
|-----------|----------|-----------|
| 90–100 | **A** | Pipeline CRAG funcional, trace completo, melhora demonstrada em Faithfulness |
| 75–89 | **B** | Pipeline funcional com pequenas falhas, trace parcial, comparativo realizado |
| 60–74 | **C** | Pipeline funcional porém sem LangFuse ou RAGAS completo |
| 0–59 | **D** | Pipeline incompleto ou não funcional |

---

## Prazo e Formato

- **Formato:** Notebook `.ipynb` compatível com Google Colab
- **Evidências:** Screenshots do LangFuse e outputs do RAGAS embutidos no notebook
- **Entrega:** Via plataforma do curso até o início da Aula 9

---

## Dicas de Avaliação

1. O avaliador verificará se o grafo LangGraph realmente executa caminhos diferentes dependendo do score do avaliador — use queries que forcem os dois caminhos
2. Para o trace LangFuse, é suficiente mostrar ao menos uma execução por rota (local e web search)
3. Para RAGAS, use o mesmo conjunto de perguntas para ambos os sistemas para garantir comparabilidade

---

*MBA em RAG & CAG Aplicados a Direito e Segurança Pública · Aula 8 de 12*
