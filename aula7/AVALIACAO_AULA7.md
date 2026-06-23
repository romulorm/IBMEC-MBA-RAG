# Avaliação — Aula 7: Query Enhancement
## Multi-Query RAG, RAG-Fusion e Step-Back Prompting
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 7 de 12 | **Peso no curso:** 8,33% (1/12)

---

## Critérios Gerais

| Critério | Descrição |
|---|---|
| **Funcionalidade** | O código executa sem erros no Google Colab com GPU T4 |
| **Documentação** | Células markdown explicam cada etapa em português |
| **Qualidade técnica** | Parâmetros justificados; thresholds documentados |
| **Análise crítica** | Aluno interpreta os resultados, não apenas exibe números |

---

## Rubrica por Entregável

### E1 — Multi-Query RAG com LangChain (25 pontos)

**Arquivo:** `labs/LAB1_Multi_Query_RAG.ipynb`

| Item | Pontos | Critério |
|---|---|---|
| MultiQueryRetriever configurado com vLLM | 5 | `llm=ChatOpenAI(base_url=...)` correto |
| 4 variações geradas para ao menos 3 queries jurídicas | 5 | Variações semanticamente distintas (não paráfrases triviais) |
| Deduplicação implementada com threshold coseno | 5 | Threshold documentado e justificado (ex: 0.85) |
| Tabela comparativa: docs únicos vs docs sem deduplicação | 5 | Evidência do ganho da deduplicação |
| Análise: o que acontece com N=8? | 5 | Observação sobre queda de qualidade ou ruído |

**Nota mínima para aprovação:** 15/25

---

### E2 — Step-Back Prompting (20 pontos)

**Arquivo:** `labs/LAB2_Step_Back_Prompting.ipynb`

| Item | Pontos | Critério |
|---|---|---|
| Prompt de abstração implementado com vLLM | 5 | Prompt claro com instrução de generalização |
| Tabela com 5 pares: query original → query abstraída | 5 | Abstrações são genuinamente mais gerais |
| Comparação de recuperação: antes vs depois do Step-Back | 5 | Evidência de melhora (ou não) com métrica ou contagem de docs relevantes |
| Análise de falhas: quando o Step-Back não ajuda | 5 | Identificação de pelo menos 1 caso onde abstração prejudica o recall |

**Nota mínima para aprovação:** 12/20

---

### E3 — RAG-Fusion Completo (25 pontos)

**Arquivo:** `labs/LAB3_RAG_Fusion_Completo.ipynb`

| Item | Pontos | Critério |
|---|---|---|
| Geração de N sub-queries com vLLM | 5 | Sub-queries cobrindo ângulos distintos do problema |
| Retrieval paralelo com asyncio | 5 | `asyncio.gather()` ou equivalente documentado |
| RRF implementado manualmente (não como black-box) | 8 | Fórmula `1/(k+rank)` visível no código com exemplo numérico |
| Geração final com Llama 3.1 a partir do contexto fundido | 4 | Prompt inclui documentos rankados por RRF |
| Exemplo de query jurídica complexa respondida corretamente | 3 | Resposta coerente e fundamentada nos documentos |

**Nota mínima para aprovação:** 15/25

---

### E4 — Benchmark N=1/3/5 (20 pontos)

**Arquivo:** `labs/LAB4_Benchmark_N_Queries.ipynb`

| Item | Pontos | Critério |
|---|---|---|
| Tabela com recall, latência e tokens para N=1, N=3, N=5 | 8 | 3 métricas × 3 valores de N medidos |
| Gráfico: recall × N (curva) | 4 | Visualização mostra ponto de retorno decrescente |
| Gráfico: latência × N (curva) | 4 | Visualização mostra crescimento (esperado) |
| Análise de trade-off: recomendação fundamentada | 4 | Aluno justifica o N ideal para o projeto final |

**Nota mínima para aprovação:** 12/20

---

### E5 — Análise de Custo e LangFuse (10 pontos)

**Arquivo:** `labs/LAB5_LangFuse_Custo_Traces.ipynb`

| Item | Pontos | Critério |
|---|---|---|
| Traces das 3 abordagens visíveis no LangFuse | 4 | Screenshots ou link para dashboard |
| Cálculo de custo por 1.000 queries (cada abordagem) | 3 | Fórmula: tokens × preço-por-token documentada |
| Conclusão escrita: qual abordagem usar e por quê | 3 | Argumento considera custo, recall e latência |

**Nota mínima para aprovação:** 6/10

---

## Pontuação Total

| Entregável | Peso | Nota Máxima |
|---|---|---|
| E1 — Multi-Query RAG | 25% | 25 pts |
| E2 — Step-Back Prompting | 20% | 20 pts |
| E3 — RAG-Fusion Completo | 25% | 25 pts |
| E4 — Benchmark N=1/3/5 | 20% | 20 pts |
| E5 — Análise e LangFuse | 10% | 10 pts |
| **TOTAL** | **100%** | **100 pts** |

**Aprovação:** ≥ 60 pontos com nota mínima em E1, E3 e E4.

---

## Bônus (até 10 pontos extras)

| Bônus | Pontos | Critério |
|---|---|---|
| Langflow funcionando (LAB6) | 5 pts | RAG-Fusion construído visualmente como fluxo paralelo |
| Query Decomposition implementada | 5 pts | Perguntas compostas decompostas em sub-queries independentes (além do escopo do LAB3) |

---

## Feedback Esperado pelo Professor

Ao corrigir, verificar especialmente:

1. **E3 (RRF):** O aluno entende a fórmula ou apenas copiou? Verificar se há um exemplo numérico comentado no código.
2. **E4 (Benchmark):** Os valores de recall são calculados com RAGAS ou são apenas contagem manual? Ambos são válidos, mas RAGAS é preferível.
3. **E5 (Custo):** O aluno considerou o custo das chamadas de geração de sub-queries, não apenas o retrieval.

---

## Referências para o Professor

MA, X. et al. **Query Rewriting for Retrieval-Augmented LLMs**. arXiv:2305.14283, 2023.

RACKAUCKAS, Z. **RAG-Fusion**. arXiv:2402.03367, 2024.

CORMACK, G. V. et al. **Reciprocal Rank Fusion**. SIGIR, 2009.
