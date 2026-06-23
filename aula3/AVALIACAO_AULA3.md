# Avaliação — Aula 3: Advanced RAG e Modular RAG
## MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 3 de 12 | **Peso na nota final:** 8% | **Data de entrega:** 7 dias após a aula

---

## Critérios e Rubricas

### Entregável 1 — Pipeline Advanced RAG Funcional (40 pontos)
*Labs: LAB1 (Query Rewriting) + LAB2 (BGE-Reranker)*

| Critério | Excelente (4) | Satisfatório (2-3) | Insuficiente (0-1) |
|---|---|---|---|
| **Query Rewriting implementado** | 3 técnicas funcionais (paraphrase, HyDE-lite, step-back) com output visível para 5 queries | 2 técnicas implementadas com saída parcial | Menos de 2 técnicas ou nenhuma saída |
| **BGE-Reranker integrado** | Reranker carregado e aplicado; scores entre 0-1 para todos os docs | Reranker carregado mas scores inconsistentes | Reranker não carregado ou scores ausentes |
| **Tabela comparativa top-5** | Tabela com rank antes/depois + score BM25 + score reranker para 5 queries | Tabela parcial (< 5 queries ou colunas faltando) | Tabela ausente |
| **Análise de vocabulary gap** | Identifica o gap entre linguagem coloquial e técnica com exemplos do corpus | Menciona o problema sem exemplos concretos | Não aborda o problema |
| **Código executável** | Código executa end-to-end no Colab sem erros | Executa com warnings menores | Erros de execução |

**Subtotal: ___/20** (cada critério vale 4 pontos)

---

### Entregável 2 — Análise Comparativa (25 pontos)
*Lab: LAB3 (Análise Qualitativa)*

| Critério | Excelente (5) | Satisfatório (3) | Insuficiente (0-1) |
|---|---|---|---|
| **Avaliação 5D realizada** | Todas as 5 dimensões avaliadas para Naive RAG e Advanced RAG (10 avaliações total) | 3-4 dimensões avaliadas | Menos de 3 dimensões |
| **Comparação quantitativa** | Tabela com scores numéricos e melhoria percentual calculada | Comparação qualitativa sem números | Comparação ausente |
| **Identificação da melhoria mais impactante** | Identifica a dimensão com maior δ e justifica com exemplo específico do corpus | Identifica sem justificativa | Não identifica |
| **Análise de casos negativos** | Menciona algum caso onde Advanced RAG não melhorou (ou piorou) e explica | Apenas casos positivos | Análise ausente |
| **Conclusão fundamentada** | Conclusão cita evidências do experimento + referência bibliográfica | Conclusão genérica sem evidências | Conclusão ausente |

**Subtotal: ___/25**

---

### Entregável 3 — Pipeline Modular RAG (20 pontos)
*Lab: LAB4 (Modular RAG)*

| Critério | Excelente (5) | Satisfatório (3) | Insuficiente (0-1) |
|---|---|---|---|
| **Interfaces abstratas (ABC)** | BaseRetriever, BaseReranker, BaseGenerator corretamente definidos com docstrings | Interfaces definidas sem docstrings | Interfaces ausentes ou incorretas |
| **Dois retrievers implementados** | OpenSearchRetriever e ChromaDBRetriever com a mesma interface | Um retriever implementado | Nenhum retriever implementado |
| **Intercambialidade demonstrada** | swap_retriever() executado sem alterar reranker ou generator; saída dos dois retrievers visível | swap_retriever() executado mas demais módulos foram alterados | Troca não demonstrada |
| **Análise de overlap** | Calcula % de documentos em comum entre retrievers e discute o resultado | Menciona diferença sem calcular | Análise ausente |

**Subtotal: ___/20**

---

### Entregável 4 — LangFuse e Análise de Trace (15 pontos)
*Labs: LAB5 (Instrumentação) + LAB6 (Otimização)*

| Critério | Excelente (5) | Satisfatório (3) | Insuficiente (0-1) |
|---|---|---|---|
| **Instrumentação com @observe** | Todos os 4 módulos instrumentados; trace com spans visível no LangFuse ou print do JSON | 2-3 módulos instrumentados | Menos de 2 módulos |
| **Análise de gargalo** | Identifica o módulo mais lento com dados quantitativos (ms e % do total) | Identifica o módulo sem dados | Análise ausente |
| **Proposta de otimização** | Propõe pelo menos 2 estratégias com: nome, descrição, impacto esperado, risco | 1 estratégia proposta com impacto esperado | Proposta ausente |

**Subtotal: ___/15**

---

## Pontuação Total

| Entregável | Pontos | Peso |
|---|---|---|
| 1. Pipeline Advanced RAG | ___/20 | 40% |
| 2. Análise Comparativa | ___/25 | 25% |
| 3. Pipeline Modular RAG | ___/20 | 20% |
| 4. LangFuse + Trace | ___/15 | 15% |
| **TOTAL** | **___/100** | **100%** |

**Nota final:** ___

---

## Penalizações

| Situação | Penalização |
|---|---|
| Código copiado sem execução/adaptação | -20 pontos |
| Entrega após prazo (por dia) | -5 pontos/dia (máx. -20) |
| Código que não executa no Google Colab | -10 pontos |
| Ausência de análise crítica (apenas código) | -15 pontos |

---

## Bônus

| Situação | Bônus |
|---|---|
| Implementar HyDE completo (não simplificado) com análise | +5 pontos |
| Adicionar um terceiro retriever (ex: FAISS) ao Modular RAG | +5 pontos |
| LangFuse com feedback humano registrado (score manual) | +3 pontos |
| Estudo de caso real de Advanced RAG na área jurídica/policial | +5 pontos |
| Análise estatística de latência com gráfico (boxplot/histograma) | +3 pontos |

---

## Observações para o Professor

**Pontos críticos a verificar:**

1. **HyDE-Lite:** Verificar se o aluno compreendeu o risco de alucinação e documentou algum caso
2. **Cross-encoder:** Verificar se o aluno entendeu a diferença fundamental bi-encoder vs cross-encoder (não apenas decorar)
3. **Modularidade:** O ponto central é que o pipeline NÃO deve depender de implementações concretas — verificar se a injeção de dependência foi compreendida
4. **LangFuse:** Se o aluno não teve acesso ao LangFuse cloud, aceitar screenshots de prints do terminal que demonstrem a estrutura de spans

**Referências para correção:**

- Tabela de scores 5D esperada para as 5 queries está em `datasets/corpus_juridico_aula3.json` (campo `resposta_esperada`)
- A melhoria esperada de Advanced RAG vs Naive RAG é de 15-40% nas dimensões de Precisão e Fundamentação

---

## Referências para os Alunos

GAO, Y. et al. **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997, 2023.

MA, X. et al. **Query Rewriting for Retrieval-Augmented Large Language Models**. arXiv:2305.14283, 2023.

NOGUEIRA, R.; CHO, K. **Passage Re-ranking with BERT**. arXiv:1901.04085, 2019.

LANGFUSE. **LangFuse Documentation**. Disponível em: <https://langfuse.com/docs>. Acesso em: abr. 2026.
