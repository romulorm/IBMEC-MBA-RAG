# Avaliação — Aula 5: Avaliação e Observabilidade — RAGAS, DeepEval e LangFuse Avançado
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

---

## Critérios Gerais

| Critério | Peso |
|---|---|
| Funcionalidade técnica (código executa sem erros) | 35% |
| Resultados RAGAS acima das metas mínimas | 30% |
| Análise crítica e diagnóstico de falhas | 25% |
| Aplicação ao domínio jurídico/segurança pública | 10% |

---

## Metas Mínimas RAGAS (obrigatórias para aprovação)

| Métrica | Meta | Consequência se não atingida |
|---|---|---|
| **Faithfulness** | ≥ 0.80 | Reprovação no LAB2 — risco de alucinação jurídica |
| **Answer Relevancy** | ≥ 0.75 | Penalização de 50% na nota do LAB2 |
| **Context Recall** | ≥ 0.70 | Reprovação no LAB2 — retriever insuficiente |
| **Context Precision** | ≥ 0.70 | Penalização de 50% na nota do LAB2 |

> **Nota:** Atingir pelo menos 2 das 4 métricas acima das metas é requisito mínimo para aprovação na aula.

---

## Rubricas por Laboratório

### LAB1 — Dataset de Avaliação com Ground-Truth (15%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Dataset com 50 pares QA exportado em formato JSON e CSV. Todos os campos presentes: `question`, `answer`, `contexts`, `ground_truth`. Ao menos 5 tipos jurídicos diferentes representados. Geração assistida por vLLM com revisão manual de ao menos 10 pares. |
| **Bom (70–89%)** | Dataset com 30–49 pares. Todos os campos presentes mas sem diversidade de tipos jurídicos. Geração automática sem revisão manual. |
| **Regular (50–69%)** | Dataset com 15–29 pares ou campos incompletos (ex.: sem ground_truth). |
| **Insuficiente (<50%)** | Dataset com menos de 15 pares ou arquivo não exportado. |

### LAB2 — RAGAS Baseline no Naive RAG (25%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | As 4 métricas calculadas. Faithfulness ≥ 0.80 e ao menos mais 2 métricas nas metas. Resultados registrados no LangFuse. Análise narrativa de ao menos 3 casos onde o pipeline falhou. |
| **Bom (70–89%)** | As 4 métricas calculadas. Ao menos 2 métricas nas metas. Sem análise qualitativa detalhada. |
| **Regular (50–69%)** | Apenas 2–3 métricas calculadas. Sem registro LangFuse. |
| **Insuficiente (<50%)** | RAGAS não executado ou com erros críticos. |

### LAB3 — RAGAS + LangFuse Scores API (20%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Pipeline completo: execução RAG → cálculo RAGAS → envio automático ao LangFuse Scores API. Trace visível no dashboard com 4 scores por execução. Callback ou decorator implementado para integração automática. |
| **Bom (70–89%)** | Integração funcional mas manual (scores enviados após cálculo separado). Trace visível com ao menos 2 scores. |
| **Regular (50–69%)** | RAGAS calculado mas scores enviados ao LangFuse de forma incorreta ou incompleta. |
| **Insuficiente (<50%)** | Integração não implementada ou LangFuse não acessível. |

### LAB4 — DeepEval: Testes Unitários (20%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | 5 testes implementados: Faithfulness (th=0.80), AnswerRelevancy (th=0.75), Hallucination (th_max=0.20), Toxicity (th_max=0.10), Bias (th_max=0.15). Ao menos 3 testes passando. Relatório de falhas com diagnóstico. Integração com `pytest` demonstrada. |
| **Bom (70–89%)** | 4–5 testes implementados, 2–3 passando. Sem integração pytest. |
| **Regular (50–69%)** | 2–3 testes implementados com ao menos 1 passando. |
| **Insuficiente (<50%)** | Menos de 2 testes ou nenhum passando. |

### LAB5 — Dashboard Naive vs Advanced RAG (15%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Dashboard Matplotlib com 4 subplots: uma barra por métrica comparando Naive vs Advanced. Tabela-resumo com delta (melhoria percentual). CSV exportado. Conclusão narrativa identificando qual pipeline é mais adequado para o corpus jurídico e por quê. |
| **Bom (70–89%)** | Dashboard com ao menos 2 métricas comparadas. Sem tabela-resumo ou CSV. Conclusão superficial. |
| **Regular (50–69%)** | Apenas tabela de texto sem gráficos, ou gráfico com 1 métrica. |
| **Insuficiente (<50%)** | Lab não executado. |

### LAB6 — Análise de Erros de Faithfulness (5%)

| Nível | Critério |
|---|---|
| **Excelente (90–100%)** | Ao menos 3 queries com Faithfulness < 0.70 identificadas. Para cada uma: afirmações problemáticas destacadas, causa classificada (alucinação paramétrica / contexto insuficiente / prompt fraco), solução proposta e resultado após correção. |
| **Bom (70–89%)** | 2–3 queries analisadas com causa identificada mas sem execução da correção. |
| **Regular (50–69%)** | 1 query analisada superficialmente. |
| **Insuficiente (<50%)** | Lab não executado. |

---

## Entregáveis

- Notebooks executados (células com output visível) para os 6 Labs
- Dataset JSON (`corpus_avaliacao_aula5.json`) com 50 pares QA preenchidos (campo `answer` gerado pelo pipeline)
- Screenshot do dashboard LangFuse mostrando as 4 métricas RAGAS em ao menos 1 trace
- Relatório de análise de erros (LAB6) em formato Markdown ou célula de texto no notebook

**Prazo:** Conforme calendário do MBA  
**Formato de entrega:** Google Colab + link compartilhado (acesso de visualização)
