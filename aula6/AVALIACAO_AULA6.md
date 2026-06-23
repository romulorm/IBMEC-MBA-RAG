# Critérios de Avaliação — Aula 6
## Indexação Avançada: Hierarchical Indexing, RAPTOR e HyDE
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 6 de 12 | **Carga:** 5h | **Peso na nota final:** 8,3% (1 de 12 aulas)

---

## Visão Geral dos Entregáveis

| # | Entregável | Peso | Ferramenta de Avaliação |
|---|---|---|---|
| E1 | Parent-Child Retriever implementado e comparado | 25 pts | Log AutoMerging + delta score |
| E2 | Visualização RAPTOR — árvore de clusters UMAP 2D | 25 pts | PNG + código de geração |
| E3 | Pipeline HyDE com gap semântico visualizado | 25 pts | PNG do UMAP + delta score |
| E4 | Dashboard RAGAS comparando 4 técnicas | 25 pts | relatorio_final_aula6.json |
| **Total** | | **100 pts** | |

---

## E1 — Parent-Child Retriever (25 pontos)

### Rubrica Detalhada

#### 1.1 Configuração do HierarchicalNodeParser (8 pts)

| Indicador | Pontos |
|---|---|
| chunk_sizes com ratio ≥ 1:4 (ex: [512, 128]) | 3 pts |
| Separação correta entre leaf_nodes e root_nodes | 2 pts |
| Docstore configurado com todos os nós (pais + filhos) | 3 pts |

**Verificação rápida:**
```python
# Professor executa após o aluno:
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
leaf = get_leaf_nodes(all_nodes)
root = get_root_nodes(all_nodes)
assert len(leaf) / len(root) >= 2, "Ratio insuficiente"
print(f"Ratio filhos/pais: {len(leaf)/len(root):.1f}")
```

#### 1.2 AutoMergingRetriever funcional (10 pts)

| Indicador | Pontos |
|---|---|
| AutoMergingRetriever instanciado com simple_ratio_thresh | 3 pts |
| Log de merging visível (verbose=True) ou evidência de merging | 4 pts |
| Query retorna nós pai (chunks maiores) para ao menos 1 query | 3 pts |

#### 1.3 Comparação com chunking plano (7 pts)

| Indicador | Pontos |
|---|---|
| Flat retriever implementado com o mesmo corpus | 2 pts |
| Comparação de score médio: Parent-Child ≥ flat | 3 pts |
| Análise qualitativa: por que o contexto do pai é melhor | 2 pts |

---

## E2 — Visualização RAPTOR (25 pontos)

### Rubrica Detalhada

#### 2.1 Pipeline RAPTOR completo (12 pts)

| Indicador | Pontos |
|---|---|
| Embeddings gerados para todos os chunks | 2 pts |
| UMAP configurado com n_components=10 para clustering | 3 pts |
| GMM com número ótimo de clusters via BIC | 4 pts |
| Resumos gerados para cada cluster (LLM ou fallback) | 3 pts |

**Verificação:**
```python
assert "nivel_0" in raptor_idx
assert "nivel_1" in raptor_idx
assert len(raptor_idx["nivel_1"]) >= 2, "Mínimo 2 clusters"
print(f"Árvore RAPTOR: {len(raptor_idx['nivel_0'])} chunks + {len(raptor_idx['nivel_1'])} resumos")
```

#### 2.2 Visualização 2D com matplotlib (8 pts)

| Indicador | Pontos |
|---|---|
| UMAP 2D gerado corretamente | 3 pts |
| Clusters coloridos por grupo | 2 pts |
| Arquivo PNG salvo (raptor_clusters.png) | 2 pts |
| Labels dos documentos visíveis no gráfico | 1 pt |

#### 2.3 Collapsed Tree Search funcional (5 pts)

| Indicador | Pontos |
|---|---|
| Busca em todos os níveis da árvore | 2 pts |
| Query abrangente prefere nível 1-2 vs query específica que prefere nível 0 | 3 pts |

---

## E3 — Pipeline HyDE (25 pontos)

### Rubrica Detalhada

#### 3.1 Geração do documento hipotético (10 pts)

| Indicador | Pontos |
|---|---|
| Documento hipotético gerado em linguagem técnico-jurídica | 4 pts |
| Hipotético tem conteúdo diferente da query original | 3 pts |
| Fallback implementado para quando LLM offline | 3 pts |

#### 3.2 Comparação HyDE vs Busca Direta (10 pts)

| Indicador | Pontos |
|---|---|
| Score médio HyDE ≥ score médio direto (ou análise do porquê não) | 5 pts |
| 3+ queries testadas | 3 pts |
| Delta de score calculado e reportado | 2 pts |

#### 3.3 Visualização geométrica do gap semântico (5 pts)

| Indicador | Pontos |
|---|---|
| UMAP 2D com corpus, queries e hipotéticos plotados | 3 pts |
| Setas mostrando a "tradução" query → hipotético | 2 pts |

**Verificação:**
```python
# Verificar que o gap semântico reduziu
delta_medio = metricas["melhoria_media_score"]
print(f"Melhoria HyDE: {delta_medio:+.4f}")
# Aceitar mesmo se negativo, desde que a análise justifique (corpus pequeno)
```

---

## E4 — Dashboard RAGAS (25 pontos)

### Rubrica Detalhada

#### 4.1 Coleta de dados para as 4 técnicas (10 pts)

| Indicador | Pontos |
|---|---|
| Naive RAG, Parent-Child, RAPTOR e HyDE representados | 4 pts |
| 4 métricas RAGAS calculadas (CP, CR, F, AR) | 4 pts |
| Dataset com ≥ 5 queries de benchmark | 2 pts |

#### 4.2 Dashboard visual (8 pts)

| Indicador | Pontos |
|---|---|
| 4 gráficos (um por métrica) ou 1 gráfico consolidado | 4 pts |
| Linha de baseline Naive RAG visível | 2 pts |
| Arquivo PNG salvo (comparacao_ragas_aula6.png) | 2 pts |

#### 4.3 Análise escrita (7 pts)

| Indicador | Pontos |
|---|---|
| Identifica qual técnica é melhor para cada métrica | 3 pts |
| Recomendação para corpus jurídico com justificativa RAGAS | 2 pts |
| Identifica pelo menos 1 limitação de cada técnica | 2 pts |

---

## Pontuação Total e Conceitos

| Pontos | Conceito | Descrição |
|---|---|---|
| 90–100 | A — Excepcional | Todos os entregáveis concluídos com análise crítica aprofundada |
| 75–89 | B — Proficiente | 3 de 4 entregáveis completos; análise adequada |
| 60–74 | C — Satisfatório | E4 (RAGAS) + pelo menos 1 técnica implementada completamente |
| 40–59 | D — Insuficiente | Código incompleto; análise superficial |
| 0–39 | F — Reprovado | Entregáveis ausentes ou não executáveis |

---

## Critérios de Aprovação Obrigatórios (Veto)

1. **E4 (RAGAS) obrigatório:** o dashboard comparativo deve estar presente e executável
2. **Melhoria mínima:** ao menos uma técnica deve mostrar melhoria ≥ 5% vs Naive RAG em alguma métrica RAGAS
3. **Arquivo relatorio_final_aula6.json:** deve existir e ser JSON válido com scores das 4 técnicas

---

## Protocolo do Professor (15 min por aluno)

| Minuto | Ação |
|---|---|
| 00–03 | Verificar estrutura de pastas e arquivos gerados; validar JSONs |
| 03–06 | Revisar LAB1: executar 2 células críticas (indexação + merging) |
| 06–09 | Ver PNG do RAPTOR; verificar se clusters fazem sentido temático |
| 09–12 | Ver PNG do HyDE gap semântico; checar delta de score |
| 12–14 | Ver dashboard RAGAS; discutir recomendação do aluno |
| 14–15 | Perguntas de verificação (nível conforme desempenho) |

---

## Perguntas de Verificação

### Nível Básico (C)
1. Por que o Parent-Child usa dois tamanhos de chunk diferentes?
2. O que é um cluster no RAPTOR e o que ele representa?
3. O que o HyDE faz antes de buscar no corpus?

### Nível Intermediário (B)
1. Se o AutoMergingRetriever tem threshold 0.3 e apenas 1 de 5 filhos de um pai foram recuperados, o que acontece?
2. Por que o GMM é preferível ao K-means no RAPTOR?
3. Em que situação o HyDE pode piorar o retrieval?

### Nível Avançado (A)
1. Como você combinaria Parent-Child e HyDE em um único pipeline? Qual módulo executa primeiro?
2. Se o corpus tem 500 acórdãos mas as queries são sempre sobre artigos específicos de leis, qual técnica você não usaria e por quê?
3. Como você mediria o impacto da qualidade do LLM (temperatura, modelo) na qualidade do HyDE?

---

## Feedback Padrão por Erro Comum

| Erro | Feedback Construtivo |
|---|---|
| Indexou pais E filhos no vector store | "O vector store deve conter APENAS os filhos (leaf_nodes). Os pais ficam no docstore para serem recuperados pelo AutoMergingRetriever" |
| RAPTOR com corpus de 3 documentos | "RAPTOR precisa de pelo menos 50-100 chunks para produzir clusters coerentes. Tente com um corpus maior ou reduza o número de clusters com k=2" |
| HyDE sem fallback para LLM offline | "Sempre implemente um hipotético pré-gerado como fallback — o Colab frequentemente não tem acesso ao vLLM" |
| RAGAS com apenas Naive RAG | "O dashboard deve comparar as 4 técnicas. Use os valores simulados baseados na literatura se o tempo for insuficiente" |
| Gráfico sem linha de baseline | "A linha de baseline do Naive RAG é essencial para mostrar a melhoria de cada técnica. Adicione `ax.axhline(baseline_val)`" |

---

## Referências para Avaliação (ABNT)

SARTHI, P. et al. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**. In: *ICLR*, 2024. arXiv:2401.18059.

GAO, L. et al. **Precise Zero-Shot Dense Retrieval without Relevance Labels**. In: *ACL*, 2023. arXiv:2212.10496.

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. In: *EACL*, 2024. arXiv:2309.15217.

LLAMAINDEX. **Auto-Merging Retriever**. LlamaIndex Documentation, 2024. Disponível em: <https://docs.llamaindex.ai>. Acesso em: abr. 2026.
