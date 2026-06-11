"""
_componentes.py - Componentes Haystack customizados da Aula 8 (CRAG).

Montam o pipeline CRAG inteiro DENTRO de um Pipeline Haystack, usando o
ConditionalRouter para o roteamento condicional (local / fusao / web). Assim, com
a auto-instrumentacao do LangFuse, cada etapa vira um span do mesmo trace.

  - AvaliarRota   : avalia os documentos locais (LLM-as-Judge), calcula o score
                    medio e decide a rota CRAG.
  - BuscaWeb      : faz o web search (Tavily com fallback offline) - so executa
                    quando o ConditionalRouter envia a query para ela.
  - MontarContexto: junta o contexto final conforme a rota (so local / local+web /
                    so web).
"""

from typing import List, Optional

from haystack import Document, component

import _comum


@component
class AvaliarRota:
    """Avalia relevancia (LLM-as-Judge) e decide a rota CRAG (local/fusao/web)."""

    @component.output_types(documents=List[Document], question=str, rota=str, score=float)
    def run(self, documents: List[Document], question: str):
        cliente, modelo = _comum.groq_client()
        media, _ = _comum.avaliar_documentos(cliente, modelo, question, documents)
        return {"documents": documents, "question": question,
                "rota": _comum.decidir_rota(media), "score": media}


@component
class BuscaWeb:
    """Web search (Tavily com fallback offline). So roda quando recebe a query."""

    def __init__(self, max_results=3):
        self.max_results = max_results

    @component.output_types(web_docs=List[Document])
    def run(self, query: str):
        resultados = _comum.web_search(query, max_results=self.max_results)
        return {"web_docs": _comum.web_para_documents(resultados)}


@component
class MontarContexto:
    """Monta o contexto final de acordo com a rota CRAG."""

    @component.output_types(documents=List[Document])
    def run(self, documents_local: List[Document], rota: str,
            web_docs: Optional[List[Document]] = None):
        web_docs = web_docs or []
        if rota == "local":
            docs = documents_local
        elif rota == "web":
            docs = web_docs
        else:  # fusao
            docs = list(documents_local) + list(web_docs)
        return {"documents": docs}
