"""
02_parent_child.py - Hierarchical Indexing / Parent-Child (#T07) com Haystack.

Problema: chunk grande perde precisao na busca; chunk pequeno perde contexto na
resposta. Solucao Parent-Child: quebrar o documento em dois niveis -
  - FILHOS (pequenos) -> usados na BUSCA (precisao)
  - PAIS (grandes)    -> usados na RESPOSTA (contexto)
Busca-se nos filhos; se varios filhos do mesmo pai aparecem, o AutoMergingRetriever
"sobe" para o pai inteiro (mais contexto para o LLM).

Stack: HierarchicalDocumentSplitter + AutoMergingRetriever (Haystack) + Ollama
(embeddings) + OpenSearch (busca dos filhos) + Groq (resposta).

Precisa de OpenSearch, Ollama e Groq.

Uso:
    python 02_parent_child.py --recriar
    python 02_parent_child.py --pergunta "o que e cadeia de custodia?"
    python 02_parent_child.py --pai 200 --filho 50 --top-k 6 --threshold 0.5
"""

import argparse

import requests
from haystack import Pipeline
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.retrievers import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)

import _comum


def construir_responder(indice, pai=200, filho=50, top_k=6, threshold=0.5, recriar=False):
    """Monta o indice hierarquico e devolve uma funcao responder(pergunta)."""
    os_cfg = _comum.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    if recriar:
        try:
            requests.delete(f"{os_cfg['url']}/{indice}", auth=auth, timeout=10)
        except Exception:
            pass

    # 1) Quebra cada documento em niveis (pais e filhos).
    splitter = HierarchicalDocumentSplitter(
        block_sizes={pai, filho}, split_overlap=0, split_by="word")
    todos_nos = splitter.run(documents=_comum.documentos_haystack())["documents"]
    folhas = [d for d in todos_nos if not d.meta.get("__children_ids")]

    # 2) Guarda TODA a hierarquia (pais + filhos) para o auto-merging consultar.
    store_hierarquia = InMemoryDocumentStore()
    store_hierarquia.write_documents(todos_nos, policy=DuplicatePolicy.OVERWRITE)

    # 3) Indexa SO os filhos (com embedding) no OpenSearch, para a busca.
    store_os = _comum.abrir_store(indice)
    embedder = _comum.doc_embedder()
    folhas_emb = embedder.run(documents=folhas)["documents"]
    store_os.write_documents(folhas_emb, policy=DuplicatePolicy.OVERWRITE)

    # 4) Pipeline de busca: embedding da query -> recupera filhos -> auto-merging.
    busca = Pipeline()
    busca.add_component("embedder", _comum.text_embedder())
    busca.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store_os, top_k=top_k))
    busca.add_component("merge", AutoMergingRetriever(document_store=store_hierarquia, threshold=threshold))
    busca.connect("embedder.embedding", "retriever.query_embedding")
    busca.connect("retriever.documents", "merge.documents")

    cliente, modelo = _comum.groq_client()

    def responder(pergunta):
        docs = busca.run({"embedder": {"text": pergunta}})["merge"]["documents"]
        contextos = [d.content for d in docs]
        resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
        return resposta, contextos

    return responder


def main():
    parser = argparse.ArgumentParser(description="Parent-Child / Hierarchical (#T07).")
    parser.add_argument("--indice", default="aula6_parent_child", help="indice no OpenSearch")
    parser.add_argument("--pergunta", default=None, help="pergunta (senao roda exemplos do corpus)")
    parser.add_argument("--pai", type=int, default=200, help="tamanho do bloco PAI (palavras)")
    parser.add_argument("--filho", type=int, default=50, help="tamanho do bloco FILHO (palavras)")
    parser.add_argument("--top-k", type=int, default=6, help="quantos filhos buscar")
    parser.add_argument("--threshold", type=float, default=0.5, help="fracao de filhos p/ subir ao pai")
    parser.add_argument("--recriar", action="store_true", help="recria o indice")
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  PARENT-CHILD / HIERARCHICAL (#T07) - Aula 6")
    print("=" * 60)
    print(f"Indice: {args.indice} | pai={args.pai} filho={args.filho} top_k={args.top_k} threshold={args.threshold}")
    print("Construindo o indice hierarquico (pode demorar - gera embeddings)...")

    responder = construir_responder(
        args.indice, args.pai, args.filho, args.top_k, args.threshold, args.recriar)

    _, perguntas = _comum.carregar_corpus()
    if args.pergunta:
        alvos = [args.pergunta]
    else:
        alvos = [p["pergunta"] for p in perguntas][:3]

    for q in alvos:
        print("\n" + "-" * 55)
        print(f"Pergunta: {q}")
        resposta, contextos = responder(q)
        print(f"Trechos usados: {len(contextos)} (tamanhos: {[len(c) for c in contextos]})")
        print(f"Resposta: {resposta}")


if __name__ == "__main__":
    main()
