"""
consulta.py - RAG de consulta (busca), roteado por storage, com OBSERVABILIDADE Langfuse.

destino:
  - 'opensearch' (ou 'auto'): PIPELINE Haystack (embed -> retrieve -> prompt -> Groq),
    instrumentado por LangfuseConnector (auto-trace de TODA a busca: embedding, recuperacao
    e geracao no mesmo trace).
  - 'grafo'                  : LightRAG (modo hibrido), rastreado com @observe do Langfuse.

A observabilidade so liga se LANGFUSE_PUBLIC_KEY/SECRET_KEY existirem (.env). O preparo do
tracing (HAYSTACK_CONTENT_TRACING_ENABLED + LANGFUSE_HOST) e feito em app/__init__.py,
ANTES de qualquer import do Haystack (exigencia da auto-instrumentacao).
"""

from . import config, indexacao
from .log import obter_logger

log = obter_logger(__name__)

# Template Jinja do PromptBuilder: recebe os 'documents' (do retriever) e a 'pergunta'.
PROMPT_TMPL = """Voce e um assistente juridico. Responda APENAS com base nos trechos abaixo, de forma objetiva. Se nao constar, diga que nao consta.

Trechos:
{% for d in documents %}- {{ d.content }}
{% endfor %}
Pergunta: {{ pergunta }}
Resposta:"""


# ---------------------------------------------------------------------------
# Busca no OpenSearch (pipeline Haystack + LangfuseConnector)
# ---------------------------------------------------------------------------
def _pipeline_opensearch(top_k):
    from haystack import Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.utils import Secret
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever

    base_url, modelo_emb = config.config_ollama()
    api_key, gmodelo, groq_base = config.config_groq()
    store = indexacao._store_opensearch()

    pipe = Pipeline()
    # LangfuseConnector: ativa o tracing de toda a execucao deste pipeline (1 trace por busca)
    if config.langfuse_configurado():
        from haystack_integrations.components.connectors.langfuse import LangfuseConnector
        pipe.add_component("tracer", LangfuseConnector("busca-rag-aula12"))
    pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_emb, url=base_url))
    pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
    pipe.add_component("prompt", PromptBuilder(template=PROMPT_TMPL, required_variables="*"))
    pipe.add_component("llm", OpenAIGenerator(
        api_key=Secret.from_token(api_key), model=gmodelo, api_base_url=groq_base,
        generation_kwargs={"temperature": 0.2, "max_tokens": 500}))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe


def consultar_opensearch(pergunta, top_k):
    log.info("Consulta OpenSearch (top_k=%d): %r", top_k, pergunta)
    pipe = _pipeline_opensearch(top_k)
    saida = pipe.run({"embedder": {"text": pergunta}, "prompt": {"pergunta": pergunta}},
                     include_outputs_from={"retriever"})
    docs = saida["retriever"]["documents"]
    replies = saida["llm"]["replies"]
    resposta = (replies[0] if replies else "").strip()
    log.info("Recuperados %d trecho(s); resposta gerada (Groq)", len(docs))
    if "tracer" in saida and saida["tracer"].get("trace_url"):
        log.info("Langfuse trace (busca): %s", saida["tracer"]["trace_url"])
    fontes = [{"id": d.meta.get("id_original") or d.meta.get("arquivo"),
               "trecho": d.content[:160]} for d in docs]
    return resposta, fontes


# ---------------------------------------------------------------------------
# Busca no grafo (LightRAG) - rastreada com @observe
# ---------------------------------------------------------------------------
def _grafo_raw(pergunta):
    from lightrag import QueryParam

    async def _q():
        rag = await indexacao._criar_lightrag()
        try:
            return await rag.aquery(pergunta, param=QueryParam(mode="hybrid"))
        finally:
            await rag.finalize_storages()

    resposta = indexacao.rodar_async(_q)  # seguro com/sem event loop ativo
    return resposta, [{"id": "grafo", "trecho": "(resposta sintetizada do grafo de conhecimento)"}]


def consultar_grafo(pergunta):
    log.info("Consulta ao GRAFO (LightRAG, modo hybrid): %r", pergunta)
    if config.langfuse_configurado():
        try:
            from langfuse import observe
            # @observe captura input (pergunta) e output (resposta) num trace proprio
            return observe(name="busca-grafo-aula12")(_grafo_raw)(pergunta)
        except Exception as e:
            log.warning("Langfuse (grafo) indisponivel (%s) -> seguindo sem trace", e)
    return _grafo_raw(pergunta)


# ---------------------------------------------------------------------------
# Roteador
# ---------------------------------------------------------------------------
def consultar(pergunta, destino="auto", top_k=5):
    if destino == "grafo":
        resp, fontes = consultar_grafo(pergunta)
        return resp, fontes, "grafo"
    resp, fontes = consultar_opensearch(pergunta, top_k)
    return resp, fontes, "opensearch"
