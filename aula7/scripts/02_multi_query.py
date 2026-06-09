"""
02_multi_query.py - Multi-Query RAG (#T10).

Problema (vocabulary mismatch): a pergunta do usuario pode usar palavras diferentes
dos documentos. Multi-Query gera VARIAS reformulacoes da pergunta (com vocabulario
diferente), busca por TODAS e junta os resultados (removendo duplicados). Assim
aumenta a chance de achar o documento certo.

Fluxo: pergunta -> LLM gera N variacoes -> busca cada uma -> dedup por id -> resposta.

Precisa de OpenSearch (indice do TCU), Ollama e Groq.

Uso:
    python 02_multi_query.py
    python 02_multi_query.py --pergunta "o orgao pode contratar sem licitacao?" --n 4
"""

import argparse

import _comum

QUERY_EXEMPLO = "quando o gestor pode ser multado pelo tribunal de contas?"


def construir_responder(indice=None, top_k=5, n_variacoes=4):
    """Devolve responder(pergunta) -> dict (Multi-Query RAG)."""
    indice = indice or _comum.INDICE_TCU
    store = _comum.abrir_store(indice)
    pipe = _comum.montar_busca(store, top_k)
    cliente, modelo = _comum.groq_client()

    def responder(pergunta):
        variacoes = _comum.gerar_variacoes(cliente, modelo, pergunta, n=n_variacoes)
        consultas = [pergunta] + variacoes
        listas = [_comum.buscar(pipe, c) for c in consultas]
        docs = _comum.dedup_por_id(listas, top_k)
        contextos = [d.content for d in docs]
        resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, contextos)
        return {"resposta": resposta, "contextos": contextos,
                "ids": [d.meta.get("id_original") for d in docs],
                "variacoes": variacoes, "n_buscas": len(consultas), "n_llm": 2}

    return responder


def main():
    parser = argparse.ArgumentParser(description="Multi-Query RAG (#T10).")
    parser.add_argument("--pergunta", default=QUERY_EXEMPLO)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--n", type=int, default=4, help="numero de variacoes")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    _comum.carregar_env()
    print("=" * 60)
    print("  MULTI-QUERY RAG (#T10) - Aula 7")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}")
    responder = construir_responder(args.indice, args.top_k, args.n)
    r = responder(args.pergunta)
    print("\nVariacoes geradas:")
    for i, v in enumerate(r["variacoes"], 1):
        print(f"  {i}. {v}")
    print(f"\nDocumentos recuperados ({len(r['ids'])}): {r['ids']}")
    print(f"Custo: {r['n_buscas']} buscas + {r['n_llm']} chamadas LLM")
    print(f"\nResposta:\n{r['resposta']}")


if __name__ == "__main__":
    main()
