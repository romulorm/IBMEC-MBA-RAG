"""
04_colbert_ragatouille.py - ColBERT (late interaction) via RAGatouille vs busca densa.

ColBERT guarda um vetor por TOKEN e pontua por MaxSim (precisao alta). A RAGatouille
encapsula o ColBERTv2 numa API simples. Aqui indexamos o corpus com ColBERT e comparamos
os resultados com a busca DENSA (bi-encoder, Ollama+OpenSearch) para a mesma query.

Precisa: pip install ragatouille  (puxa torch + faiss + baixa o colbertv2 ~440MB)
e o indice denso: python 01_indexar.py

Uso:
    python 04_colbert_ragatouille.py --pergunta "prazo de recurso de apelacao"
    python 04_colbert_ragatouille.py --pergunta "..." --top-k 5
"""

import argparse

import _comum

_comum.carregar_env()

INDEX_NAME = "aula11_colbert"


def resultados_colbert(pergunta, top_k):
    from ragatouille import RAGPretrainedModel

    docs = _comum.carregar_corpus()
    textos = [d["texto"] for d in docs]
    ids = [d["id"] for d in docs]

    print("Carregando ColBERTv2 e indexando (1a vez baixa o modelo)...")
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RAG.index(collection=textos, document_ids=ids, index_name=INDEX_NAME,
              max_document_length=256, split_documents=False)
    return RAG.search(pergunta, k=top_k)


def main():
    parser = argparse.ArgumentParser(description="ColBERT/RAGatouille vs busca densa.")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  ColBERT (RAGatouille) vs BUSCA DENSA - Aula 11")
    print("=" * 60)
    print(f"Pergunta: {args.pergunta}\n")

    # 1) busca densa (bi-encoder) - reusa o indice da aula
    store = _comum.abrir_store()
    if store.count_documents() == 0:
        print("[ATENCAO] indice denso vazio. Rode antes: python 01_indexar.py")
        return
    densos = _comum.buscar(_comum.montar_busca(store, args.top_k), args.pergunta)
    print("DENSA (bi-encoder):")
    for i, d in enumerate(densos, 1):
        print(f"  {i}. {d.score:.3f}  {d.meta.get('id_original')}  {d.content[:60]}")

    # 2) ColBERT (late interaction)
    try:
        res = resultados_colbert(args.pergunta, args.top_k)
    except ImportError:
        print("\n[ERRO] instale: pip install ragatouille")
        return
    print("\nColBERT (late interaction / MaxSim):")
    for i, r in enumerate(res, 1):
        print(f"  {i}. {r.get('score'):.3f}  {r.get('document_id')}  {str(r.get('content'))[:60]}")

    print("\nLeitura: o ColBERT tende a casar melhor termos especificos (granularidade por "
          "token), ao custo de um indice maior. Compare a ordem/precisao das duas listas.")


if __name__ == "__main__":
    main()
