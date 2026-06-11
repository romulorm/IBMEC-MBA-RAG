"""
02_self_rag.py - Self-RAG (training-free) com os 4 tokens de controle via Groq.

O Self-RAG "de verdade" exige um modelo com fine-tuning especifico (ex.: llama-2-7b
-selfrag). Como usamos a Groq, fazemos uma versao TRAINING-FREE: o LLM EMITE os
tokens de controle por prompting (decisao explicita em JSON), imitando o fluxo:

  [Retrieve] yes/no   -> precisa recuperar documentos?
  [ISREL]    rel/irrel-> cada documento recuperado e relevante? (avaliador 0-1)
  (gera a resposta com os documentos relevantes)
  [ISSUP]    fully/partially/no -> a resposta tem suporte nos trechos?
  [ISUSE]    1-5       -> a resposta e util para a pergunta?

Uso:
    python 02_self_rag.py --pergunta "o gestor pode ser multado pelo TCU?"
    python 02_self_rag.py --pergunta "o que e responsabilidade civil?" --top-k 4
"""

import argparse

import _comum

_comum.carregar_env()

LIMITE_REL = 0.5  # [ISREL]: documento e "relevante" se score do avaliador >= isto


def main():
    parser = argparse.ArgumentParser(description="Self-RAG training-free (Aula 8).")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--indice", default=_comum.INDICE_TCU)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    cliente, modelo = _comum.groq_client()
    pergunta = args.pergunta

    print("=" * 60)
    print("  SELF-RAG (training-free) - Aula 8")
    print("=" * 60)
    print(f"Pergunta: {pergunta}\n")

    # [Retrieve] -------------------------------------------------------------
    retrieve, motivo_r = _comum.token_retrieve(cliente, modelo, pergunta)
    print(f"[Retrieve] = {retrieve}  ({motivo_r})")

    documentos_usados = []
    if retrieve == "yes":
        store = _comum.abrir_store(args.indice)
        pipe = _comum.montar_busca(store, args.top_k)
        recuperados = _comum.buscar(pipe, pergunta)

        # [ISREL] por documento (usa o avaliador LLM-as-Judge) -------------
        print("[ISREL] avaliando relevancia de cada documento:")
        for d in recuperados:
            score, motivo = _comum.avaliar_documento(cliente, modelo, pergunta, d.content)
            rel = "relevant" if score >= LIMITE_REL else "irrelevant"
            print(f"   - {d.meta.get('id_original')}: score={score:.2f} -> {rel}")
            if score >= LIMITE_REL:
                documentos_usados.append(d)
        if not documentos_usados:
            print("   (nenhum documento relevante; respondendo sem contexto)")
    else:
        print("   (conhecimento geral; sem recuperacao)")

    # Geracao ----------------------------------------------------------------
    resposta = _comum.responder_com_contexto(cliente, modelo, pergunta, documentos_usados)
    print(f"\nResposta:\n{resposta}\n")

    # [ISSUP] e [ISUSE] ------------------------------------------------------
    contextos = [d.content for d in documentos_usados]
    issup, motivo_s = _comum.token_issup(cliente, modelo, contextos, resposta)
    isuse, motivo_u = _comum.token_isuse(cliente, modelo, pergunta, resposta)
    print(f"[ISSUP] = {issup}  ({motivo_s})")
    print(f"[ISUSE] = {isuse}/5  ({motivo_u})")

    if issup == "no" and documentos_usados:
        print("\n[ALERTA] resposta SEM suporte nos trechos - revisar (possivel alucinacao).")


if __name__ == "__main__":
    main()
