"""Listings 3.5.1, 3.5.4, 3.5.5 - Embeddings, filtered similarity search, reranking.

By default this runs offline with deterministic fake embeddings and an in-memory vector store.
Set OPENAI_API_KEY (and COHERE_API_KEY for reranking) and pass live=True to use the book's models.
Note: the book's dict-style `filter=` is vector-store specific. InMemoryVectorStore takes a
callable, so the offline listing shows that form; Chroma/pgvector/etc. accept the dict form.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

DOCS = [
    Document("Damaged items can be returned within 30 days of delivery.",
             metadata={"department": "support", "language": "en"}),
    Document("Refunds above 5000 rupees require human approval.",
             metadata={"department": "support", "language": "en"}),
    Document("Our office is closed on public holidays.",
             metadata={"department": "hr", "language": "en"}),
    Document("Les articles endommagés peuvent être retournés sous 30 jours.",
             metadata={"department": "support", "language": "fr"}),
]


def make_embeddings(live: bool = False):
    if live:  # --- Listing 3.5.1 (live) ---
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return DeterministicFakeEmbedding(size=64)


def embed_examples(embeddings):  # Listing 3.5.1
    return embeddings.embed_documents([
        "How do I reset my password?",
        "I forgot my login details",
        "What is your refund policy?",
    ])  # each vector is a list of floats, e.g. 1536 of them with text-embedding-3-small


def build_vectorstore(embeddings) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embeddings)
    store.add_documents(DOCS)
    return store


def filtered_search(vectorstore):  # Listing 3.5.4
    return vectorstore.similarity_search(
        query="refund policy for damaged items",
        k=5,
        filter=lambda d: d.metadata.get("department") == "support"
        and d.metadata.get("language") == "en",
    )


def build_reranking_retriever(vectorstore, compressor=None):  # Listing 3.5.5
    from langchain_classic.retrievers import ContextualCompressionRetriever

    if compressor is None:  # live: Cohere reranker, as in the book
        from langchain_cohere import CohereRerank
        compressor = CohereRerank(model="rerank-v3.5", top_n=4)
    base = vectorstore.as_retriever(search_kwargs={"k": 20})
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)


if __name__ == "__main__":
    emb = make_embeddings()
    print(len(embed_examples(emb)), "vectors of", len(embed_examples(emb)[0]), "floats")
    store = build_vectorstore(emb)
    for d in filtered_search(store):
        print("-", d.page_content)
