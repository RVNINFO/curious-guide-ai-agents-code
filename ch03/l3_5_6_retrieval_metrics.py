"""Listing 3.5.6 - Measuring whether RAG works: recall@k and MRR. Run:  python -m ch03.l3_5_6_retrieval_metrics"""


def recall_at_k(retrieved: list[str], relevant: set[str],
                k: int) -> float:
    """Share of the relevant chunks that appear in the
    top k results."""
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str],
                    relevant: set[str]) -> float:
    """1 / rank of the first relevant chunk;
    0 if none was retrieved."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


# Each question has the chunks a human says must be found,
# and what retrieval returned.
cases = [
    {"relevant": {"c12"},
     "retrieved": ["c07", "c12", "c30", "c02", "c19"]},
    {"relevant": {"c41"},
     "retrieved": ["c41", "c05", "c33", "c18", "c27"]},
    {"relevant": {"c88"},
     "retrieved": ["c03", "c15", "c22", "c09", "c61"]},
]

k = 5
recall = sum(recall_at_k(c["retrieved"], c["relevant"], k)
             for c in cases) / len(cases)
mrr = sum(reciprocal_rank(c["retrieved"], c["relevant"])
          for c in cases) / len(cases)

print(f"recall@{k} = {recall:.2f}   MRR = {mrr:.2f}")
# recall@5 = 0.67   MRR = 0.50
