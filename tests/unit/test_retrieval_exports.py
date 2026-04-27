from ragaroo.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ProprietaryEmbedder,
    SentenceTransformerEmbedder,
    SentenceTransformerSparseEmbedder,
    SparseRetriever,
)


def test_retrieval_package_exports_public_retrievers():
    assert BM25Retriever is not None
    assert DenseRetriever is not None
    assert HybridRetriever is not None
    assert ProprietaryEmbedder is not None
    assert SentenceTransformerEmbedder is not None
    assert SentenceTransformerSparseEmbedder is not None
    assert SparseRetriever is not None
