"""
reranker.py — optional cross-encoder reranking stage.

Bi-encoder retrieval (vector search) scores query and document independently,
which is fast but lossy. A cross-encoder reads (query, document) pairs jointly
and produces a much sharper relevance score — the standard pattern is:

    retrieve top-N with hybrid search (cheap, high recall)
        → rerank with cross-encoder (expensive, high precision)
            → keep top-k

The model is lazy-loaded on first use so the simulation runs fine without
sentence-transformers installed (reranking silently no-ops with a warning).

Default model: BAAI/bge-reranker-base (~1.1GB download on first use).
"""

import logging

logger = logging.getLogger("lab_sim")

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self._model = None
        self._unavailable = False

    def _load(self):
        if self._model is not None or self._unavailable:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.warning(
                f"Reranker unavailable ({e}) — falling back to fusion order. "
                "Install with: pip install sentence-transformers"
            )
            self._unavailable = True

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[str]:
        """
        Score (query, doc) pairs jointly and return the top_k documents in
        descending relevance order. Falls back to the input order (truncated)
        if the model can't be loaded.
        """
        if not documents:
            return []
        self._load()
        if self._model is None:
            return documents[:top_k]
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
