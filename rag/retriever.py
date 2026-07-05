"""
RAGRetriever — hybrid (vector + BM25) retrieval over ChromaDB with optional
cross-encoder reranking.

Each agent gets its own instance pointing at a different collection,
giving every agent a genuinely private knowledge base.
The shared lab KB is just another instance with collection_name="shared".

Retrieval pipeline (mode="hybrid", the default):

    query ─┬─ vector search (nomic-embed-text, cosine) ─┐
           │                                            ├─ reciprocal rank
           └─ BM25 keyword search (rank_bm25) ──────────┘  fusion (k=60)
                                                             │
                                       optional cross-encoder reranker
                                                             │
                                                          top-k chunks

Why hybrid: vector search captures semantic similarity but misses exact
terms (method names, dataset names, acronyms); BM25 nails exact-term matches
but has no semantic generalization. RRF fuses the two rank lists without
needing to calibrate their incomparable score scales.

The BM25 index lives in memory and is rebuilt from the Chroma collection on
init, so persisted KBs stay consistent across restarts.
"""

import re
import uuid
import logging
import ollama
import chromadb
from chromadb.config import Settings

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # keep vector-only mode working without the dep
    BM25Okapi = None

logger = logging.getLogger("lab_sim")

EMBED_MODEL = "nomic-embed-text"
RRF_K = 60  # standard constant from Cormack et al. (2009)


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Ollama's nomic-embed-text model."""
    embeddings = []
    for text in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        embeddings.append(resp["embedding"])
    return embeddings


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """
    Fuse multiple ranked lists of documents into one.

    Each document scores sum(1 / (k + rank_i)) over the lists it appears in.
    Documents ranked highly by *both* retrievers float to the top; k=60
    dampens the advantage of a single #1 placement.
    """
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class RAGRetriever:
    def __init__(
        self,
        collection_name: str,
        persist_dir: str = "./chroma_db",
        reranker=None,           # optional rag.reranker.CrossEncoderReranker
        fetch_multiplier: int = 4,  # candidates fetched per retriever = k * this
    ):
        self.collection_name = collection_name
        self.reranker = reranker
        self.fetch_multiplier = fetch_multiplier
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory BM25 corpus, rebuilt from the persisted collection.
        self._docs: list[str] = []
        self._bm25 = None
        self._rebuild_bm25_from_collection()

    # ── BM25 index management ─────────────────────────────────────────────

    def _rebuild_bm25_from_collection(self) -> None:
        if self.collection.count() == 0:
            return
        try:
            existing = self.collection.get(include=["documents"])
            self._docs = existing["documents"] or []
            self._refresh_bm25()
        except Exception as e:
            logger.error(f"BM25 rebuild error ({self.collection_name}): {e}")

    def _refresh_bm25(self) -> None:
        if BM25Okapi is None:
            if self._docs:
                logger.warning("rank_bm25 not installed — hybrid mode degrades to vector-only")
            return
        if self._docs:
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs])

    # ── Indexing ──────────────────────────────────────────────────────────

    def index(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """Embed and store a list of text chunks (updates both indexes)."""
        if not texts:
            return
        try:
            embeddings = _embed(texts)
            ids = [str(uuid.uuid4()) for _ in texts]
            metadatas = metadatas or [{"source": "seed"} for _ in texts]
            metadatas = [m if m else {"source": "seed"} for m in metadatas]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            self._docs.extend(texts)
            self._refresh_bm25()
            logger.debug(f"[{self.collection_name}] indexed {len(texts)} chunks")
        except Exception as e:
            logger.error(f"RAG index error ({self.collection_name}): {e}")

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _vector_search(self, query: str, n: int) -> list[str]:
        docs, _ = self._vector_search_scored(query, n)
        return docs

    def _vector_search_scored(self, query: str, n: int):
        """Vector search returning (documents, similarities).
        Chroma returns cosine *distance* (0=identical … 2=opposite); we convert
        to similarity = 1 - distance so higher = more relevant."""
        query_embedding = _embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, self.collection.count()),
        )
        if not results["documents"] or not results["documents"][0]:
            return [], []
        docs = results["documents"][0]
        dists = results.get("distances", [[None]])[0]
        sims = [(1.0 - d) if d is not None else 0.0 for d in dists]
        return docs, sims

    def _bm25_search(self, query: str, n: int) -> list[str]:
        if self._bm25 is None or not self._docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:n] if score > 0]

    def retrieve(self, query: str, k: int = 3, mode: str = "hybrid",
                 min_similarity: float | None = None) -> list[str]:
        """
        Return top-k most relevant chunks for a query.

        mode: "hybrid" (default) | "vector" | "bm25"
        Pipeline: fetch k * fetch_multiplier candidates per retriever,
        RRF-fuse, optionally rerank with a cross-encoder, return top-k.

        min_similarity: if set, retrieval *abstains* when the best candidate's
        cosine similarity is below this threshold — returning [] instead of
        forcing through loosely-related chunks. This is the structural fix for
        out-of-context ("trap") questions: with no context to build on, the
        agent can't confabulate a grounded-looking answer. Typical values
        0.2–0.4 for nomic-embed-text; higher = stricter abstention.
        """
        if self.collection.count() == 0:
            return []
        n_fetch = k * self.fetch_multiplier
        try:
            # Abstention gate: judged on vector similarity, which is comparable
            # across queries (BM25 scores are not).
            if min_similarity is not None:
                _, sims = self._vector_search_scored(query, n_fetch)
                if not sims or max(sims) < min_similarity:
                    logger.debug(f"[{self.collection_name}] abstained: "
                                 f"best sim {max(sims) if sims else 0:.3f} "
                                 f"< {min_similarity}")
                    return []

            if mode == "vector":
                fused = self._vector_search(query, n_fetch)
            elif mode == "bm25":
                fused = self._bm25_search(query, n_fetch)
            else:
                vec = self._vector_search(query, n_fetch)
                kw = self._bm25_search(query, n_fetch)
                fused = reciprocal_rank_fusion([vec, kw]) if kw else vec

            if self.reranker is not None and len(fused) > k:
                return self.reranker.rerank(query, fused, top_k=k)
            return fused[:k]
        except Exception as e:
            logger.error(f"RAG retrieve error ({self.collection_name}): {e}")
            return []

    def retrieve_as_string(self, query: str, k: int = 3, mode: str = "hybrid",
                           min_similarity: float | None = None) -> str:
        """Convenience wrapper — returns chunks joined for prompt injection.
        Returns "" when retrieval abstains (see retrieve's min_similarity)."""
        chunks = self.retrieve(query, k, mode, min_similarity=min_similarity)
        return "\n---\n".join(chunks) if chunks else ""

    def count(self) -> int:
        return self.collection.count()
