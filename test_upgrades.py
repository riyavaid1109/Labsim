"""
test_upgrades.py — validates the RAG/LLM upgrades without a live Ollama.

Embeddings and LLM calls are mocked (deterministic hash-based vectors and
scripted responses) so BM25/RRF, chunking, eval metrics, observability,
structured-output parsing, and the tool loop can all be exercised offline.

Run: python test_upgrades.py
"""

import os
import json
import shutil
import hashlib
import unittest
from unittest.mock import patch

os.environ["ANONYMIZED_TELEMETRY"] = "False"

TEST_DIR = "./test_artifacts"


def fake_embed_one(text: str) -> list[float]:
    """Deterministic pseudo-embedding: texts sharing words get closer vectors."""
    vec = [0.0] * 64
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % 64] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def fake_ollama_embeddings(model=None, prompt=""):
    return {"embedding": fake_embed_one(prompt)}


class TestChunking(unittest.TestCase):
    def test_short_text_single_chunk(self):
        from rag.chunking import chunk_text
        self.assertEqual(chunk_text("short note"), ["short note"])

    def test_respects_size_and_overlap(self):
        from rag.chunking import chunk_text
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(6))
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        # allow overlap slack on top of chunk_size
        self.assertTrue(all(len(c) <= 300 + 60 for c in chunks))
        # overlap: chunk i+1 starts with the tail of chunk i's content
        self.assertIn(chunks[0][-20:].split()[-1], chunks[1][:120])

    def test_markdown_headers_are_boundaries(self):
        from rag.chunking import chunk_text
        text = ("# Section A\n" + "alpha " * 60 +
                "\n# Section B\n" + "beta " * 60)
        chunks = chunk_text(text, chunk_size=400, overlap=0)
        joined = [c for c in chunks if "alpha" in c and "beta" in c]
        self.assertEqual(joined, [])  # sections never merged

    def test_chunk_documents_metadata(self):
        from rag.chunking import chunk_documents
        chunks, meta = chunk_documents(["a " * 400, "short"],
                                       chunk_size=200, overlap=20)
        self.assertEqual(len(chunks), len(meta))
        self.assertEqual(meta[-1]["source"], "doc_1")
        self.assertTrue(all("chunk_index" in m for m in meta))


class TestHybridRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        cls.patcher = patch("rag.retriever.ollama.embeddings",
                            side_effect=fake_ollama_embeddings)
        cls.patcher.start()
        from rag.retriever import RAGRetriever
        cls.r = RAGRetriever("test_hybrid", persist_dir=TEST_DIR)
        cls.docs = [
            "BM25 is a keyword ranking function used in search engines",
            "transformers use self attention for sequence modeling",
            "the XQC-9000 dataset contains annotated retail videos",
            "vector embeddings capture semantic similarity between texts",
            "reciprocal rank fusion combines multiple ranked lists",
        ]
        cls.r.index(cls.docs)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        shutil.rmtree(TEST_DIR, ignore_errors=True)

    def test_bm25_exact_term(self):
        # exact rare token: BM25 must nail it
        out = self.r.retrieve("XQC-9000", k=2, mode="bm25")
        self.assertTrue(out and "XQC-9000" in out[0])

    def test_vector_mode_runs(self):
        out = self.r.retrieve("semantic similarity embeddings", k=2, mode="vector")
        self.assertEqual(len(out), 2)

    def test_hybrid_includes_keyword_hit(self):
        out = self.r.retrieve("XQC-9000 annotated videos", k=3, mode="hybrid")
        self.assertTrue(any("XQC-9000" in d for d in out))

    def test_rrf_math(self):
        from rag.retriever import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
        # b: 1/62+1/61 > a: 1/61+1/63 > c: 1/63+1/62
        self.assertEqual(fused, ["b", "a", "c"])

    def test_bm25_survives_restart(self):
        from rag.retriever import RAGRetriever
        r2 = RAGRetriever("test_hybrid", persist_dir=TEST_DIR)
        out = r2.retrieve("reciprocal rank fusion", k=1, mode="bm25")
        self.assertTrue(out and "fusion" in out[0])


class TestRerankerFallback(unittest.TestCase):
    def test_graceful_without_model(self):
        from rag.reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(model_name="definitely/not-a-model")
        rr._unavailable = True  # simulate missing sentence-transformers
        out = rr.rerank("q", ["d1", "d2", "d3"], top_k=2)
        self.assertEqual(out, ["d1", "d2"])


class TestEvalMetrics(unittest.TestCase):
    def test_score_retrieval(self):
        from rag.eval import score_retrieval
        retrieved = ["about attention mechanisms", "irrelevant text",
                     "sparse attention variants"]
        s = score_retrieval(retrieved, ["attention", "sparse attention"], k=3)
        self.assertAlmostEqual(s["precision_at_k"], 2 / 3)
        self.assertEqual(s["recall_at_k"], 1.0)
        self.assertEqual(s["mrr"], 1.0)
        self.assertEqual(s["hit_at_k"], 1.0)

    def test_miss(self):
        from rag.eval import score_retrieval
        s = score_retrieval(["nothing here"], ["quantum"], k=1)
        self.assertEqual(s["precision_at_k"], 0.0)
        self.assertEqual(s["mrr"], 0.0)


class TestObservability(unittest.TestCase):
    def test_trace_roundtrip(self):
        from llm import observability as obs
        db = os.path.join(TEST_DIR, "traces.db")
        os.makedirs(TEST_DIR, exist_ok=True)
        obs.configure(db_path=db, enabled=True)
        with obs.traced_call("ollama", "llama3", "hi", caller="test") as t:
            t["response"] = "hello"
            t["prompt_tokens"] = 5
            t["completion_tokens"] = 2
        summary = obs.trace_summary(db)
        self.assertIn("test", summary)
        self.assertIn("llama3", summary)
        n = obs.export_traces(os.path.join(TEST_DIR, "t.jsonl"), db)
        self.assertEqual(n, 1)
        obs.configure(enabled=False)


class TestStructuredParsing(unittest.TestCase):
    def test_parse_json_variants(self):
        from llm.client import _parse_json
        self.assertEqual(_parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(_parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_parse_json('Sure! {"a": 1} hope that helps'), {"a": 1})
        with self.assertRaises(ValueError):
            _parse_json("no json here")


class TestToolLoop(unittest.TestCase):
    def test_tool_loop_search_then_respond(self):
        from llm import observability as obs
        obs.configure(enabled=False)

        class FakeAgent:
            name = "TestAgent"

            class kb:
                @staticmethod
                def retrieve_as_string(q, k=3):
                    return "relevant chunk about " + q

            def persona_block(self):
                return "You are TestAgent."

            def speak(self, p, c=""):
                return "freeform fallback"

            def retrieve_context(self, q, k=3):
                return ""

        script = iter([
            {"tool": "search_kb", "args": {"query": "reid embeddings"}},
            {"tool": "cite_source", "args": {"claim": "OSNet works",
                                             "source": "Zhou 2019"}},
            {"tool": "respond", "args": {"text": "final answer"}},
        ])
        with patch("agents.tools.query_llm_json",
                   side_effect=lambda *a, **k: next(script)):
            from agents.tools import run_tool_loop
            out = run_tool_loop(FakeAgent(), "discuss reid")

        self.assertEqual(out["response"], "final answer")
        self.assertEqual(len(out["citations"]), 1)
        self.assertEqual(out["tool_trace"][0]["tool"], "search_kb")

    def test_tool_loop_falls_back_on_failure(self):
        from llm import observability as obs
        obs.configure(enabled=False)

        class FakeAgent:
            name = "TestAgent"
            kb = None

            def persona_block(self):
                return "persona"

            def speak(self, p, c=""):
                return "freeform fallback"

            def retrieve_context(self, q, k=3):
                return ""

        with patch("agents.tools.query_llm_json", return_value=None):
            from agents.tools import run_tool_loop
            out = run_tool_loop(FakeAgent(), "task")
        self.assertEqual(out["response"], "freeform fallback")


class TestStructuredReview(unittest.TestCase):
    def test_write_review_structured_path(self):
        fake_review = {"assessment": "Solid work overall.",
                       "comments": ["good baselines", "weak ablations",
                                    "clear writing"],
                       "score": 4}
        with patch("agents.base.query_llm_json", return_value=fake_review), \
             patch("agents.base.query_ollama", return_value="persona text"), \
             patch("rag.retriever.ollama.embeddings",
                   side_effect=fake_ollama_embeddings):
            from agents.base import BaseAgent
            agent = BaseAgent(99, "Rev", "postdoc", 4, "cautious",
                              persist_dir=TEST_DIR + "_rev")
            review = agent.write_review("a paper draft", "Author X")
        self.assertEqual(review["score"], 4)
        self.assertEqual(len(review["comments"]), 3)
        self.assertEqual(review["assessment"], "Solid work overall.")

    def test_write_review_fallback_path(self):
        raw = ("ASSESSMENT: Decent paper.\nCOMMENT 1: strong intro\n"
               "COMMENT 2: missing citations\nCOMMENT 3: good figures\nSCORE: 3")
        with patch("agents.base.query_llm_json", return_value=None), \
             patch("agents.base.query_ollama", return_value=raw), \
             patch("rag.retriever.ollama.embeddings",
                   side_effect=fake_ollama_embeddings):
            from agents.base import BaseAgent
            agent = BaseAgent(98, "Rev2", "postdoc", 4, "cautious",
                              persist_dir=TEST_DIR + "_rev")
            review = agent.write_review("a paper draft", "Author X")
        self.assertEqual(review["score"], 3)
        self.assertEqual(review["comments"],
                         ["strong intro", "missing citations", "good figures"])


class TestEvaluateRetrieverEndToEnd(unittest.TestCase):
    def test_modes_compared(self):
        shutil.rmtree(TEST_DIR + "_e2e", ignore_errors=True)
        with patch("rag.retriever.ollama.embeddings",
                   side_effect=fake_ollama_embeddings):
            from rag.retriever import RAGRetriever
            from rag.eval import evaluate_retriever, format_report
            r = RAGRetriever("e2e", persist_dir=TEST_DIR + "_e2e")
            r.index([
                "BoT-SORT is a multi object tracker with camera compensation",
                "OSNet learns omni scale features for person re identification",
                "the kalman filter predicts object motion between frames",
            ])
            evalset = [
                {"query": "person re identification features",
                 "relevant": ["OSNet", "re identification"]},
                {"query": "BoT-SORT tracker", "relevant": ["BoT-SORT"]},
            ]
            results = evaluate_retriever(r, evalset, k=2)
            report = format_report(results, 2)
        shutil.rmtree(TEST_DIR + "_e2e", ignore_errors=True)
        for mode in ("vector", "bm25", "hybrid"):
            self.assertIn(mode, results)
        self.assertGreaterEqual(results["hybrid"]["hit_at_k"], 0.5)
        self.assertIn("P@k", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
