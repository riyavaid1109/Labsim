"""
KnowledgeSpace — the lab's shared memory.

Two things in one:
  1. A message log (Slack-like) of everything that happens in the simulation
  2. A shared RAG store (meeting notes, past drafts, decisions) queryable by all agents

Analogous to ConversationSpace in LLAMOSC but oriented around a research lab.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
from rag.retriever import RAGRetriever

logger = logging.getLogger("lab_sim")


@dataclass
class Message:
    sender: str
    content: str
    msg_type: str          # "discussion", "review", "decision", "system"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    timestep: int = 0


class KnowledgeSpace:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.messages: list[Message] = []
        self.shared_kb = RAGRetriever(collection_name="shared_lab", persist_dir=persist_dir)
        self.current_timestep = 0

    # ── Message log ──────────────────────────────────────────────────────────

    def post(self, sender: str, content: str, msg_type: str = "discussion") -> None:
        msg = Message(
            sender=sender,
            content=content,
            msg_type=msg_type,
            timestep=self.current_timestep,
        )
        self.messages.append(msg)
        logger.debug(f"[{msg_type}] {sender}: {content[:80]}")

    def get_history(self, msg_type: str = None, last_n: int = None) -> list[Message]:
        msgs = self.messages
        if msg_type:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        if last_n:
            msgs = msgs[-last_n:]
        return msgs

    def get_history_as_string(self, msg_type: str = None, last_n: int = 20) -> str:
        msgs = self.get_history(msg_type=msg_type, last_n=last_n)
        return "\n".join(f"[{m.msg_type}] {m.sender}: {m.content}" for m in msgs)

    # ── Shared KB ─────────────────────────────────────────────────────────────

    def index_to_shared(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """Add documents to the shared lab knowledge base.
        Documents are recursively chunked before indexing; per-document
        metadata is propagated to every chunk of that document."""
        from rag.chunking import chunk_documents
        chunks, chunk_meta = chunk_documents(texts)
        if metadatas:
            for cm in chunk_meta:
                doc_idx = int(cm["source"].split("_")[-1])
                if doc_idx < len(metadatas) and metadatas[doc_idx]:
                    cm.update(metadatas[doc_idx])
        self.shared_kb.index(chunks, chunk_meta)

    def retrieve_shared(self, query: str, k: int = 3) -> str:
        """Query the shared KB — available to all agents."""
        return self.shared_kb.retrieve_as_string(query, k)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def bus_factor(self) -> int:
        """
        How many contributors account for 80% of discussion messages?
        Low bus factor = dangerous concentration of knowledge.
        """
        counts = Counter(
            m.sender for m in self.messages
            if m.msg_type == "discussion" and m.sender != "system"
        )
        if not counts:
            return 0
        total = sum(counts.values())
        cumulative, bf = 0, 0
        for c in sorted(counts.values(), reverse=True):
            cumulative += c
            bf += 1
            if cumulative / total >= 0.8:
                break
        return bf

    def message_entropy(self) -> float:
        """
        Shannon entropy of participation.
        High entropy = diverse contributions. Low = dominated by one person.
        """
        counts = Counter(
            m.sender for m in self.messages
            if m.msg_type == "discussion" and m.sender != "system"
        )
        if not counts:
            return 0.0
        total = sum(counts.values())
        return round(-sum((c / total) * math.log2(c / total) for c in counts.values()), 3)
    def message_entropy_this_timestep(self, timestep: int) -> float:
        counts = Counter(
        m.sender for m in self.messages
        if m.msg_type == "discussion"
        and m.sender != "system"
        and m.timestep == timestep
    )
        if not counts:
            return 0.0
        total = sum(counts.values())
        return round(-sum((c/total) * math.log2(c/total) for c in counts.values()), 3)

    def avg_review_score(self) -> float:
        """Mean score across all completed peer reviews."""
        review_msgs = [m for m in self.messages if m.msg_type == "review" and "SCORE:" in m.content]
        if not review_msgs:
            return 0.0
        import re
        scores = []
        for m in review_msgs:
            match = re.search(r"SCORE:\s*(\d)", m.content)
            if match:
                scores.append(int(match.group(1)))
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    def credit_dispute_rate(self, agents: list) -> float:
        """Fraction of agents who have experienced at least one credit dispute."""
        if not agents:
            return 0.0
        disputed = sum(1 for a in agents if hasattr(a, "credit_disputes") and a.credit_disputes > 0)
        return round(disputed / len(agents), 2)

    def summary(self, agents: list = None) -> dict:
        return {
            "total_messages": len(self.messages),
            "bus_factor": self.bus_factor(),
            "message_entropy": self.message_entropy(),
            "avg_review_score": self.avg_review_score(),
            "credit_dispute_rate": self.credit_dispute_rate(agents or []),
            "shared_kb_chunks": self.shared_kb.count(),
        }
