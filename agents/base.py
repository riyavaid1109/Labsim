"""
BaseAgent — shared state and behaviour for all lab members.

Every agent has:
  - A private RAGRetriever (their own reading history / notes)
  - A motivation level (0–10) that updates based on outcomes
  - A reputation score that accumulates across the simulation
  - A personality dict that shapes LLM prompt tone
"""

import re
import random
import logging
from rag.retriever import RAGRetriever
from rag.chunking import chunk_documents
from utils import query_ollama, query_llm_json

# JSON schema for structured peer reviews — replaces regex parsing of
# freeform text. Ollama constrains decoding to this schema natively.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "comments": {"type": "array", "items": {"type": "string"},
                     "minItems": 3, "maxItems": 3},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["assessment", "comments", "score"],
}

logger = logging.getLogger("lab_sim")


class BaseAgent:
    def __init__(
        self,
        agent_id: int,
        name: str,
        role: str,
        experience: int,          # 1–5
        personality: str,         # e.g. "collaborative", "competitive", "cautious"
        persist_dir: str = "./chroma_db",
    ):
        self.id = agent_id
        self.name = name
        self.role = role
        self.experience = experience
        self.personality = personality

        # State
        self.motivation = self._initial_motivation()
        self.motivation_history = [self.motivation]
        self.reputation = 0.0
        self.available = True

        # Private knowledge base — unique per agent
        collection_name = f"agent_{agent_id}_{name.lower().replace(' ', '_')}"
        self.kb = RAGRetriever(collection_name=collection_name, persist_dir=persist_dir)

        # Role description generated once at init
        self.role_description = self._generate_role_description()

    def _initial_motivation(self) -> float:
        """Seed motivation from experience + randomness."""
        base = 5.0 + (self.experience - 3) * 0.5
        return round(max(1.0, min(10.0, base + random.uniform(-1, 1))), 2)

    def _generate_role_description(self) -> str:
        prompt = f"""You are {self.name}, a {self.role} in a research lab.
Your experience level is {self.experience}/5.
Your personality is: {self.personality}.
Write a 2-sentence first-person description of who you are and what drives your work.
Be specific and realistic. Do not be generically positive."""
        return query_ollama(prompt)

    def seed_knowledge(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """Load initial papers / notes into this agent's private KB.
        Documents are recursively chunked (512 chars, 64 overlap) before
        indexing so retrieval granularity stays consistent."""
        chunks, chunk_meta = chunk_documents(texts)
        if metadatas:  # preserve caller-provided source labels
            for meta, src in zip(chunk_meta, self._expand_meta(metadatas, chunk_meta)):
                meta.update(src or {})
        self.kb.index(chunks, chunk_meta)
        logger.info(f"{self.name}: seeded KB with {len(chunks)} chunks "
                    f"from {len(texts)} documents")

    @staticmethod
    def _expand_meta(doc_metas: list[dict], chunk_meta: list[dict]) -> list[dict]:
        """Map per-document metadata onto per-chunk metadata rows."""
        expanded = []
        for cm in chunk_meta:
            doc_idx = int(cm["source"].split("_")[-1]) if cm["source"].startswith("doc_") else 0
            expanded.append(doc_metas[doc_idx] if doc_idx < len(doc_metas) else {})
        return expanded

    def persona_block(self) -> str:
        """The identity preamble injected into every prompt for this agent."""
        return f"""You are {self.name}, a {self.role} in a research lab.
{self.role_description}
Your personality: {self.personality}.
Motivation level: {self.motivation:.1f}/10.
Speak in first person. Be concise (under 80 words). Stay in character."""

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Pull relevant context from private KB for a given query."""
        return self.kb.retrieve_as_string(query, k)

    def update_motivation(
        self,
        credited: bool = False,
        paper_accepted: bool = False,
        ignored: bool = False,
        task_difficulty: int = 3,
    ) -> None:
        delta = 0.0
        if credited and paper_accepted:
            delta += max(0.5, 0.15 * task_difficulty)
        elif credited and not paper_accepted:
            delta -= max(0.3, 0.08 * task_difficulty)
        elif ignored:
            delta -= max(0.5, 0.1 * task_difficulty)

        # Small random fluctuation
        delta += random.uniform(-0.3, 0.3)
        self.motivation = round(max(0.0, min(10.0, self.motivation + delta)), 2)
        self.motivation_history.append(self.motivation)
        logger.info(f"{self.name} motivation: {self.motivation_history[-2]} → {self.motivation}")

    def update_reputation(self, delta: float) -> None:
        self.reputation = round(self.reputation + delta, 2)
        logger.info(f"{self.name} reputation: {self.reputation}")

    def speak(self, prompt: str, context: str = "") -> str:
        """Core speak method — injects RAG context and persona into every LLM call."""
        full_prompt = self.persona_block()
        if context:
            full_prompt += f"\n\n[Relevant context from your notes and papers]\n{context}"
        full_prompt += f"\n\n{prompt}"
        return query_ollama(full_prompt, caller=f"{self.name}.speak")

    def speak_with_tools(self, task_prompt: str, knowledge_space=None) -> dict:
        """Agentic counterpart of speak(): the agent acts through structured
        tool calls (search_kb / cite_source / flag_dispute) before responding.
        Returns {"response", "citations", "disputes", "tool_trace"}."""
        from agents.tools import run_tool_loop  # local import avoids a cycle
        return run_tool_loop(self, task_prompt, knowledge_space)

    def write_review(self, paper_draft: str, author_name: str) -> dict:
        """Peer review via JSON-schema-constrained output. Falls back to the
        original freeform prompt + regex parse if the structured call fails,
        so weaker models degrade instead of breaking the review cycle."""
        context = self.retrieve_context(query=paper_draft, k=4)
        prompt = f"""{self.persona_block()}

[Relevant context from your notes and papers]
{context}

You are reviewing a paper submitted by {author_name}.

Paper:
{paper_draft}

Write an honest peer review: a 2-3 sentence overall assessment, exactly 3
specific comments (strengths or weaknesses), and an integer score from 1 to 5."""

        data = query_llm_json(prompt, REVIEW_SCHEMA, caller=f"{self.name}.write_review")
        if data is not None:
            try:
                score = max(1, min(5, int(data["score"])))
                comments = [str(c) for c in data["comments"]][:3]
                return {
                    "reviewer": self.name,
                    "score": score,
                    "assessment": str(data["assessment"]).strip(),
                    "comments": comments,
                    "raw": str(data),
                }
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"{self.name}: structured review malformed ({e}), "
                               "falling back to freeform")
        return self._write_review_freeform(paper_draft, author_name, context)

    def _write_review_freeform(self, paper_draft: str, author_name: str,
                               context: str) -> dict:
        """Original freeform review path, kept as the fallback."""
        prompt = f"""You are reviewing a paper submitted by {author_name}.

Paper:
{paper_draft}

Write a peer review with:
ASSESSMENT: <2-3 sentence overall assessment>
COMMENT 1: <specific strength or weakness>
COMMENT 2: <specific strength or weakness>
COMMENT 3: <specific strength or weakness>
SCORE: <number from 1 to 5>

Be honest. Your role is {self.role}, personality: {self.personality}."""
        raw = self.speak(prompt, context)
        score_match = re.search(r"SCORE:\s*(\d)", raw)
        score = max(1, min(5, int(score_match.group(1)))) if score_match else 3
        comments = re.findall(r"COMMENT \d+:\s*(.+)", raw)
        assessment_match = re.search(r"ASSESSMENT:\s*(.+)", raw)
        assessment = assessment_match.group(1).strip() if assessment_match else raw[:200]
        return {
            "reviewer": self.name,
            "score": score,
            "assessment": assessment,
            "comments": comments if comments else [raw],
            "raw": raw,
        }

    def __repr__(self):
        return f"<{self.role} {self.name} exp={self.experience} mot={self.motivation}>"