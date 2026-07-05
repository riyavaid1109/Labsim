"""
Postdoc — the most complex agent.

Simultaneously:
  - Mentor: helps PhD students, gives constructive feedback
  - Competitor: wants first authorship, has stronger KB than students

Their review style is the most substantive because they've read the most.
Their personality trait (collaborative vs competitive) determines
whether they mentor or undercut in ambiguous situations.
"""

import random
from agents.base import BaseAgent
from utils import query_ollama


POSTDOC_PERSONALITIES = [
    "collaborative and generous with credit",
    "ambitious and protective of their ideas",
    "burned out but still technically sharp",
    "strategic — helps others when it benefits them",
]


class Postdoc(BaseAgent):
    def __init__(self, agent_id: int, name: str, experience: int = None, personality: str = None):
        experience = experience or random.randint(3, 4)
        personality = personality or random.choice(POSTDOC_PERSONALITIES)
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="postdoc",
            experience=experience,
            personality=personality,
        )
        self.mentoring_acts = 0
        self.authorship_wins = 0

    def contribute_to_discussion(self, problem_description: str, discussion_history: list[str]) -> str:
        context = self.retrieve_context(query=problem_description, k=4)
        history_text = "\n".join(discussion_history[-4:]) if discussion_history else "No prior discussion."
        prompt = f"""The lab is discussing:
{problem_description}

Discussion so far:
{history_text}

Give your perspective. You have more experience than the students — you may build on,
correct, or redirect what's been said. Stay under 80 words."""
        return self.speak(prompt, context)

    def write_review(self, paper_draft: str, author_name: str) -> dict:
        """
        Write a peer review. Returns a dict with score and comments.
        Score is 1–5. Comments are grounded in private KB.
        """
        context = self.retrieve_context(query=paper_draft, k=5)
        prompt = f"""You are reviewing a paper submitted by {author_name}.

Paper:
{paper_draft}

Write a peer review with:
1. An overall assessment (2–3 sentences)
2. Three specific comments (strengths or weaknesses), each grounded in methodology or prior work
3. A score from 1 (reject) to 5 (strong accept)

Format your response as:
ASSESSMENT: ...
COMMENT 1: ...
COMMENT 2: ...
COMMENT 3: ...
SCORE: <number>

Be honest. Your personality is: {self.personality}."""
        raw = self.speak(prompt, context)
        return self._parse_review(raw)

    def _parse_review(self, raw: str) -> dict:
        """Extract structured fields from LLM review text."""
        import re
        score_match = re.search(r"SCORE:\s*(\d)", raw)
        score = int(score_match.group(1)) if score_match else 3
        score = max(1, min(5, score))

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
    
    def write_draft(self, problem_description: str, experiment_notes: str) -> str:
        context = self.retrieve_context(query=problem_description, k=5)
        prompt = f"""You are writing a draft for the following research problem:
{problem_description}

Your experiment notes and prior knowledge:
{experiment_notes}

Write a short paper draft (abstract + 3 key points). Be specific about methodology.
Under 200 words."""
        return self.speak(prompt, context)

    def write_rebuttal(self, review_comments: list[str]) -> str:
        comments_text = "\n".join(f"- {c}" for c in review_comments)
        context = self.retrieve_context(query=" ".join(review_comments), k=4)
        prompt = f"""You received these reviewer comments on your paper:
{comments_text}

Write a rebuttal addressing each point. You are experienced — be direct and confident
where the reviewers are wrong, concede gracefully where they are right.
Under 150 words."""
        return self.speak(prompt, context)

    def mentor(self, student_name: str, student_draft: str) -> str:
        """Give mentoring feedback on a student's draft."""
        context = self.retrieve_context(query=student_draft, k=3)
        prompt = f"""You are giving informal feedback to {student_name} on their draft before submission.

Draft:
{student_draft}

Give constructive, specific feedback in 2–3 sentences. Your tone reflects your personality: {self.personality}."""
        self.mentoring_acts += 1
        return self.speak(prompt, context)

    def claim_authorship(self, paper_draft: str) -> float:
        """
        Returns a 0–1 score of how strongly this postdoc claims authorship.
        Driven by their personality and how much their KB contributed to the paper.
        """
        context_overlap = len(self.retrieve(paper_draft, k=3))
        base_claim = 0.5 + (self.experience - 3) * 0.1
        if "ambitious" in self.personality or "strategic" in self.personality:
            base_claim += 0.2
        if "collaborative" in self.personality:
            base_claim -= 0.15
        return round(min(1.0, max(0.0, base_claim + (context_overlap * 0.05) + random.uniform(-0.1, 0.1))), 2)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        return self.kb.retrieve(query, k)
