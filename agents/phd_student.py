"""
PhDStudent — junior lab member, high energy, fragile motivation.

Specific behaviours:
  - Contributes ideas but hedges language (defers to seniority)
  - Motivation drops sharply if credit is taken by a senior member
  - Experience grows when their work gets accepted
"""

import random
from agents.base import BaseAgent
from utils import query_ollama


class PhDStudent(BaseAgent):
    def __init__(self, agent_id: int, name: str, experience: int = None, personality: str = None):
        experience = experience or random.randint(1, 3)
        personality = personality or random.choice([
            "eager to prove themselves",
            "cautious and methodical",
            "creative but disorganised",
            "quietly ambitious",
        ])
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="PhD student",
            experience=experience,
            personality=personality,
        )
        self.papers_contributed = 0
        self.credit_disputes = 0     # times their idea was attributed to someone else

    def contribute_to_discussion(self, problem_description: str, discussion_history: list[str]) -> str:
        """Generate a discussion contribution grounded in private KB."""
        context = self.retrieve_context(
            query=problem_description,
            k=3,
        )
        history_text = "\n".join(discussion_history[-4:]) if discussion_history else "No prior discussion."
        prompt = f"""The lab is discussing this research problem:
{problem_description}

Discussion so far:
{history_text}

Share your perspective on this problem. Propose an approach or raise a concern.
Remember you are relatively junior — acknowledge uncertainty where appropriate.
Keep your response under 80 words."""
        return self.speak(prompt, context)

    def write_draft(self, problem_description: str, experiment_notes: str) -> str:
        """Produce a paper draft given a problem and their experiment notes."""
        context = self.retrieve_context(query=problem_description, k=5)
        prompt = f"""You are writing a draft for the following research problem:
{problem_description}

Your experiment notes:
{experiment_notes}

Write a short paper draft (abstract + 3 key points). Be specific about methodology.
Under 200 words."""
        return self.speak(prompt, context)

    def write_rebuttal(self, review_comments: list[str]) -> str:
        """Respond to reviewer comments."""
        comments_text = "\n".join(f"- {c}" for c in review_comments)
        context = self.retrieve_context(query=" ".join(review_comments), k=4)
        prompt = f"""You received these reviewer comments on your paper:
{comments_text}

Write a rebuttal addressing each point. Be respectful but defend your work where justified.
Under 150 words."""
        return self.speak(prompt, context)

    def record_credit_dispute(self) -> None:
        self.credit_disputes += 1
        self.update_motivation(ignored=True, task_difficulty=4)

    def grow_experience(self, amount: float = 0.5) -> None:
        self.experience = round(min(5.0, self.experience + amount), 2)
        self.papers_contributed += 1
