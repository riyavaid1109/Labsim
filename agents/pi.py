# """
# PI (Principal Investigator) — high authority, limited bandwidth.

# Does not participate in every discussion. Makes final calls on:
#   - Who gets assigned to a problem
#   - Authorship order
#   - Accept / revise / reject in peer review

# Has a fairness trait (0–1): 0 = plays pure favorites, 1 = purely merit-based.
# This is the main experimental variable you can toggle between runs.
# """

# import random
# import re
# import logging
# from agents.base import BaseAgent
# from utils import query_ollama

# logger = logging.getLogger("lab_sim")


# class PI(BaseAgent):
#     def __init__(
#         self,
#         agent_id: int,
#         name: str,
#         experience: int = 5,
#         personality: str = "results-driven and protective of the lab's reputation",
#         fairness: float = None,   # 0.0 = plays favorites, 1.0 = pure merit
#     ):
#         super().__init__(
#             agent_id=agent_id,
#             name=name,
#             role="PI",
#             experience=experience,
#             personality=personality,
#         )
#         self.fairness = fairness if fairness is not None else round(random.uniform(0.3, 0.9), 2)
#         self.favorites: list[int] = []   # agent IDs the PI tends to favor
#         logger.info(f"PI {name} created with fairness={self.fairness}")

#     def assign_problem(self, candidates: list, problem_description: str) -> object:
#         """
#         Pick who leads the next research problem.
#         Balances merit (experience + reputation) vs favoritism.
#         """
#         if not candidates:
#             return None

#         def score(agent):
#             merit = agent.experience * 0.5 + agent.reputation * 0.3 + agent.motivation * 0.2
#             is_favorite = agent.id in self.favorites
#             # fairness=1.0 → pure merit score
#             # fairness=0.0 → merit score + large favorite bonus
#             favor_add = (1 - self.fairness) * 3.0 if is_favorite else 0.0
#             return merit + favor_add

#         ranked = sorted(candidates, key=score, reverse=True)
#         selected = ranked[0]
#         logger.info(f"PI assigned {selected.name} to problem (fairness={self.fairness})")
#         return selected

#     def assign_reviewers(self, all_agents: list, author_id: int, n: int = 2) -> list:
#         """Pick reviewers — excludes the author, picks by experience."""
#         eligible = [a for a in all_agents if a.id != author_id and a.available]
#         eligible.sort(key=lambda a: a.experience, reverse=True)
#         return eligible[:n]

#     def resolve_authorship(self, candidates: list[dict]) -> list[dict]:
#         """
#         Decide authorship order given competing claims.
#         candidates: list of {agent, claim_score}
#         Returns ordered list with authorship position assigned.
#         """
#         def effective_claim(c):
#             merit_weight = self.fairness
#             favor_weight = 1 - self.fairness
#             favor_bonus = 1.3 if c["agent"].id in self.favorites else 1.0
#             return (c["claim_score"] * merit_weight) + (c["agent"].reputation * favor_weight * favor_bonus)

#         ordered = sorted(candidates, key=effective_claim, reverse=True)
#         for i, c in enumerate(ordered):
#             c["authorship_position"] = i + 1
#         return ordered

#     def make_decision(self, paper_draft: str, reviews: list[dict], rebuttal: str) -> dict:
#         """
#         Final accept/revise/reject decision after seeing reviews + rebuttal.
#         Returns {decision, rationale, score_adjustment}
#         """
#         avg_score = sum(r["score"] for r in reviews) / len(reviews) if reviews else 3.0
#         context = self.retrieve_context(query=paper_draft, k=3)

#         review_summaries = "\n".join(
#             f"Reviewer {r['reviewer']}: score={r['score']}, {r['assessment']}"
#             for r in reviews
#         )
#         prompt = f"""You are the PI making a final decision on a paper submission.

# Paper (excerpt):
# {paper_draft[:400]}

# Reviewer assessments:
# {review_summaries}

# Author rebuttal:
# {rebuttal}

# Average reviewer score: {avg_score:.1f}/5

# Make a decision: ACCEPT, REVISE, or REJECT.
# Format:
# DECISION: <one word>
# RATIONALE: <2 sentences>"""

#         raw = self.speak(prompt, context)
#         decision_match = re.search(r"DECISION:\s*(ACCEPT|REVISE|REJECT)", raw, re.IGNORECASE)
#         decision = decision_match.group(1).upper() if decision_match else self._fallback_decision(avg_score)
#         rationale_match = re.search(r"RATIONALE:\s*(.+)", raw, re.DOTALL)
#         rationale = rationale_match.group(1).strip()[:300] if rationale_match else raw[:300]

#         score_adj = {"ACCEPT": 1.0, "REVISE": 0.0, "REJECT": -0.5}.get(decision, 0.0)
#         return {"decision": decision, "rationale": rationale, "score_adjustment": score_adj}

#     def _fallback_decision(self, avg_score: float) -> str:
#         if avg_score >= 3.5:
#             return "ACCEPT"
#         elif avg_score >= 2.5:
#             return "REVISE"
#         return "REJECT"

#     def add_favorite(self, agent_id: int) -> None:
#         if agent_id not in self.favorites:
#             self.favorites.append(agent_id)







"""
PI (Principal Investigator) — high authority, limited bandwidth.

Does not participate in every discussion. Makes final calls on:
  - Who gets assigned to a problem
  - Authorship order
  - Accept / revise / reject in peer review

Has a fairness trait (0–1): 0 = plays pure favorites, 1 = purely merit-based.
This is the main experimental variable you can toggle between runs.
"""

import random
import re
import logging
from agents.base import BaseAgent
from utils import query_ollama

logger = logging.getLogger("lab_sim")


class PI(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        name: str,
        experience: int = 5,
        personality: str = "results-driven and protective of the lab's reputation",
        fairness: float = None,   # 0.0 = plays favorites, 1.0 = pure merit
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="PI",
            experience=experience,
            personality=personality,
        )
        self.fairness = fairness if fairness is not None else round(random.uniform(0.3, 0.9), 2)
        self.favorites: list[int] = []   # agent IDs the PI tends to favor
        self.lead_counts: dict[int, int] = {}  # agent_id → times assigned as lead
        logger.info(f"PI {name} created with fairness={self.fairness}")

    def assign_problem(self, candidates: list, problem_description: str) -> object:
        """
        Pick who leads the next research problem.

        Fairness controls *opportunity distribution*, not just a favorite bonus:

          - fairness → 1.0: the PI actively spreads leadership. Candidates who
            have led less are boosted (an equity term), and selection is
            softmax-sampled rather than pure argmax, so strong-but-overused
            agents don't monopolize and students get real chances.
          - fairness → 0.0: the PI concentrates on favorites and raw merit,
            ignoring how lopsided the distribution becomes.

        This makes the fairness knob change the *dynamics* of who leads over
        time, not merely which single agent wins every round.
        """
        if not candidates:
            return None

        max_led = max((self.lead_counts.get(a.id, 0) for a in candidates), default=0)

        def score(agent):
            led = self.lead_counts.get(agent.id, 0)
            # Raw merit — what a purely results-driven PI sees.
            merit = agent.experience * 0.5 + agent.reputation * 0.3 + agent.motivation * 0.2
            # Favoritism: matters only as fairness drops.
            favor_add = (1 - self.fairness) * 3.0 if agent.id in self.favorites else 0.0
            # Equity: reward under-used agents, scaled by fairness. A high-fairness
            # PI treats "hasn't led yet" as a strong reason to pick someone.
            equity = self.fairness * 2.0 * (max_led - led)
            return merit + favor_add + equity

        scores = [score(a) for a in candidates]

        # Selection temperature also scales with fairness: a fair PI samples
        # (spreads opportunity), an unfair PI takes the argmax (locks in).
        if self.fairness >= 0.5:
            selected = self._softmax_sample(candidates, scores,
                                            temperature=0.5 + self.fairness)
        else:
            selected = candidates[max(range(len(scores)), key=scores.__getitem__)]

        self.lead_counts[selected.id] = self.lead_counts.get(selected.id, 0) + 1
        logger.info(f"PI assigned {selected.name} to problem "
                    f"(fairness={self.fairness}, led={self.lead_counts[selected.id]}x)")
        return selected

    @staticmethod
    def _softmax_sample(candidates: list, scores: list, temperature: float):
        """Sample a candidate with probability proportional to
        exp(score / temperature). Higher temperature flattens the distribution
        and spreads opportunity."""
        import math
        t = max(0.1, temperature)
        mx = max(scores)
        weights = [math.exp((s - mx) / t) for s in scores]  # subtract max for stability
        total = sum(weights) or 1.0
        r = random.random() * total
        upto = 0.0
        for agent, w in zip(candidates, weights):
            upto += w
            if upto >= r:
                return agent
        return candidates[-1]

    def assign_reviewers(self, all_agents: list, author_id: int, n: int = 2) -> list:
        """Pick reviewers — excludes the author, picks by experience."""
        eligible = [a for a in all_agents if a.id != author_id and a.available]
        eligible.sort(key=lambda a: a.experience, reverse=True)
        return eligible[:n]

    def resolve_authorship(self, candidates: list[dict]) -> list[dict]:
        """
        Decide authorship order given competing claims.
        candidates: list of {agent, claim_score}
        Returns ordered list with authorship position assigned.
        """
        def effective_claim(c):
            merit_weight = self.fairness
            favor_weight = 1 - self.fairness
            favor_bonus = 1.3 if c["agent"].id in self.favorites else 1.0
            return (c["claim_score"] * merit_weight) + (c["agent"].reputation * favor_weight * favor_bonus)

        ordered = sorted(candidates, key=effective_claim, reverse=True)
        for i, c in enumerate(ordered):
            c["authorship_position"] = i + 1
        return ordered

    def make_decision(self, paper_draft: str, reviews: list[dict], rebuttal: str) -> dict:
        """
        Final accept/revise/reject decision after seeing reviews + rebuttal.
        Returns {decision, rationale, score_adjustment}
        """
        avg_score = sum(r["score"] for r in reviews) / len(reviews) if reviews else 3.0
        context = self.retrieve_context(query=paper_draft, k=3)

        review_summaries = "\n".join(
            f"Reviewer {r['reviewer']}: score={r['score']}, {r['assessment']}"
            for r in reviews
        )
        prompt = f"""You are the PI making a final decision on a paper submission.

Paper (excerpt):
{paper_draft[:400]}

Reviewer assessments:
{review_summaries}

Author rebuttal:
{rebuttal}

Average reviewer score: {avg_score:.1f}/5

Make a decision: ACCEPT, REVISE, or REJECT.
Format:
DECISION: <one word>
RATIONALE: <2 sentences>"""

        raw = self.speak(prompt, context)
        decision_match = re.search(r"DECISION:\s*(ACCEPT|REVISE|REJECT)", raw, re.IGNORECASE)
        decision = decision_match.group(1).upper() if decision_match else self._fallback_decision(avg_score)
        rationale_match = re.search(r"RATIONALE:\s*(.+)", raw, re.DOTALL)
        rationale = rationale_match.group(1).strip()[:300] if rationale_match else raw[:300]

        score_adj = {"ACCEPT": 1.0, "REVISE": 0.0, "REJECT": -0.5}.get(decision, 0.0)
        return {"decision": decision, "rationale": rationale, "score_adjustment": score_adj}

    def _fallback_decision(self, avg_score: float) -> str:
        if avg_score >= 3.5:
            return "ACCEPT"
        elif avg_score >= 2.5:
            return "REVISE"
        return "REJECT"

    def add_favorite(self, agent_id: int) -> None:
        if agent_id not in self.favorites:
            self.favorites.append(agent_id)