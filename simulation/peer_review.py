"""
PeerReview — the review sub-loop triggered when a paper draft is ready.

Flow:
  1. PI assigns reviewers (excludes author)
  2. Each reviewer reads the draft + queries their private KB → writes review
  3. Author reads all reviews + queries their KB → writes rebuttal
  4. PI reads reviews + rebuttal → makes final decision
  5. Outcomes update reputation and motivation for all involved

The interesting dynamic: reviewers grounded in different KBs may
give contradictory feedback. The rebuttal has to address both.
"""

import logging
from simulation.knowledge_space import KnowledgeSpace

logger = logging.getLogger("lab_sim")


class PeerReview:
    def __init__(self, knowledge_space: KnowledgeSpace):
        self.ks = knowledge_space

    def run(
        self,
        paper_draft: str,
        author,
        pi,
        all_agents: list,
        n_reviewers: int = 2,
    ) -> dict:
        """
        Full peer review cycle.

        Returns a result dict with decision, reviews, rebuttal, and outcome deltas.
        """
        logger.info(f"Peer review starting for paper by {author.name}")
        self.ks.post("system", f"Peer review opened for {author.name}'s paper.", "system")

        # ── Step 1: Assign reviewers ──────────────────────────────────────────
        reviewers = pi.assign_reviewers(all_agents, author_id=author.id, n=n_reviewers)
        if not reviewers:
            logger.warning("No eligible reviewers — skipping peer review")
            return {"decision": "SKIP", "reason": "no eligible reviewers"}

        reviewer_names = ", ".join(r.name for r in reviewers)
        self.ks.post("system", f"Reviewers assigned: {reviewer_names}", "system")

        # ── Step 2: Each reviewer writes a review ─────────────────────────────
        reviews = []
        for reviewer in reviewers:
            logger.info(f"  {reviewer.name} writing review...")
            review = reviewer.write_review(paper_draft, author_name=author.name)
            reviews.append(review)
            self.ks.post(
                reviewer.name,
                f"SCORE: {review['score']}\n{review['assessment']}\n" +
                "\n".join(review["comments"]),
                "review",
            )
            # Index the review into shared KB so the author can retrieve it
            self.ks.index_to_shared(
                [f"Review by {reviewer.name}: {review['assessment']}"],
                [{"type": "review", "reviewer": reviewer.name}],
            )

        # ── Step 3: Author writes rebuttal ────────────────────────────────────
        all_comments = [c for r in reviews for c in r["comments"]]
        logger.info(f"  {author.name} writing rebuttal...")
        rebuttal = author.write_rebuttal(all_comments)
        self.ks.post(author.name, rebuttal, "discussion")
        self.ks.index_to_shared(
            [f"Author rebuttal by {author.name}: {rebuttal}"],
            [{"type": "rebuttal", "author": author.name}],
        )

        # ── Step 4: PI makes final decision ───────────────────────────────────
        logger.info("  PI making final decision...")
        result = pi.make_decision(paper_draft, reviews, rebuttal)
        decision = result["decision"]
        self.ks.post(pi.name, f"Decision: {decision}. {result['rationale']}", "decision")

        # ── Step 5: Update reputation and motivation ──────────────────────────
        self._apply_outcomes(author, reviewers, result)

        summary = {
            "decision": decision,
            "rationale": result["rationale"],
            "reviews": reviews,
            "rebuttal": rebuttal,
            "avg_score": sum(r["score"] for r in reviews) / len(reviews),
            "author": author.name,
            "reviewers": [r.name for r in reviewers],
        }

        logger.info(f"Peer review done: {decision} (avg score {summary['avg_score']:.1f})")
        return summary

    def _apply_outcomes(self, author, reviewers: list, result: dict) -> None:
        decision = result["decision"]
        score_adj = result["score_adjustment"]

        # Author outcomes
        accepted = decision == "ACCEPT"
        author.update_motivation(
            credited=True,
            paper_accepted=accepted,
            task_difficulty=3,
        )
        author.update_reputation(score_adj)
        if hasattr(author, "grow_experience") and accepted:
            author.grow_experience(0.5)

        # Reviewer outcomes — reviewers gain small reputation for doing reviews
        for reviewer in reviewers:
            reviewer.update_reputation(0.1)
            reviewer.update_motivation(credited=True, paper_accepted=True, task_difficulty=2)
