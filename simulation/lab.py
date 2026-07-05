"""
Lab — main simulation class.

Each timestep:
  1. A research problem is loaded from /problems/
  2. PI assigns a lead agent
  3. All agents discuss the problem (RAG-grounded)
  4. Lead agent writes a draft
  5. Peer review runs
  6. Outcomes update agent state
  7. Metrics are logged

Tracks: experience, motivation, reputation, credit disputes,
bus factor, message entropy, review scores over time.
"""

import os
import logging
# from agents import postdoc
from simulation.discussion import run_discussion
from simulation.peer_review import PeerReview
from simulation.knowledge_space import KnowledgeSpace

logger = logging.getLogger("lab_sim")


class Lab:
    def __init__(
        self,
        phd_students: list,
        postdocs: list,
        pi,
        problems_dir: str = "./problems",
        persist_dir: str = "./chroma_db",
    ):
        self.students = phd_students
        self.postdocs = postdocs
        self.pi = pi
        self.all_agents = phd_students + postdocs + [pi]
        self.problems_dir = problems_dir

        self.ks = KnowledgeSpace(persist_dir=persist_dir)
        self.peer_review = PeerReview(self.ks)

        self.timestep = 0
        self.history: list[dict] = []   # one entry per timestep

    def load_problems(self) -> list[str]:
        problems = []
        if not os.path.exists(self.problems_dir):
            logger.warning(f"Problems directory not found: {self.problems_dir}")
            return problems
        for fname in sorted(os.listdir(self.problems_dir)):
            if fname.endswith(".md"):
                with open(os.path.join(self.problems_dir, fname)) as f:
                    problems.append(f.read())
        return problems

    def run(self, n_timesteps: int = None) -> list[dict]:
        problems = self.load_problems()
        if not problems:
            logger.error("No problems found — add .md files to ./problems/")
            return []

        to_run = problems[:n_timesteps] if n_timesteps else problems
        logger.info(f"Starting lab simulation: {len(to_run)} problems, {len(self.all_agents)} agents")

        for problem in to_run:
            self.timestep += 1
            self.ks.current_timestep = self.timestep
            logger.info(f"\n{'='*60}\nTimestep {self.timestep}\n{'='*60}")
            result = self._run_timestep(problem)
            self.history.append(result)
            self._print_timestep_summary(result)

        return self.history

    def _run_timestep(self, problem_description: str) -> dict:
        # ── 1. Assign lead agent ──────────────────────────────────────────────
        candidates = [a for a in self.students + self.postdocs if a.available]
        lead = self.pi.assign_problem(candidates, problem_description)
        if not lead:
            logger.warning("No available agents — skipping timestep")
            return {"timestep": self.timestep, "skipped": True}

        logger.info(f"Lead agent: {lead.name}")
        self.ks.post("system", f"Problem assigned to {lead.name}", "system")

        # ── 2. Lab discussion ─────────────────────────────────────────────────
        # PI joins discussions only occasionally (low bandwidth)
        import random
        discussants = self.students + self.postdocs
        if random.random() < 0.3:   # PI joins 30% of discussions
            discussants = discussants + [self.pi]

        discussion_history = run_discussion(
            problem_description=problem_description,
            agents=discussants,
            knowledge_space=self.ks,
            rounds=1,
        )

        # ── 3. Lead writes draft ──────────────────────────────────────────────
        experiment_notes = lead.retrieve_context(problem_description, k=5)
        draft = lead.write_draft(problem_description, experiment_notes)
        self.ks.post(lead.name, f"[DRAFT]\n{draft}", "discussion")
        self.ks.index_to_shared([draft], [{"type": "draft", "author": lead.name}])
        logger.info(f"Draft written by {lead.name}")

        # ── 4. Authorship resolution (if postdoc also contributed strongly) ──
        authorship = self._resolve_authorship(lead, problem_description, draft)

        # ── 5. Peer review ────────────────────────────────────────────────────
        review_result = self.peer_review.run(
            paper_draft=draft,
            author=lead,
            pi=self.pi,
            all_agents=self.all_agents,
            n_reviewers=min(2, len(self.postdocs)),
        )

        # ── 6. Update non-selected agents' motivation ─────────────────────────
        for agent in candidates:
            if agent.id != lead.id:
                agent.update_motivation(ignored=True, task_difficulty=2)

        # ── 7. Snapshot metrics ───────────────────────────────────────────────
        # metrics = self.ks.summary(self.all_agents)
        metrics = self.ks.summary(self.all_agents)
        metrics["entropy_this_timestep"] = self.ks.message_entropy_this_timestep(self.timestep)
        metrics["agent_states"] = [
            {
                "name": a.name,
                "role": a.role,
                "motivation": a.motivation,
                "reputation": a.reputation,
                "experience": a.experience,
            }
            for a in self.all_agents
        ]

        return {
            "timestep": self.timestep,
            "problem": problem_description[:80],
            "lead": lead.name,
            "authorship": authorship,
            "review_decision": review_result.get("decision"),
            "avg_review_score": review_result.get("avg_score", 0),
            "metrics": metrics,
            "skipped": False,
        }

    def _resolve_authorship(self, lead, problem_description: str, draft: str) -> list[dict]:
        """Build authorship claim list and let PI resolve it."""
        # candidates = []
        # # Lead always has a base claim
        # candidates.append({"agent": lead, "claim_score": 0.8})

        # # Postdocs may also claim authorship based on their KB overlap
        # for postdoc in self.postdocs:
        #     claim = postdoc.claim_authorship(draft)
        #     if claim > 0.3:
        #         candidates.append({"agent": postdoc, "claim_score": claim})
        base_claim = 0.8 if lead.role != "PhD student" else 0.5
        candidates = [{"agent": lead, "claim_score": base_claim}]

        # candidates = [{"agent": lead, "claim_score": 0.8}]
        seen_ids = {lead.id}
        for pd in self.postdocs:
            if pd.id in seen_ids:
                continue
            claim = pd.claim_authorship(draft)
            if claim > 0.3:
                candidates.append({"agent": pd, "claim_score": claim})
                seen_ids.add(pd.id)
                if claim > 0.6:
                    pd.update_reputation(0.05)

        ordered = self.pi.resolve_authorship(candidates)

        # Credit disputes: if a student's idea was ranked below a postdoc with higher PI favor
        # for entry in ordered:
        #     if entry["authorship_position"] > 1 and hasattr(entry["agent"], "credit_disputes"):
        #         # Check if a less-experienced agent ranked above them
        #         above = [e for e in ordered if e["authorship_position"] < entry["authorship_position"]]
        #         for a in above:
        #             if entry["agent"].id == lead.id and entry["authorship_position"] > 1:
        #                 entry["agent"].record_credit_dispute()

        for entry in ordered:
            if entry["agent"].id == lead.id and entry["authorship_position"] > 1:
                if hasattr(entry["agent"], "record_credit_dispute"):
                    entry["agent"].record_credit_dispute()

        self.ks.post(
            "system",
            "Authorship: " + ", ".join(
                f"{e['authorship_position']}. {e['agent'].name}"
                for e in ordered
            ),
            "system",
        )
        return [{"name": e["agent"].name, "position": e["authorship_position"]} for e in ordered]

    def _print_timestep_summary(self, result: dict) -> None:
        if result.get("skipped"):
            print(f"\n[T{result['timestep']}] Skipped — no available agents")
            return
        print(f"\n[T{self.timestep}] Lead: {result['lead']} | "
              f"Decision: {result['review_decision']} | "
              f"Avg score: {result['avg_review_score']:.1f}")
        m = result["metrics"]
        print(f"  Bus factor: {m['bus_factor']} | "
              f"Entropy: {m['message_entropy']} | "
              f"Credit disputes: {m['credit_dispute_rate']:.0%}")
        for s in result["metrics"]["agent_states"]:
            print(f"  {s['name']} ({s['role']}): "
                  f"mot={s['motivation']:.1f} rep={s['reputation']:.1f} exp={s['experience']:.1f}")
