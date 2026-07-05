"""
Discussion — runs a RAG-grounded multi-agent lab meeting.

Each agent speaks in turn. Their response is grounded in:
  1. Their private KB (what they've personally read / experimented on)
  2. The shared lab KB (past decisions, meeting notes)
  3. The conversation so far (context window)

The discussion history is indexed into the shared KB after completion
so future agents can retrieve it.
"""

import logging
from simulation.knowledge_space import KnowledgeSpace

logger = logging.getLogger("lab_sim")


def run_discussion(
    problem_description: str,
    agents: list,
    knowledge_space: KnowledgeSpace,
    rounds: int = 1,
) -> list[dict]:
    """
    Run a structured lab discussion on a research problem.

    Args:
        problem_description: The .md problem text (equivalent to LLAMOSC's issue)
        agents: Ordered list of agents who participate (PI participates rarely)
        knowledge_space: Shared lab KB + message log
        rounds: How many times each agent speaks

    Returns:
        discussion_history: list of {agent_name, role, content}
    """
    logger.info(f"Starting discussion on: {problem_description[:60]}...")
    discussion_history: list[dict] = []

    # Seed the shared KB with the problem statement
    knowledge_space.index_to_shared(
        [problem_description],
        [{"type": "problem", "timestep": knowledge_space.current_timestep}],
    )
    knowledge_space.post("system", f"Discussion started: {problem_description[:100]}", "system")

    for round_num in range(rounds):
        for agent in agents:
            # Build context: private KB + shared KB
            private_ctx = agent.retrieve_context(problem_description, k=3)
            shared_ctx = knowledge_space.retrieve_shared(problem_description, k=2)

            combined_context = ""
            if private_ctx:
                combined_context += f"[Your notes]\n{private_ctx}\n"
            if shared_ctx:
                combined_context += f"\n[Lab shared memory]\n{shared_ctx}\n"

            # Build conversation history string (last 6 turns)
            recent = discussion_history[-6:]
            history_text = "\n".join(
                f"{e['agent_name']} ({e['role']}): {e['content']}"
                for e in recent
            ) if recent else "No prior discussion."

            prompt = f"""Research problem being discussed:
{problem_description}

Discussion so far:
{history_text}

It's your turn. Contribute your perspective, propose an approach, or respond to what's been said.
Be specific. Under 80 words."""

            response = agent.speak(prompt, combined_context)

            entry = {
                "agent_name": agent.name,
                "role": agent.role,
                "content": response,
                "round": round_num + 1,
            }
            discussion_history.append(entry)
            knowledge_space.post(agent.name, response, "discussion")

            # Index each response into the shared KB so later speakers benefit
            knowledge_space.index_to_shared(
                [f"{agent.name} ({agent.role}): {response}"],
                [{"type": "discussion_turn", "agent": agent.name}],
            )

            logger.debug(f"  {agent.name}: {response[:60]}...")

    knowledge_space.post("system", "Discussion concluded.", "system")
    logger.info(f"Discussion complete — {len(discussion_history)} turns")
    return discussion_history
