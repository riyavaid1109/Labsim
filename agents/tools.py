"""
tools.py — structured tool calling for agents.

Instead of freeform text, agents can act through typed tool calls:

    search_kb(query)              — search their private knowledge base
    cite_source(claim, source)    — attach a citation to a claim
    flag_dispute(target, reason)  — formally dispute credit/feedback
    respond(text)                 — final answer, ends the loop

Design note: tool calls are implemented via JSON-schema-constrained output
(one action per turn) rather than the provider-native `tools` API. Plain
llama3 has no native tool support in Ollama, and constrained decoding to a
schema works identically across Ollama / OpenAI / Anthropic — so this stays
model-agnostic at the cost of one schema per turn. The loop is a standard
ReAct-style cycle: model picks an action → runtime executes it → observation
is appended → repeat until `respond` or max_steps.
"""

import json
import logging

from utils import query_llm_json

logger = logging.getLogger("lab_sim")

MAX_TOOL_STEPS = 4

# One-action-per-turn schema. `args` is a free object because Ollama's
# constrained decoding can't express per-tool arg unions cleanly; arg
# validation happens in the executors below.
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": ["search_kb", "cite_source", "flag_dispute", "respond"],
        },
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}

TOOL_DESCRIPTIONS = """You can act by calling ONE tool per turn:

- search_kb: search your private notes/papers. args: {"query": "<search terms>"}
- cite_source: attach a citation to a claim you are making. args: {"claim": "...", "source": "<paper/note the claim comes from>"}
- flag_dispute: formally dispute credit assignment or a review you believe is unfair. args: {"target": "<who/what>", "reason": "..."}
- respond: give your final answer and finish. args: {"text": "<your response, under 80 words>"}

Respond with a JSON object: {"tool": <name>, "args": {...}}"""


class ToolRuntime:
    """Executes tool calls for one agent and records side effects."""

    def __init__(self, agent, knowledge_space=None):
        self.agent = agent
        self.ks = knowledge_space
        self.citations: list[dict] = []
        self.disputes: list[dict] = []
        self.tool_trace: list[dict] = []

    def execute(self, tool: str, args: dict) -> str:
        """Run one tool call, return the observation string fed back to
        the model."""
        self.tool_trace.append({"tool": tool, "args": args})
        try:
            if tool == "search_kb":
                query = str(args.get("query", ""))
                if not query:
                    return "search_kb error: empty query"
                result = self.agent.kb.retrieve_as_string(query, k=3)
                return f"KB results:\n{result}" if result else "KB results: (nothing relevant)"

            if tool == "cite_source":
                citation = {
                    "claim": str(args.get("claim", "")),
                    "source": str(args.get("source", "")),
                }
                self.citations.append(citation)
                return f"Citation recorded: '{citation['claim'][:60]}' ← {citation['source']}"

            if tool == "flag_dispute":
                dispute = {
                    "agent": self.agent.name,
                    "target": str(args.get("target", "")),
                    "reason": str(args.get("reason", "")),
                }
                self.disputes.append(dispute)
                if self.ks is not None:
                    self.ks.post(
                        self.agent.name,
                        f"DISPUTE re {dispute['target']}: {dispute['reason']}",
                        "dispute",
                    )
                return f"Dispute filed against {dispute['target']}."

            return f"Unknown tool: {tool}"
        except Exception as e:
            logger.error(f"Tool execution error ({tool}): {e}")
            return f"{tool} error: {e}"


def run_tool_loop(agent, task_prompt: str, knowledge_space=None,
                  max_steps: int = MAX_TOOL_STEPS) -> dict:
    """
    ReAct-style loop: the agent takes up to max_steps tool actions, then
    must respond. Returns:

        {"response": str, "citations": [...], "disputes": [...],
         "tool_trace": [...]}

    Falls back to plain agent.speak() if structured calls fail, so a weak
    model degrades gracefully instead of crashing the simulation.
    """
    runtime = ToolRuntime(agent, knowledge_space)
    transcript = ""

    for step in range(max_steps):
        remaining = max_steps - step - 1
        prompt = (
            f"{agent.persona_block()}\n\n{TOOL_DESCRIPTIONS}\n\n"
            f"Task:\n{task_prompt}\n"
        )
        if transcript:
            prompt += f"\nYour actions so far:\n{transcript}\n"
        if remaining == 0:
            prompt += "\nYou MUST use the respond tool now."

        call = query_llm_json(prompt, TOOL_CALL_SCHEMA,
                              caller=f"{agent.name}.tool_loop")
        if call is None or "tool" not in call:
            logger.warning(f"{agent.name}: tool loop fell back to freeform speak()")
            return {
                "response": agent.speak(task_prompt,
                                        agent.retrieve_context(task_prompt)),
                "citations": runtime.citations,
                "disputes": runtime.disputes,
                "tool_trace": runtime.tool_trace,
            }

        tool, args = call["tool"], call.get("args", {}) or {}
        if tool == "respond":
            runtime.tool_trace.append({"tool": "respond", "args": args})
            return {
                "response": str(args.get("text", "")).strip() or "[empty response]",
                "citations": runtime.citations,
                "disputes": runtime.disputes,
                "tool_trace": runtime.tool_trace,
            }

        observation = runtime.execute(tool, args)
        transcript += f"\n[{tool}({json.dumps(args)[:120]})]\n→ {observation}\n"

    # Loop exhausted without an explicit respond.
    return {
        "response": agent.speak(task_prompt, transcript),
        "citations": runtime.citations,
        "disputes": runtime.disputes,
        "tool_trace": runtime.tool_trace,
    }
