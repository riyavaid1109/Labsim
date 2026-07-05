"""
utils.py — shared helpers.

query_ollama() remains the single injection point for freeform LLM calls,
now backed by the model-agnostic client layer in llm/client.py — set
LAB_SIM_PROVIDER=openai|anthropic|ollama to swap the whole simulation onto
a different backend. query_llm_json() is the structured-output counterpart.
"""

import logging
from llm.client import get_client

logger = logging.getLogger("lab_sim")

DEFAULT_MODEL = "llama3"


def query_ollama(prompt: str, model: str | None = None, context: str = "",
                 caller: str | None = None) -> str:
    """
    Send a prompt to the configured LLM backend and return the response.
    context is prepended when provided — this is the RAG injection point.
    (Name kept for backward compatibility; provider may not be Ollama.)
    """
    full_prompt = prompt
    if context:
        full_prompt = f"[Context]\n{context}\n\n{prompt}"
    try:
        client = get_client(model=model) if model else get_client()
        return client.generate(full_prompt, caller=caller)
    except Exception as e:
        logger.error(f"LLM unavailable: {e}")
        return "[LLM unavailable]"


def query_llm_json(prompt: str, schema: dict, model: str | None = None,
                   context: str = "", caller: str | None = None) -> dict | None:
    """
    Structured-output counterpart of query_ollama — the response is
    constrained to the given JSON schema (natively on Ollama/OpenAI).
    Returns a parsed dict, or None on failure so callers can fall back.
    """
    full_prompt = prompt
    if context:
        full_prompt = f"[Context]\n{context}\n\n{prompt}"
    try:
        client = get_client(model=model) if model else get_client()
        return client.generate_json(full_prompt, schema, caller=caller)
    except Exception as e:
        logger.error(f"Structured LLM call failed: {e}")
        return None


def log(msg: str) -> None:
    logger.debug(msg)


def log_and_print(msg: str) -> None:
    logger.info(msg)
    print(msg)
