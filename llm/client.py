"""
client.py — model-agnostic LLM orchestration layer.

One interface, three providers:

    client = get_client("ollama")          # or "openai", "anthropic"
    text   = client.generate(prompt)
    data   = client.generate_json(prompt, schema)   # schema-constrained dict

Provider selection can also come from the LAB_SIM_PROVIDER env var, so the
whole simulation can be pointed at a different backend without touching code.
OpenAI / Anthropic SDKs are imported lazily — only Ollama is a hard dependency.

Structured outputs:
  - Ollama:    native `format=<json schema>` constraint (ollama >= 0.5)
  - OpenAI:    response_format json_schema (strict mode)
  - Anthropic: schema embedded in the prompt + robust JSON extraction
All three funnel through _parse_json(), which strips code fences and finds
the outermost JSON object, so a slightly sloppy model response still parses.

Every call is traced via llm.observability (latency, token counts, caller).
"""

import os
import json
import re
import logging

import ollama

from llm.observability import traced_call

logger = logging.getLogger("lab_sim")

DEFAULT_PROVIDER = os.environ.get("LAB_SIM_PROVIDER", "ollama")
DEFAULT_MODELS = {
    "ollama": os.environ.get("LAB_SIM_OLLAMA_MODEL", "llama3"),
    "openai": os.environ.get("LAB_SIM_OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.environ.get("LAB_SIM_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
}


def _parse_json(text: str) -> dict:
    """Parse a JSON object out of an LLM response, tolerating code fences
    and surrounding prose. Raises ValueError if nothing parses."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"No JSON object found in response: {text[:200]}")


class BaseLLMClient:
    provider = "base"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODELS.get(self.provider, "")

    def generate(self, prompt: str, caller: str | None = None) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, schema: dict,
                      caller: str | None = None) -> dict:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    provider = "ollama"

    def generate(self, prompt: str, caller: str | None = None) -> str:
        with traced_call(self.provider, self.model, prompt, caller) as trace:
            resp = ollama.generate(model=self.model, prompt=prompt)
            trace["response"] = resp["response"].strip()
            trace["prompt_tokens"] = resp.get("prompt_eval_count")
            trace["completion_tokens"] = resp.get("eval_count")
            return trace["response"]

    def generate_json(self, prompt: str, schema: dict,
                      caller: str | None = None) -> dict:
        with traced_call(self.provider, self.model, prompt, caller,
                         structured=True) as trace:
            # Ollama constrains decoding to the schema natively.
            resp = ollama.generate(model=self.model, prompt=prompt, format=schema)
            trace["response"] = resp["response"].strip()
            trace["prompt_tokens"] = resp.get("prompt_eval_count")
            trace["completion_tokens"] = resp.get("eval_count")
            return _parse_json(trace["response"])


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        from openai import OpenAI  # lazy — optional dependency
        self._client = OpenAI()

    def generate(self, prompt: str, caller: str | None = None) -> str:
        with traced_call(self.provider, self.model, prompt, caller) as trace:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            trace["response"] = resp.choices[0].message.content.strip()
            if resp.usage:
                trace["prompt_tokens"] = resp.usage.prompt_tokens
                trace["completion_tokens"] = resp.usage.completion_tokens
            return trace["response"]

    def generate_json(self, prompt: str, schema: dict,
                      caller: str | None = None) -> dict:
        with traced_call(self.provider, self.model, prompt, caller,
                         structured=True) as trace:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": True},
                },
            )
            trace["response"] = resp.choices[0].message.content.strip()
            if resp.usage:
                trace["prompt_tokens"] = resp.usage.prompt_tokens
                trace["completion_tokens"] = resp.usage.completion_tokens
            return _parse_json(trace["response"])


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        import anthropic  # lazy — optional dependency
        self._client = anthropic.Anthropic()

    def _call(self, prompt: str, trace: dict) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        trace["response"] = resp.content[0].text.strip()
        trace["prompt_tokens"] = resp.usage.input_tokens
        trace["completion_tokens"] = resp.usage.output_tokens
        return trace["response"]

    def generate(self, prompt: str, caller: str | None = None) -> str:
        with traced_call(self.provider, self.model, prompt, caller) as trace:
            return self._call(prompt, trace)

    def generate_json(self, prompt: str, schema: dict,
                      caller: str | None = None) -> dict:
        json_prompt = (
            f"{prompt}\n\nRespond with ONLY a JSON object matching this schema, "
            f"no prose, no code fences:\n{json.dumps(schema)}"
        )
        with traced_call(self.provider, self.model, json_prompt, caller,
                         structured=True) as trace:
            return _parse_json(self._call(json_prompt, trace))


_REGISTRY = {
    "ollama": OllamaClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

_client_cache: dict[tuple[str, str | None], BaseLLMClient] = {}


def get_client(provider: str | None = None, model: str | None = None) -> BaseLLMClient:
    """Return a (cached) client for the given provider. Defaults to the
    LAB_SIM_PROVIDER env var, falling back to Ollama."""
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(_REGISTRY)}")
    key = (provider, model)
    if key not in _client_cache:
        _client_cache[key] = _REGISTRY[provider](model)
    return _client_cache[key]
