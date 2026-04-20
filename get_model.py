"""
Unified LLM / embedding abstraction — mirrors ndp_agents/aoi/get_model.py,
extended with call_embedding() for RAG pipelines.

Configuration is loaded from models.yaml with environment-variable expansion.
All calls go through the OpenAI SDK, so any OpenAI-compatible endpoint works
(NRP's ellm, OpenAI, proxies, etc.).

Usage:
    from get_model import load_models_config, call_model, call_embedding

    cfg = load_models_config("models.yaml")
    chat = cfg["qwen3"]

    reply = call_model(
        model_name=chat["model"],
        prompt="Hello",
        api_key=chat["api_key"],
        base_url=chat["base_url"],
    )

    # Only works if an embed/ entry exists in models.yaml
    embed = cfg["openai/text-embedding-3-small"]
    vecs = call_embedding(
        texts=["doc one", "doc two"],
        model_name=embed["model"],
        api_key=embed["api_key"],
        base_url=embed["base_url"],
    )
"""

import os
import re
from pathlib import Path
from typing import List, Optional

import yaml
from openai import OpenAI


def _expand_env_vars(value):
    """Expand ${VAR} patterns in a string using environment variables."""
    if not isinstance(value, str):
        return value

    def replace_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r"\$\{(\w+)\}", replace_var, value)


def load_models_config(path: str) -> dict:
    """Load model configurations from a YAML file.

    Resolves `path` relative to this file (so the notebook and scripts can
    both call `load_models_config("models.yaml")` regardless of CWD).

    Each entry must have `model`; `task` defaults to "chat" if missing.
    """
    config_path = Path(__file__).parent / Path(path)
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    available = {}
    for key, v in raw.items():
        if not isinstance(v, dict) or "model" not in v:
            continue
        entry = {k: _expand_env_vars(val) for k, val in v.items()}
        entry.setdefault("task", "chat")
        available[key] = entry
    return available


def filter_by_task(config: dict, task: str) -> dict:
    """Return only entries whose `task` matches (e.g. 'chat' or 'embed')."""
    return {k: v for k, v in config.items() if v.get("task") == task}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def call_model(
    model_name: str,
    prompt: Optional[str] = None,
    history: Optional[list] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = 2048,
) -> str:
    """Call any OpenAI-compatible chat completion endpoint.

    Either `prompt` (single-turn) or `history` (multi-turn) must be provided.

    `max_tokens` defaults to 2048 (generous) so students can swap between
    reasoning models (qwen3, which spend tokens on chain-of-thought) and
    plain models (gemma) without one of them silently returning empty content.
    Pass `max_tokens=None` to use the provider default.
    """
    if history is not None:
        messages = list(history)
        if prompt:
            messages.append({"role": "user", "content": prompt})
    else:
        if prompt is None:
            raise ValueError("Either prompt or history must be provided.")
        messages = [{"role": "user", "content": prompt}]

    client = OpenAI(api_key=api_key or "", base_url=base_url)
    kwargs = dict(model=model_name, messages=messages, temperature=temperature)
    if seed is not None:
        kwargs["seed"] = seed
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)

    choice = resp.choices[0]
    content = choice.message.content
    if content:
        return content

    # Some reasoning-style models (e.g. qwen3 on NRP) emit chain-of-thought in
    # a separate `reasoning` field and leave `content` None — especially when
    # max_tokens is too small and they finish_reason="length" mid-thought.
    reasoning = getattr(choice.message, "reasoning", None)
    if choice.finish_reason == "length" and reasoning:
        raise ValueError(
            f"Model {model_name!r} returned no content "
            f"(finish_reason=length). It looks like a reasoning model that "
            f"exhausted max_tokens before producing a final answer. "
            f"Raise max_tokens or use a non-reasoning chat model."
        )
    raise ValueError(
        f"Model {model_name!r} returned empty content "
        f"(finish_reason={choice.finish_reason})."
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def call_embedding(
    texts: List[str],
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[List[float]]:
    """Embed a list of strings via an OpenAI-compatible /v1/embeddings endpoint.

    Returns a list of vectors aligned with `texts`.
    """
    if not texts:
        return []
    client = OpenAI(api_key=api_key or "", base_url=base_url)
    resp = client.embeddings.create(model=model_name, input=texts)
    return [d.embedding for d in resp.data]
