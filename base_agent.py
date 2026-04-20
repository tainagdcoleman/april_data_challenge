"""
BaseAgent — minimal base class for agents, mirroring ndp_agents/aoi/base_agent.py.

Provides:
- LLM configuration and calling (wraps get_model.call_model)
- Structured JSON output parsing
- Conversation history management
- Standard result format ({"ok": bool, ...})
"""

import json
from typing import Any, Dict, List, Optional

from get_model import call_model


class BaseAgent:
    name: str = "BaseAgent"
    system_prompt: str = "You are a helpful assistant."

    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []

    # ---- LLM helpers ----------------------------------------------------

    def call_llm(
        self,
        prompt: str = None,
        history: List[Dict] = None,
        temperature: float = None,
    ) -> str:
        return call_model(
            model_name=self.model_name,
            prompt=prompt,
            history=history,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature if temperature is not None else self.temperature,
        )

    def call_llm_json(
        self,
        prompt: str = None,
        history: List[Dict] = None,
        temperature: float = None,
    ) -> Dict[str, Any]:
        raw = self.call_llm(prompt=prompt, history=history, temperature=temperature)
        return self._parse_json(raw)

    # ---- History management ---------------------------------------------

    def reset_history(self):
        self.history = []

    def add_system_message(self, content: str = None):
        msg = {"role": "system", "content": content or self.system_prompt}
        if self.history and self.history[0].get("role") == "system":
            self.history[0] = msg
        else:
            self.history.insert(0, msg)

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self.history)

    # ---- Standard result builders ---------------------------------------

    @staticmethod
    def ok_result(**kwargs) -> Dict[str, Any]:
        return {"ok": True, "error": None, **kwargs}

    @staticmethod
    def error_result(error: str, **kwargs) -> Dict[str, Any]:
        return {"ok": False, "error": error, **kwargs}

    # ---- Subclass entry point -------------------------------------------

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} must implement run()")

    # ---- Internal helpers -----------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from an LLM response."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("Could not extract JSON from LLM output", text, 0)
