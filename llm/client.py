"""Provider-agnostic LLM client (spec section 5).

Default DeepSeek; swap to OpenAI/Anthropic by changing configs/default.yaml `model:`.
DeepSeek and OpenAI share the OpenAI chat-completions shape; Anthropic uses its own.

Reproducibility: provider/model/temperature/seed are recorded by the caller. Token usage
is accumulated here so an experiment can enforce the (advisor-TBD) budget cap.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

# Rough public prices (USD per 1M tokens) for a cost estimate; override via cfg["prices"].
_DEFAULT_PRICES = {
    "deepseek-chat":     {"in": 0.27, "out": 1.10},
    "deepseek-reasoner": {"in": 0.55, "out": 2.19},
}


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines -> os.environ (does not overwrite existing)."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prices: dict = field(default_factory=dict)

    def add(self, pt: int, ct: int):
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct

    def usd(self, model: str) -> float:
        p = self.prices.get(model) or _DEFAULT_PRICES.get(model, {"in": 0.0, "out": 0.0})
        return self.prompt_tokens / 1e6 * p["in"] + self.completion_tokens / 1e6 * p["out"]


class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: dict, budget_usd: float | None = None):
        load_dotenv()
        self.provider = cfg["provider"]
        self.model = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.7))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.api_key = os.environ.get(cfg["api_key_env"], "")
        if not self.api_key:
            raise RuntimeError(
                f"missing API key: set {cfg['api_key_env']} (e.g. in .env). "
                f"provider={self.provider}"
            )
        self.budget_usd = budget_usd
        self.usage = Usage(prices=cfg.get("prices", {}))

        if self.provider in ("deepseek", "openai"):
            base = {"deepseek": "https://api.deepseek.com",
                    "openai": "https://api.openai.com/v1"}[self.provider]
            self._url = base + "/chat/completions"
        elif self.provider == "anthropic":
            self._url = "https://api.anthropic.com/v1/messages"
        else:
            raise ValueError(f"unknown provider {self.provider}")

    # -- public ----------------------------------------------------------------
    def complete(self, messages, *, json_mode: bool = False,
                 temperature: float | None = None, max_tokens: int | None = None,
                 retries: int = 4, timeout: float = 180.0) -> str:
        """messages = list of (role, content). Returns the assistant text."""
        if self.budget_usd is not None and self.usage.usd(self.model) >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self.usage.usd(self.model):.2f} >= cap ${self.budget_usd:.2f}")

        temp = self.temperature if temperature is None else temperature
        mt = max_tokens or self.max_tokens
        if self.provider == "anthropic":
            return self._anthropic(messages, json_mode, temp, mt, retries, timeout)
        return self._openai_like(messages, json_mode, temp, mt, retries, timeout)

    # -- backends --------------------------------------------------------------
    def _openai_like(self, messages, json_mode, temp, mt, retries, timeout) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": [{"role": r, "content": c} for r, c in messages],
            "temperature": temp,
            "max_tokens": mt,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post(headers, body, retries, timeout)
        u = data.get("usage", {})
        self.usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return data["choices"][0]["message"]["content"]

    def _anthropic(self, messages, json_mode, temp, mt, retries, timeout) -> str:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        system = "\n\n".join(c for r, c in messages if r == "system")
        body = {
            "model": self.model, "max_tokens": mt, "temperature": temp,
            "messages": [{"role": r, "content": c} for r, c in messages if r != "system"],
        }
        if system:
            body["system"] = system
        data = self._post(headers, body, retries, timeout)
        u = data.get("usage", {})
        self.usage.add(u.get("input_tokens", 0), u.get("output_tokens", 0))
        return "".join(blk.get("text", "") for blk in data.get("content", []))

    def _post(self, headers, body, retries, timeout) -> dict:
        last = None
        for attempt in range(retries):
            try:
                resp = requests.post(self._url, headers=headers, json=body, timeout=timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (401, 403):
                    raise RuntimeError(f"auth failed ({resp.status_code}): {resp.text[:200]}")
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20))  # exponential backoff
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")
