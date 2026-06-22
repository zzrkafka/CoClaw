"""Prompt loading: render a .jinja template and split it into role-tagged messages.

Templates use literal [SYSTEM] / [USER] / [ASSISTANT] marker lines (spec section 10); the
content between markers becomes a message. An [OUTPUT] block stays inside the system
message (it is an instruction, not a separate turn). Returns list[(role, content)] ready
for LLMClient.complete.

(Named prompt_loader, not prompts, to avoid clashing with the llm/prompts/ template dir.)
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_env = Environment(loader=FileSystemLoader(str(_PROMPT_DIR)), keep_trailing_newline=True)

_ROLES = {"[SYSTEM]": "system", "[USER]": "user", "[ASSISTANT]": "assistant"}


def _split_roles(text: str) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    role: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        marker = _ROLES.get(line.strip())
        if marker is not None:
            if role is not None:
                messages.append((role, "\n".join(buf).strip()))
            role, buf = marker, []
        else:
            buf.append(line)
    if role is not None:
        messages.append((role, "\n".join(buf).strip()))
    if not messages:  # template without markers -> single user message
        return [("user", text.strip())]
    return messages


def render_messages(name: str, **ctx) -> list[tuple[str, str]]:
    tmpl = _env.get_template(name if name.endswith(".jinja") else name + ".jinja")
    return _split_roles(tmpl.render(**ctx))


def render_text(name: str, **ctx) -> str:
    """Render without splitting (e.g. for a template used as a plain string)."""
    tmpl = _env.get_template(name if name.endswith(".jinja") else name + ".jinja")
    return tmpl.render(**ctx)
