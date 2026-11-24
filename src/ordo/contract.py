"""Contrato do Ordo 3B: system prompt, formato da mensagem e modos.

Tem que bater exatamente com o que o modelo publicado aprendeu (Modelfile do repo
devlucascfarias/ordo-3b no Hugging Face). Mudar algo aqui exige retreinar.
"""
from __future__ import annotations

from enum import Enum

SYSTEM_PROMPT = (
    "You are Ordo 3B, a surgical Neural Context Garbage Collector for coding agents. "
    "Your sole role is to compress noisy terminal outputs, source files, and agent trajectories "
    "into lossless causal YAML based on the agent's <intent>.\n"
    "RULES:\n"
    "1. Output ONLY valid YAML matching the schema for <gc_mode>. Never include markdown fences, "
    "greetings, or explanations.\n"
    "2. Never invent solutions or guess fixes. Report only factual traces present in <raw_input>.\n"
    "3. Preserve 100% of exact file paths, line numbers, symbol names, and error messages tied to <intent>.\n"
    "4. If status is PASS and no errors exist, omit causal_trace and exact_snippets completely."
)


class GcMode(str, Enum):
    TOOL_OUTPUT = "TOOL_OUTPUT"
    CODE_SLICE = "CODE_SLICE"
    MEMORY_COMPACT = "MEMORY_COMPACT"


def build_user_message(mode: GcMode | str, intent: str, raw_input: str) -> str:
    mode = GcMode(mode)
    return f"<gc_mode>{mode.value}</gc_mode>\n<intent>{intent}</intent>\n<raw_input>{raw_input}</raw_input>"


def build_messages(mode: GcMode | str, intent: str, raw_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(mode, intent, raw_input)},
    ]


def number_lines(text: str, start: int = 1) -> str:
    """Prefixa linhas no formato usado em exact_snippets (`NN | código`), 1-based."""
    lines = text.split("\n")  # como um editor conta (splitlines também quebra em \x0c, \x1c...)
    if lines and lines[-1] == "":
        lines.pop()
    width = len(str(start + len(lines) - 1)) if lines else 1
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, start))
