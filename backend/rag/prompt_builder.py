"""Builds the system + user prompt sent to the local LLM from retrieved chunks."""
from typing import List

from backend.models.schemas import ChatMessage, SourceChunk

SYSTEM_PROMPT = """You are CodeSense, an assistant that helps developers understand a codebase.
Answer ONLY using the provided code context. If the context doesn't contain the answer,
say you don't have enough information rather than guessing.
Always reference the specific file path and line numbers when pointing to code.
Be concise and technical; prefer short explanations with a code reference over long prose."""


def build_context_block(sources: List[SourceChunk]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        header = f"[{i}] {s.file_path} (lines {s.line_start}-{s.line_end})"
        if s.function_name:
            header += f" — function/class: {s.function_name}"
        blocks.append(f"{header}\n```\n{s.snippet}\n```")
    return "\n\n".join(blocks)


def build_prompt(question: str, sources: List[SourceChunk], history: List[ChatMessage]) -> str:
    context = build_context_block(sources)

    history_block = ""
    if history:
        turns = [f"{m.role.upper()}: {m.content}" for m in history[-6:]]  # last 6 turns
        history_block = "Previous conversation:\n" + "\n".join(turns) + "\n\n"

    prompt = (
        f"{history_block}"
        f"Relevant code context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer, citing file paths and line numbers from the context above:"
    )
    return prompt
