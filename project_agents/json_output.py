"""Strict extraction of ONE JSON object from model text.

Rules (deliberately conservative - we never invent content the model did not produce):
* ``<think>...</think>`` reasoning blocks are removed first (a closed block only; an unclosed one means
  the answer was cut off).
* Markdown fences and prose before/after the object are tolerated.
* The object is delimited by a string-aware brace scanner, so braces inside quoted strings, escaped
  quotes and nested objects/arrays are handled correctly.
* If the outermost object never closes the output was TRUNCATED. We do NOT append closing brackets or
  quotes (that would fabricate semantic content); we raise ``LLMOutputError(kind="truncated")`` so the
  caller can retry within its bounded policy or fail clearly.
"""
from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.S | re.I)


class LLMOutputError(ValueError):
    """The model's text could not be turned into the expected object.

    kind: ``malformed`` (no/invalid JSON), ``truncated`` (object never closed) or ``invalid``
    (valid JSON that does not match the schema).
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def strip_reasoning(text: str) -> str:
    text = _THINK_CLOSED.sub("", text)
    if re.search(r"<think>", text, re.I):          # opened and never closed -> the answer was cut off
        raise LLMOutputError("truncated", "output ended inside a <think> block (no answer was produced)")
    return text


def _balanced_end(text: str, start: int) -> int:
    """Index just past the object that opens at ``start``, or -1 when it never closes."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def extract_json_text(raw: str) -> str:
    text = strip_reasoning(raw)
    start = text.find("{")
    if start == -1:
        raise LLMOutputError("malformed", "no JSON object found in the model output")
    end = _balanced_end(text, start)
    if end == -1:
        raise LLMOutputError("truncated", "the JSON object was cut off before it was complete")
    return text[start:end]


def parse_json_output(raw: Any, output_type: Type[T]) -> T:
    if isinstance(raw, output_type):
        return raw
    try:
        if isinstance(raw, dict):
            return output_type.model_validate(raw)
        data = json.loads(extract_json_text(str(raw)))
    except LLMOutputError:
        raise
    except json.JSONDecodeError as exc:
        raise LLMOutputError("malformed", f"invalid JSON ({exc.msg} at position {exc.pos})") from None
    except ValidationError as exc:
        raise LLMOutputError("invalid", _summarize(exc)) from None
    try:
        return output_type.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError("invalid", _summarize(exc)) from None


def _summarize(exc: ValidationError) -> str:
    parts = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(p) for p in e.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {e.get('msg', 'invalid')}")
    return "schema validation failed: " + "; ".join(parts)
