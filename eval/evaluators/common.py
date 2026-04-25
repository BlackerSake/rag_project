"""Shared helpers for RAG evaluators."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

LLMClient = Callable[[str], Awaitable[str] | str]


def clamp_score(value: Any, default: float = 0.0) -> float:
    """Convert a value to a score in [0.0, 1.0].

    Args:
        value: Value returned by an evaluator or LLM.
        default: Score used when conversion fails.

    Returns:
        A float clipped to the closed interval [0.0, 1.0].

    Raises:
        No exceptions are raised; invalid values use ``default``.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid score value %r, using default %.3f", value, default)
        score = default
    return max(0.0, min(1.0, score))


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers with zero protection.

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        ``numerator / denominator`` when denominator is non-zero, otherwise 0.0.

    Raises:
        No exceptions are raised.
    """
    if denominator == 0:
        logger.warning("Division by zero avoided; returning 0.0")
        return 0.0
    return numerator / denominator


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output.

    Args:
        text: Raw LLM response, possibly wrapped in Markdown fences.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no JSON object can be found.
        json.JSONDecodeError: If JSON syntax is invalid.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response")
        parsed = json.loads(raw[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed


async def call_llm_json(
    llm_client: LLMClient,
    prompt: str,
    fallback: dict[str, Any],
    call_name: str,
) -> dict[str, Any]:
    """Call an LLM client and parse a JSON object with fallback handling.

    Args:
        llm_client: Async or sync callable accepting a prompt and returning text.
        prompt: Prompt sent to the LLM.
        fallback: Default object returned on call or parse failure.
        call_name: Name used in log records.

    Returns:
        Parsed JSON object or a copy of ``fallback``.

    Raises:
        No exceptions are raised; all failures are logged and downgraded.
    """
    try:
        logger.info("Calling LLM for %s", call_name)
        response = llm_client(prompt)
        if inspect.isawaitable(response):
            response = await response
        return extract_json_object(str(response))
    except Exception as exc:
        logger.error("LLM JSON call failed for %s: %s", call_name, exc)
        return dict(fallback)


async def maybe_await(value: Any) -> Any:
    """Await a value only when it is awaitable.

    Args:
        value: Any value returned by an injected dependency.

    Returns:
        Awaited result for awaitables, otherwise the original value.

    Raises:
        Propagates exceptions raised by the awaitable.
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def run_sync(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous function in the default executor.

    Args:
        func: Callable to execute.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Callable result.

    Raises:
        Propagates exceptions raised by ``func``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def mean(values: list[float]) -> float:
    """Compute the arithmetic mean with empty-list protection.

    Args:
        values: List of numeric values.

    Returns:
        Average value or 0.0 for an empty list.

    Raises:
        No exceptions are raised.
    """
    if not values:
        logger.warning("Mean requested for an empty list; returning 0.0")
        return 0.0
    return sum(values) / len(values)

