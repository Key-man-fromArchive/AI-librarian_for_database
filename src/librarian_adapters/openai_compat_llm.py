"""Streaming against any OpenAI-compatible ``/chat/completions`` endpoint.

Covers OpenAI, Azure OpenAI, OpenRouter, Together, Groq, vLLM, Ollama and most
gateways. If you already have a model router, wrap it in :class:`CallableLLM`
instead of replacing it.

Only ``httpx`` is required — no vendor SDK, so nothing to keep in lockstep with
provider releases.

Errors are yielded as :class:`LLMError`, never raised. The turn needs to
distinguish "failed before producing anything" (fall back) from "failed
midway" (keep what we have), and an exception erases that distinction.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import httpx

from librarian_core.ports import LLMChunk, LLMError, LLMEvent, LLMRequest

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


class OpenAICompatLLM:
    """An :class:`LLMPort` for OpenAI-compatible chat completions.

    The API key is read from the environment by default and never logged. Do not
    pass it as a literal in code that gets committed.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        extra_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, request: LLMRequest) -> dict:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        payload.update(request.extra)
        return payload

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        if not self._api_key:
            yield LLMError("no API key configured", code="missing_credentials")
            return

        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        owns_client = self._client is None
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(request),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    # Surface the provider's own words: "temperature is not
                    # supported for this model" is diagnosable, "HTTP 400" is not.
                    yield LLMError(
                        _provider_message(body) or f"HTTP {response.status_code}",
                        code=f"http_{response.status_code}",
                    )
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and payload.get("error"):
                        error = payload["error"]
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        yield LLMError(str(message or "provider error"))
                        return
                    for choice in payload.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            yield LLMChunk(piece)
        except httpx.HTTPError as exc:
            yield LLMError(f"{type(exc).__name__}: {exc}", code="transport_error")
        finally:
            if owns_client:
                await client.aclose()


def _provider_message(body: str) -> str:
    """Best-effort extraction of the provider's error text."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:300]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "")[:300]
    return str(error or "")[:300]


class CallableLLM:
    """Adapts an existing async generator into an :class:`LLMPort`.

    Use this to keep your current model router — provider fallbacks, budget
    accounting, egress policy — and still gain the librarian turn::

        llm = CallableLLM(lambda req: my_router.stream(req.model, req.messages))

    The wrapped callable may yield plain strings; they become text chunks. Yield
    :class:`LLMError` for provider failures so fallback still works.
    """

    def __init__(
        self, fn: Callable[[LLMRequest], AsyncIterator[object] | Awaitable[Sequence[object]]]
    ) -> None:
        self._fn = fn

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        try:
            result = self._fn(request)
            if hasattr(result, "__aiter__"):
                async for item in result:  # type: ignore[union-attr]
                    event = _coerce(item)
                    if event is not None:
                        yield event
                return
            for item in await result:  # type: ignore[misc]
                event = _coerce(item)
                if event is not None:
                    yield event
        except Exception as exc:  # noqa: BLE001 — the port contract is to yield, not raise
            yield LLMError(f"{type(exc).__name__}: {exc}")


def _coerce(item: object) -> LLMEvent | None:
    if isinstance(item, LLMChunk | LLMError):
        return item
    if isinstance(item, str):
        return LLMChunk(item) if item else None
    return None


__all__ = ["CallableLLM", "OpenAICompatLLM"]
