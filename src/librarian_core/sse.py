"""Server-Sent Events encoding, and the error-extraction helper.

The wire format is deliberately boring::

    event: trace\\ndata: {"state": "retrieving"}\\n\\n
    event: citations\\ndata: {"citations": [...]}\\n\\n
    data: {"chunk": "partial text"}\\n\\n
    event: error\\ndata: {"code": "...", "message": "..."}\\n\\n
    data: [DONE]\\n\\n

Text chunks are bare ``data:`` lines; everything else is a named event. A client
that only understands ``data:`` still renders the answer correctly — it just
misses the extras.

``extract_error_message`` exists because of a specific production incident: a
consumer checked only for ``data:`` prefixes, so ``event: error`` blocks were
dropped, and a provider rejecting one request parameter surfaced to users as a
vague "no content generated". Any code consuming an SSE stream from another
producer should route error blocks through this helper.
"""

from __future__ import annotations

import json
from typing import Any

DONE = "data: [DONE]\n\n"

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # nginx buffers proxied responses by default, which holds the whole stream
    # until it completes and makes streaming look broken.
    "X-Accel-Buffering": "no",
}


def encode_chunk(text: str) -> str:
    """A fragment of answer text."""
    return f"data: {json.dumps({'chunk': text}, ensure_ascii=False)}\n\n"


def encode_event(name: str, payload: dict[str, Any]) -> str:
    """A named event carrying a JSON payload."""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def encode_error(message: str, code: str = "provider_error") -> str:
    return encode_event("error", {"code": code, "message": message})


def extract_error_message(block: str, default: str = "provider error") -> str:
    """Pull a human-readable reason out of an ``event: error`` SSE block.

    Falls back to the raw payload (truncated) when it is not valid JSON, because
    a truncated real message still beats a generic one.
    """
    for line in block.split("\n"):
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw[:300] or default
        if not isinstance(data, dict):
            return str(data)[:300] or default
        return str(data.get("message") or data.get("error") or data.get("code") or default)
    return default


__all__ = [
    "DONE",
    "SSE_HEADERS",
    "encode_chunk",
    "encode_error",
    "encode_event",
    "extract_error_message",
]
