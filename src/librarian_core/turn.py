"""One question → one streamed, cited answer.

This module is the reason the project exists. Wiring an LLM to a vector search
is an afternoon; the details below are what took production incidents to learn.

**A turn fails only if it produced nothing.** Errors that arrive after text has
already streamed are not failures — the user has a usable answer on screen, and
flagging it as failed is a lie that erodes trust in the whole feature. The
``emitted`` flag is checked before any error is surfaced or the message is
persisted as errored.

**Fallback runs only before first output.** Once tokens are on screen, switching
models would splice two different answers together. Before first output there is
nothing to lose, so a different-provider fallback converts a hard failure into a
slightly-degraded success — and the user is told it happened.

**Persistence is in a ``finally``.** A client that disconnects mid-stream must
still leave a complete stored message; otherwise the thread shows a truncated
answer forever, and rows stick in ``streaming`` state.

**The title is generated from the first question.** A sidebar of "New chat" rows
is unnavigable. Only an empty title on the first turn is filled, so a
user-chosen title is never overwritten.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_CONFIG, LibrarianConfig
from .ports import (
    ChatMessage,
    Citation,
    LLMError,
    LLMPort,
    LLMRequest,
    Principal,
    RetrievalPort,
    SessionStorePort,
    StoredMessage,
)
from .prompts import DEFAULT_SYSTEM_PROMPT, abstain_message, build_grounded_messages
from .rag import retrieve_context
from .sse import DONE, encode_chunk, encode_error, encode_event

logger = logging.getLogger(__name__)


@dataclass
class TurnRequest:
    session_id: str
    content: str
    model: str | None = None
    scope: Sequence[str] | None = None
    #: Extra provider arguments (e.g. reasoning effort). Passed through
    #: untouched; gate them by model in your adapter, not here.
    extra: dict[str, Any] = field(default_factory=dict)


def build_history(messages: Sequence[StoredMessage], *, max_chars: int) -> list[ChatMessage]:
    """Newest complete messages that fit the budget, in chronological order.

    Walks backwards so the most recent exchange always survives truncation, and
    skips anything not ``complete`` — replaying a half-streamed or errored turn
    as context teaches the model to imitate broken output.
    """
    selected: list[ChatMessage] = []
    used = 0
    for message in reversed(messages):
        if message.status != "complete":
            continue
        if message.role not in {"user", "assistant", "system"}:
            continue
        size = len(message.content)
        if selected and used + size > max_chars:
            break
        selected.append(ChatMessage(role=message.role, content=message.content))  # type: ignore[arg-type]
        used += size
    return list(reversed(selected))


def assemble_messages(
    *,
    history: Sequence[ChatMessage],
    passage_section: str,
    citations: Sequence[Citation],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    memory_section: str = "",
    rag_enabled: bool = True,
) -> list[ChatMessage]:
    """Order the system messages, then the conversation.

    Ordering is deliberate: safety directive first, then untrusted reference
    data, then the conversation. Retrieved passages and memories are DATA placed
    after the policy that governs how to treat them, never before it.
    """
    system: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
    if memory_section:
        system.append(ChatMessage(role="system", content=memory_section))

    grounded = build_grounded_messages(passage_section, citations)
    if grounded:
        system.extend(grounded)
    elif rag_enabled:
        # Retrieval ran and found nothing usable. Saying so beats silence: it
        # stops the model confabulating document contents to fill the gap.
        system.append(abstain_message())

    return [*system, *history]


class LibrarianTurn:
    """Runs turns against a set of adapters.

    Construct once per application (adapters are stateless) or per request if
    your store holds a transaction — the SQLAlchemy adapter does.
    """

    def __init__(
        self,
        *,
        store: SessionStorePort,
        llm: LLMPort,
        retrieval: RetrievalPort | None = None,
        config: LibrarianConfig = DEFAULT_CONFIG,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.store = store
        self.llm = llm
        self.retrieval = retrieval
        self.config = config
        self.system_prompt = system_prompt

    async def run(
        self,
        request: TurnRequest,
        *,
        principal: Principal,
        scope_candidates: Sequence[str] | None = None,
        memory_section: str = "",
    ) -> AsyncIterator[str]:
        """Execute a turn, yielding SSE blocks.

        The generator owns persistence: it stores the user message, streams the
        answer, and finalises the assistant message even if the consumer stops
        reading.
        """
        cfg = self.config

        await self.store.append_message(
            principal=principal,
            session_id=request.session_id,
            role="user",
            content=request.content,
        )

        session = await self.store.get_session(principal=principal, session_id=request.session_id)
        history_rows = await self.store.list_messages(principal=principal, session_id=request.session_id)
        if not session.title and len([m for m in history_rows if m.role == "user"]) == 1:
            await self.store.set_title(
                principal=principal,
                session_id=request.session_id,
                title=request.content.strip()[: cfg.auto_title_chars],
            )

        passage_section, citations = "", []
        if self.retrieval is not None and cfg.rag_enabled:
            passage_section, citations = await retrieve_context(
                self.retrieval,
                request.content,
                principal=principal,
                config=cfg,
                scope=request.scope,
                scope_candidates=scope_candidates,
            )

        assistant = await self.store.append_message(
            principal=principal,
            session_id=request.session_id,
            role="assistant",
            status="streaming",
        )
        # Commit the provisional row before streaming: a client that polls the
        # thread while tokens arrive must see the turn exists.
        await self.store.commit()

        messages = assemble_messages(
            history=build_history(history_rows, max_chars=cfg.max_history_chars),
            passage_section=passage_section,
            citations=citations,
            system_prompt=self.system_prompt,
            memory_section=memory_section,
            rag_enabled=cfg.rag_enabled and self.retrieval is not None,
        )

        primary_model = request.model or cfg.answer_model
        base_request = LLMRequest(
            messages=messages,
            model=primary_model,
            max_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature_grounded if passage_section else cfg.temperature_ungrounded,
            extra=dict(request.extra),
        )

        collected: list[str] = []
        emitted = False
        error_message: str | None = None

        async def attempt(req: LLMRequest) -> AsyncIterator[str]:
            """Stream one model attempt, recording output and any error."""
            nonlocal emitted, error_message
            error_message = None
            try:
                async for event in self.llm.stream(req):
                    if isinstance(event, LLMError):
                        error_message = event.message
                        return
                    if event.text:
                        collected.append(event.text)
                        emitted = True
                        yield encode_chunk(event.text)
            except Exception as exc:  # noqa: BLE001 — a stream error must not kill the response
                error_message = str(exc)

        try:
            yield encode_event("trace", {"state": "generating", "model": primary_model})
            if citations:
                # Emitted before the answer so the client can resolve [n] markers
                # into links as text streams in, not after it finishes.
                yield encode_event("citations", {"citations": [c.as_dict() for c in citations]})

            async for block in attempt(base_request):
                yield block

            fallback = cfg.fallback_model
            if error_message and not emitted and fallback and fallback != primary_model:
                logger.warning(
                    "Primary model %s failed (%s) — falling back to %s",
                    primary_model,
                    error_message,
                    fallback,
                )
                notice = (
                    f"> ⚠️ The default model (`{primary_model}`) failed; "
                    f"answering with the fallback model (`{fallback}`).\n\n"
                )
                collected.append(notice)
                emitted = True
                yield encode_chunk(notice)
                fallback_request = LLMRequest(
                    messages=base_request.messages,
                    model=fallback,
                    max_tokens=base_request.max_tokens,
                    temperature=base_request.temperature,
                )
                async for block in attempt(fallback_request):
                    yield block

            if error_message and not emitted:
                logger.warning("Turn produced no output: %s", error_message)
                yield encode_error(error_message, code="answer_failed")
        finally:
            # Persistence must survive a consumer that stops reading, so it lives
            # in `finally`. Awaiting here is fine during close; yielding is not,
            # which is why DONE is emitted on the normal path below instead.
            await self.store.complete_message(
                message=assistant,
                content="".join(collected),
                # Errored only when nothing was produced. Text on screen means
                # the user got an answer, whatever happened afterwards.
                error=not emitted,
                citations=[c.as_dict() for c in citations] or None,
            )
            await self.store.commit()

        yield DONE


__all__ = ["LibrarianTurn", "TurnRequest", "assemble_messages", "build_history"]
