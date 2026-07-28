"""Behavioural tests for the turn — the failure modes that motivated its design.

Each test here corresponds to a bug that reached real users in the origin
system. They are the regression suite for the whole point of the project.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from librarian_core.config import LibrarianConfig
from librarian_core.ports import (
    LLMChunk,
    LLMError,
    LLMEvent,
    LLMRequest,
    Passage,
    Principal,
    StoredMessage,
    StoredSession,
)
from librarian_core.turn import LibrarianTurn, TurnRequest, build_history

PRINCIPAL = Principal(user_id=1, tenant_id=1)


class MemoryStore:
    """Minimal in-memory SessionStorePort."""

    def __init__(self) -> None:
        self.sessions: dict[str, StoredSession] = {}
        self.messages: dict[str, list[StoredMessage]] = {}
        self.commits = 0
        self._next_id = 1

    async def create_session(self, *, principal: Principal, title: str = "") -> StoredSession:
        session = StoredSession(id="s1", title=title)
        self.sessions[session.id] = session
        self.messages[session.id] = []
        return session

    async def get_session(self, *, principal: Principal, session_id: str) -> StoredSession:
        return self.sessions[session_id]

    async def list_sessions(self, *, principal: Principal, limit: int = 50) -> list[StoredSession]:
        return list(self.sessions.values())

    async def list_messages(self, *, principal: Principal, session_id: str) -> list[StoredMessage]:
        return list(self.messages[session_id])

    async def append_message(
        self,
        *,
        principal: Principal,
        session_id: str,
        role: str,
        content: str = "",
        status: str = "complete",
    ) -> StoredMessage:
        message = StoredMessage(
            id=self._next_id,
            sequence=len(self.messages[session_id]) + 1,
            role=role,
            content=content,
            status=status,
        )
        self._next_id += 1
        self.messages[session_id].append(message)
        return message

    async def complete_message(
        self,
        *,
        message: StoredMessage,
        content: str,
        error: bool = False,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        message.content = content
        message.status = "error" if error else "complete"
        message.citations = citations

    async def set_title(self, *, principal: Principal, session_id: str, title: str) -> None:
        self.sessions[session_id].title = title

    async def commit(self) -> None:
        self.commits += 1


class ScriptedLLM:
    """Replays a fixed event script per call."""

    def __init__(self, *scripts: Sequence[LLMEvent]) -> None:
        self.scripts = [list(s) for s in scripts]
        self.calls: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        self.calls.append(request)
        script = self.scripts[min(len(self.calls) - 1, len(self.scripts) - 1)]
        for event in script:
            yield event


class StubRetrieval:
    def __init__(self, passages: Sequence[Passage] = ()) -> None:
        self.passages = list(passages)

    async def search(self, query, *, principal, limit, min_score, scope=None) -> list[Passage]:
        return [p for p in self.passages if p.score >= min_score][:limit]


class ExplodingRetrieval:
    async def search(self, *args, **kwargs):
        raise RuntimeError("vector store unreachable")


def blocks(chunks: list[str]) -> str:
    return "".join(chunks)


def texts(chunks: list[str]) -> str:
    """Concatenate the answer text carried by data: chunks."""
    out = []
    for block in chunks:
        for line in block.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and "chunk" in payload:
                    out.append(payload["chunk"])
    return "".join(out)


async def run_turn(turn: LibrarianTurn, store: MemoryStore, content: str = "question?") -> list[str]:
    await store.create_session(principal=PRINCIPAL)
    return [
        block async for block in turn.run(TurnRequest(session_id="s1", content=content), principal=PRINCIPAL)
    ]


# ---------------------------------------------------------------------------


async def test_answer_streams_and_persists_as_complete():
    store = MemoryStore()
    turn = LibrarianTurn(store=store, llm=ScriptedLLM([LLMChunk("Hello "), LLMChunk("world")]))

    out = await run_turn(turn, store)

    assert texts(out) == "Hello world"
    assert out[-1] == "data: [DONE]\n\n"
    assistant = store.messages["s1"][-1]
    assert assistant.status == "complete"
    assert assistant.content == "Hello world"


async def test_error_after_output_does_not_mark_the_answer_failed():
    """The bug this project exists to prevent.

    A provider hiccup after tokens have streamed left users looking at a
    complete, correct answer stamped 'AI response failed'.
    """
    store = MemoryStore()
    llm = ScriptedLLM([LLMChunk("A useful answer."), LLMError("connection reset")])
    turn = LibrarianTurn(store=store, llm=llm)

    out = await run_turn(turn, store)

    assert "A useful answer." in texts(out)
    assert "event: error" not in blocks(out)
    assert store.messages["s1"][-1].status == "complete"


async def test_failure_before_any_output_surfaces_the_real_reason():
    store = MemoryStore()
    llm = ScriptedLLM([LLMError("temperature is not supported for this model")])
    turn = LibrarianTurn(store=store, llm=llm)

    out = await run_turn(turn, store)

    joined = blocks(out)
    assert "event: error" in joined
    # The provider's own words, not a generic message.
    assert "temperature is not supported" in joined
    assert store.messages["s1"][-1].status == "error"


async def test_fallback_runs_only_when_nothing_was_emitted():
    store = MemoryStore()
    llm = ScriptedLLM([LLMError("provider down")], [LLMChunk("fallback answer")])
    turn = LibrarianTurn(
        store=store,
        llm=llm,
        config=LibrarianConfig(answer_model="primary", fallback_model="secondary"),
    )

    out = await run_turn(turn, store)

    assert [c.model for c in llm.calls] == ["primary", "secondary"]
    assert "fallback answer" in texts(out)
    assert store.messages["s1"][-1].status == "complete"


async def test_no_fallback_once_text_has_streamed():
    """Switching models mid-answer would splice two different answers together."""
    store = MemoryStore()
    llm = ScriptedLLM([LLMChunk("partial "), LLMError("died midway")], [LLMChunk("SHOULD NOT APPEAR")])
    turn = LibrarianTurn(
        store=store,
        llm=llm,
        config=LibrarianConfig(answer_model="primary", fallback_model="secondary"),
    )

    out = await run_turn(turn, store)

    assert len(llm.calls) == 1
    assert "SHOULD NOT APPEAR" not in texts(out)
    assert store.messages["s1"][-1].content == "partial "


async def test_citations_are_emitted_before_the_answer():
    store = MemoryStore()
    retrieval = StubRetrieval([Passage(source_id="d1", title="Doc", text="body text", score=0.9)])
    turn = LibrarianTurn(store=store, llm=ScriptedLLM([LLMChunk("grounded")]), retrieval=retrieval)

    out = await run_turn(turn, store)

    citation_at = next(i for i, b in enumerate(out) if b.startswith("event: citations"))
    first_chunk_at = next(i for i, b in enumerate(out) if b.startswith("data: {"))
    assert citation_at < first_chunk_at
    assert store.messages["s1"][-1].citations[0]["source_id"] == "d1"


async def test_retrieval_failure_degrades_instead_of_erroring():
    store = MemoryStore()
    turn = LibrarianTurn(
        store=store, llm=ScriptedLLM([LLMChunk("general answer")]), retrieval=ExplodingRetrieval()
    )

    out = await run_turn(turn, store)

    assert "general answer" in texts(out)
    assert "event: error" not in blocks(out)


async def test_title_is_generated_from_the_first_question_only():
    store = MemoryStore()
    turn = LibrarianTurn(store=store, llm=ScriptedLLM([LLMChunk("ok")]))

    await store.create_session(principal=PRINCIPAL)
    async for _ in turn.run(
        TurnRequest(session_id="s1", content="What is the drop temperature?"), principal=PRINCIPAL
    ):
        pass
    assert store.sessions["s1"].title == "What is the drop temperature?"

    async for _ in turn.run(
        TurnRequest(session_id="s1", content="And the charge temperature?"), principal=PRINCIPAL
    ):
        pass
    assert store.sessions["s1"].title == "What is the drop temperature?"


async def test_user_chosen_title_is_never_overwritten():
    store = MemoryStore()
    turn = LibrarianTurn(store=store, llm=ScriptedLLM([LLMChunk("ok")]))
    await store.create_session(principal=PRINCIPAL, title="My roast log")

    async for _ in turn.run(TurnRequest(session_id="s1", content="anything"), principal=PRINCIPAL):
        pass

    assert store.sessions["s1"].title == "My roast log"


async def test_message_is_finalised_even_if_the_consumer_stops_reading():
    """A disconnecting client must not leave a row stuck in 'streaming'."""
    store = MemoryStore()
    turn = LibrarianTurn(store=store, llm=ScriptedLLM([LLMChunk("one"), LLMChunk("two")]))
    await store.create_session(principal=PRINCIPAL)

    generator = turn.run(TurnRequest(session_id="s1", content="q"), principal=PRINCIPAL)
    await generator.__anext__()
    await generator.aclose()

    assert store.messages["s1"][-1].status != "streaming"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["complete", "complete"], 2),
        (["complete", "streaming"], 1),
        (["error", "complete"], 1),
    ],
)
def test_history_replays_only_complete_messages(statuses, expected):
    messages = [
        StoredMessage(id=i, sequence=i, role="user", content="text", status=status)
        for i, status in enumerate(statuses, start=1)
    ]
    assert len(build_history(messages, max_chars=10_000)) == expected


def test_history_keeps_the_newest_messages_when_truncating():
    messages = [
        StoredMessage(id=1, sequence=1, role="user", content="old" * 100, status="complete"),
        StoredMessage(id=2, sequence=2, role="assistant", content="new", status="complete"),
    ]
    history = build_history(messages, max_chars=50)
    assert [m.content for m in history] == ["new"]
