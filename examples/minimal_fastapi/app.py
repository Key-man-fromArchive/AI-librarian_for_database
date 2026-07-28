"""A complete librarian in one file — no database, no API key, no network.

Run it::

    pip install "ai-librarian-for-database[fastapi]" uvicorn
    uvicorn examples.minimal_fastapi.app:app --reload

Then::

    SID=$(curl -sX POST localhost:8000/api/librarian/sessions \\
            -H 'content-type: application/json' -d '{}' | python -c 'import json,sys;print(json.load(sys.stdin)["id"])')

    curl -N -X POST localhost:8000/api/librarian/sessions/$SID/turn \\
      -H 'content-type: application/json' \\
      -d '{"content": "What was the drop temperature on the Ethiopia roast?"}'

Note that the ``event: citations`` block arrives *before* any answer text — that
ordering is what lets a UI turn ``[1]`` into a link while tokens stream.

Ask something the corpus cannot answer at all and the librarian abstains::

    curl -N -X POST localhost:8000/api/librarian/sessions/$SID/turn \\
      -H 'content-type: application/json' \\
      -d '{"content": "What is our refund policy?"}'

A caveat worth understanding, because it is the whole argument for embeddings:
the retriever below scores **shared words**, so "the Kenya Nyeri roast" still
matches the Ethiopia document — they share *drop*, *temperature* and *roast*.
Lexical overlap cannot tell two entities apart. Cosine similarity over embedded
chunks can, which is why `PgVectorRetrieval` is what you use for real.

Everything stateful here is an in-memory stub. Swap the three adapters for the
real ones and the rest of the file is unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from fastapi import FastAPI

from librarian_adapters.fastapi_router import LibrarianDeps, build_librarian_router
from librarian_core import LibrarianConfig, LibrarianTurn
from librarian_core.ports import (
    LLMChunk,
    LLMEvent,
    LLMRequest,
    Passage,
    Principal,
    StoredMessage,
    StoredSession,
)

PRINCIPAL = Principal(user_id=1, tenant_id=1)

DOCUMENTS = [
    {
        "source_id": "doc-roast-001",
        "title": "Ethiopia Yirgacheffe — roast profile v3",
        "scope": "roast-profiles",
        "text": (
            "Charge temperature 196C. First crack at 8:42, bean temperature 201C. "
            "Development time ratio 21%. Total roast 11:05, drop temperature 210C."
        ),
    },
    {
        "source_id": "doc-cup-014",
        "title": "Cupping session 2031-03-04",
        "scope": "cupping",
        "text": (
            "Ethiopia Yirgacheffe v3 scored 86.5 overall; bergamot and stone fruit, "
            "bright acidity. Colombia Huila v3 scored 84.0; heavier body, cocoa finish."
        ),
    },
]


class DemoRetrieval:
    """Word-overlap retrieval. Real deployments use PgVectorRetrieval."""

    async def search(
        self,
        query: str,
        *,
        principal: Principal,
        limit: int,
        min_score: float,
        scope: Sequence[str] | None = None,
    ) -> list[Passage]:
        terms = {w.lower().strip("?.,") for w in query.split() if len(w) > 2}
        found: list[Passage] = []
        for doc in DOCUMENTS:
            if scope and doc["scope"] not in scope:
                continue
            words = {w.lower().strip(".,%") for w in f"{doc['title']} {doc['text']}".split()}
            score = len(terms & words) / max(len(terms), 1)
            if score >= min_score:
                found.append(
                    Passage(
                        source_id=doc["source_id"],
                        title=doc["title"],
                        text=doc["text"],
                        score=round(score, 3),
                    )
                )
        found.sort(key=lambda p: p.score, reverse=True)
        return found[:limit]


class EchoLLM:
    """Echoes the passages it was given, so grounding is visible without a key."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        passages = [m.content for m in request.messages if m.content.startswith("## Retrieved")]
        if not passages:
            yield LLMChunk("I have no relevant passages for that, so I will not guess.")
            return
        yield LLMChunk("Based on the retrieved passages [1]:\n\n")
        for line in passages[0].splitlines():
            if line.startswith("["):
                yield LLMChunk(f"- {line.strip()}\n")


class MemoryStore:
    def __init__(self) -> None:
        self.sessions: dict[str, StoredSession] = {}
        self.messages: dict[str, list[StoredMessage]] = {}
        self._next = 1

    async def create_session(self, *, principal: Principal, title: str = "") -> StoredSession:
        session = StoredSession(id=f"s{len(self.sessions) + 1}", title=title)
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
            id=self._next,
            sequence=len(self.messages[session_id]) + 1,
            role=role,
            content=content,
            status=status,
        )
        self._next += 1
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
        message.content, message.citations = content, citations
        message.status = "error" if error else "complete"

    async def set_title(self, *, principal: Principal, session_id: str, title: str) -> None:
        self.sessions[session_id].title = title

    async def delete_session(self, *, principal: Principal, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def commit(self) -> None:
        """No-op: nothing to commit in memory."""


STORE = MemoryStore()
RETRIEVAL = DemoRetrieval()
# 0.2 suits this toy word-overlap scorer; cosine retrieval wants ~0.5.
CONFIG = LibrarianConfig(answer_model="demo", min_similarity=0.2, max_passages=3)


# These must be `async def`, not sync lambdas returning a coroutine: FastAPI
# awaits a dependency only when it is a coroutine function, so a lambda would
# inject the un-awaited coroutine object and every handler would fail on first
# attribute access.
async def get_principal() -> Principal:
    return PRINCIPAL


async def get_store() -> MemoryStore:
    return STORE


async def get_turn() -> LibrarianTurn:
    return LibrarianTurn(store=STORE, llm=EchoLLM(), retrieval=RETRIEVAL, config=CONFIG)


async def get_scope_candidates(principal: Principal) -> list[str]:
    return sorted({d["scope"] for d in DOCUMENTS})


app = FastAPI(title="AI Librarian — minimal example")
app.include_router(
    build_librarian_router(
        dependencies=LibrarianDeps(
            get_principal=get_principal,
            get_turn=get_turn,
            get_store=get_store,
            get_scope_candidates=get_scope_candidates,
        )
    ),
    prefix="/api",
)
