"""The four contracts a host application must satisfy.

The librarian core knows nothing about your schema. It asks for passages, an
identity to scope them by, a model to stream from, and somewhere to persist the
conversation. Implement these four protocols against your own tables and the
whole system works — see ``librarian_adapters`` for ready-made implementations.

Nothing in this module imports a web framework, an ORM, or an LLM SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """Who is asking. Every retrieval and every stored row is scoped by this.

    ``tenant_id`` is the isolation boundary (organisation, workspace, team). Use
    a constant for single-tenant apps. ``scope_ids`` are the container ids the
    user may read (notebooks, folders, projects); an empty tuple means "no
    containers" and MUST be treated as deny-all, never as allow-all — see
    ``ACLPort`` for why that distinction matters.
    """

    user_id: int | str
    tenant_id: int | str = 0
    scope_ids: tuple[int | str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk of source material, ready to be cited.

    ``source_id`` identifies the document the chunk came from (used to cap how
    many passages a single document may contribute). ``chunk_index`` locates the
    chunk inside it, so the UI can deep-link back to the exact excerpt.
    """

    source_id: str
    title: str
    text: str
    score: float
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """A passage as referenced by ``[n]`` in the answer."""

    index: int
    source_id: str
    title: str
    chunk_index: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_id": self.source_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "score": self.score,
        }


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: Sequence[ChatMessage]
    model: str
    max_tokens: int | None = None
    temperature: float = 0.7
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMChunk:
    """A fragment of generated text."""

    text: str


@dataclass(frozen=True)
class LLMError:
    """A provider failure, surfaced verbatim rather than swallowed.

    The single most expensive bug this project has seen was an error event that
    a consumer silently dropped, leaving users with a generic "no content"
    message while the real cause (an unsupported parameter) went undiagnosed for
    days. Errors are a first-class event type here for that reason.
    """

    message: str
    code: str = "provider_error"


LLMEvent = LLMChunk | LLMError


@dataclass
class StoredMessage:
    id: int | str
    sequence: int
    role: str
    content: str
    status: str
    citations: list[dict[str, Any]] | None = None
    created_at: datetime | None = None


@dataclass
class StoredSession:
    id: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    archived_at: datetime | None = None


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalPort(Protocol):
    """Finds candidate passages for a question.

    Implementations must apply the caller's ACL themselves — the core will not
    filter results after the fact. Returning more than ``limit`` is allowed but
    pointless; returning unauthorised rows is a security bug.

    Results should be ordered by descending ``score`` and already filtered to
    ``score >= min_score``. Scores must be comparable across queries (cosine
    similarity works; raw rank-fusion scores do not — see docs/ARCHITECTURE.md).
    """

    async def search(
        self,
        query: str,
        *,
        principal: Principal,
        limit: int,
        min_score: float,
        scope: Sequence[str] | None = None,
    ) -> list[Passage]: ...


@runtime_checkable
class ACLPort(Protocol):
    """Resolves an application user into a scoped :class:`Principal`.

    Kept separate from retrieval so the "who can see what" decision lives in one
    auditable place. ``librarian_adapters.acl`` ships a single-user adapter and
    a tenant-scoped one; anything more specific belongs in your application.
    """

    async def resolve(self, user: Any) -> Principal: ...


@runtime_checkable
class LLMPort(Protocol):
    """Streams a completion.

    Must yield :class:`LLMError` rather than raising for provider-side failures,
    so the turn can decide whether to fall back to another model. Raising is
    reserved for programmer errors.
    """

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]: ...


@runtime_checkable
class SessionStorePort(Protocol):
    """Persists conversations.

    ``append_message`` must allocate ``sequence`` atomically — two concurrent
    turns in one session otherwise collide. The SQLAlchemy adapter does this
    with a row lock; an in-memory implementation can use a mutex.
    """

    async def create_session(self, *, principal: Principal, title: str = "") -> StoredSession: ...

    async def get_session(self, *, principal: Principal, session_id: str) -> StoredSession: ...

    async def list_sessions(self, *, principal: Principal, limit: int = 50) -> list[StoredSession]: ...

    async def list_messages(self, *, principal: Principal, session_id: str) -> list[StoredMessage]: ...

    async def append_message(
        self,
        *,
        principal: Principal,
        session_id: str,
        role: str,
        content: str = "",
        status: str = "complete",
    ) -> StoredMessage: ...

    async def complete_message(
        self,
        *,
        message: StoredMessage,
        content: str,
        error: bool = False,
        citations: list[dict[str, Any]] | None = None,
    ) -> None: ...

    async def set_title(self, *, principal: Principal, session_id: str, title: str) -> None: ...

    async def commit(self) -> None: ...


__all__ = [
    "ACLPort",
    "ChatMessage",
    "Citation",
    "LLMChunk",
    "LLMError",
    "LLMEvent",
    "LLMPort",
    "LLMRequest",
    "Passage",
    "Principal",
    "RetrievalPort",
    "SessionStorePort",
    "StoredMessage",
    "StoredSession",
]
