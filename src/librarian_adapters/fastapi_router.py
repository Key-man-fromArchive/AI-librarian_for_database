"""A drop-in FastAPI router: sessions plus a streaming turn endpoint.

Mount it and you have a working librarian API::

    app.include_router(
        build_librarian_router(
            dependencies=LibrarianDeps(
                get_principal=my_principal_dep,
                get_turn=my_turn_dep,
                get_store=my_store_dep,
            )
        ),
        prefix="/api",
    )

Endpoints:

===========================================  ====================================
``POST   /librarian/sessions``               create a conversation
``GET    /librarian/sessions``               list conversations
``GET    /librarian/sessions/{id}``          full thread with citations
``DELETE /librarian/sessions/{id}``          soft-delete
``POST   /librarian/sessions/{id}/turn``     ask; streams SSE
===========================================  ====================================

Write your own if your app has different conventions — the router is thin, and
everything it does is public API on the core.
"""

# NOTE: deliberately no `from __future__ import annotations` here.
# The route handlers below are annotated with Annotated[...] aliases built
# *inside* build_librarian_router. Postponed evaluation would turn those into
# strings that FastAPI resolves against module globals, where the local aliases
# do not exist — every dependency would silently degrade into a query parameter
# and the endpoints would 422. Eager evaluation keeps them real objects.

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from librarian_core.ports import Principal, SessionStorePort
from librarian_core.sse import SSE_HEADERS
from librarian_core.turn import LibrarianTurn, TurnRequest

from .sqlalchemy_store import SessionNotFound


class SessionCreate(BaseModel):
    title: str = Field(default="", max_length=500)


class TurnCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    model: str | None = Field(default=None, max_length=200)
    scope: list[str] | None = Field(
        default=None,
        max_length=20,
        description="Container names to restrict retrieval to, for precision.",
    )


class MessageOut(BaseModel):
    id: int | str
    sequence: int
    role: str
    content: str
    status: str
    citations: list[dict] | None = None
    created_at: Any = None


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: Any = None
    updated_at: Any = None
    archived_at: Any = None


class SessionDetailOut(SessionOut):
    messages: list[MessageOut]


@dataclass
class LibrarianDeps:
    """Application-supplied dependency callables.

    Every callable here must be declared ``async def``. FastAPI awaits a
    dependency only when it is a coroutine *function*; a sync lambda that
    returns a coroutine gets injected un-awaited, and the handler then fails
    with ``'coroutine' object has no attribute ...``.

    ``get_principal``, ``get_turn`` and ``get_store`` may themselves use
    ``Depends`` in their signatures — they are resolved by FastAPI normally, so
    request-scoped database sessions work as usual.

    ``get_scope_candidates`` is optional; supply it to enable automatic
    narrowing when a question names one of the user's containers.
    """

    get_principal: Callable[..., Awaitable[Principal]]
    get_turn: Callable[..., Awaitable[LibrarianTurn]]
    get_store: Callable[..., Awaitable[SessionStorePort]]
    get_scope_candidates: Callable[[Principal], Awaitable[Sequence[str]]] | None = None


def build_librarian_router(*, dependencies: LibrarianDeps, prefix: str = "/librarian") -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["librarian"])
    deps = dependencies

    PrincipalDep = Annotated[Principal, Depends(deps.get_principal)]
    StoreDep = Annotated[SessionStorePort, Depends(deps.get_store)]
    TurnDep = Annotated[LibrarianTurn, Depends(deps.get_turn)]

    def _not_found() -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    @router.post(
        "/sessions",
        response_model=SessionOut,
        status_code=status.HTTP_201_CREATED,
        summary="Create a conversation",
    )
    async def create_session(body: SessionCreate, principal: PrincipalDep, store: StoreDep) -> SessionOut:
        session = await store.create_session(principal=principal, title=body.title)
        # Commit before responding: the client immediately POSTs a turn against
        # this id, and an uncommitted row would 404 that request.
        await store.commit()
        return SessionOut(**session.__dict__)

    @router.get("/sessions", response_model=list[SessionOut], summary="List conversations")
    async def list_sessions(principal: PrincipalDep, store: StoreDep) -> list[SessionOut]:
        return [SessionOut(**s.__dict__) for s in await store.list_sessions(principal=principal)]

    @router.get(
        "/sessions/{session_id}",
        response_model=SessionDetailOut,
        summary="Read a conversation with its messages",
    )
    async def get_session(session_id: str, principal: PrincipalDep, store: StoreDep) -> SessionDetailOut:
        try:
            session = await store.get_session(principal=principal, session_id=session_id)
            messages = await store.list_messages(principal=principal, session_id=session_id)
        except SessionNotFound as exc:
            raise _not_found() from exc
        return SessionDetailOut(
            **session.__dict__,
            messages=[MessageOut(**m.__dict__) for m in messages],
        )

    @router.delete(
        "/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Soft-delete a conversation",
    )
    async def delete_session(session_id: str, principal: PrincipalDep, store: StoreDep) -> None:
        deleter = getattr(store, "delete_session", None)
        if deleter is None:
            raise HTTPException(status_code=501, detail="store does not support deletion")
        try:
            await deleter(principal=principal, session_id=session_id)
        except SessionNotFound as exc:
            raise _not_found() from exc
        await store.commit()

    @router.post(
        "/sessions/{session_id}/turn",
        summary="Ask a question; streams the answer as SSE",
        response_class=StreamingResponse,
    )
    async def create_turn(
        session_id: str,
        body: TurnCreate,
        principal: PrincipalDep,
        store: StoreDep,
        turn: TurnDep,
    ) -> StreamingResponse:
        try:
            await store.get_session(principal=principal, session_id=session_id)
        except SessionNotFound as exc:
            raise _not_found() from exc

        candidates: Sequence[str] | None = None
        if deps.get_scope_candidates is not None:
            candidates = await deps.get_scope_candidates(principal)

        stream = turn.run(
            TurnRequest(
                session_id=session_id,
                content=body.content,
                model=body.model,
                scope=body.scope,
            ),
            principal=principal,
            scope_candidates=candidates,
        )
        return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)

    return router


__all__ = [
    "LibrarianDeps",
    "MessageOut",
    "SessionCreate",
    "SessionDetailOut",
    "SessionOut",
    "TurnCreate",
    "build_librarian_router",
]
