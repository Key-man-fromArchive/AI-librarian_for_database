"""Session persistence on SQLAlchemy 2.0 async.

Two tables, declared on their own ``Base`` so they can be created alongside your
existing models without colliding. Prefix them if ``librarian_sessions`` is
taken — the names are only referenced here.

The one subtle part is sequence allocation. ``append_message`` takes a row lock
on the session before reading ``next_sequence``, because two concurrent turns in
one session would otherwise read the same value and write duplicate sequences,
which silently reorders the conversation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from librarian_core.ports import Principal, StoredMessage, StoredSession


class LibrarianBase(DeclarativeBase):
    """Separate declarative base so these tables can join any metadata."""


def _now() -> datetime:
    return datetime.now(UTC)


class LibrarianSession(LibrarianBase):
    __tablename__ = "librarian_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(500), default="")
    #: Monotonic counter for message ordering; allocated under a row lock.
    next_sequence: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class LibrarianMessage(LibrarianBase):
    __tablename__ = "librarian_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("librarian_sessions.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    #: ``streaming`` while tokens arrive, then ``complete`` or ``error``.
    #: Only ``complete`` rows are replayed as conversation history.
    status: Mapped[str] = mapped_column(String(16), default="complete")
    citations: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SessionNotFound(Exception):
    """Raised for a missing session *and* for another user's session.

    Deliberately indistinguishable: a distinct "forbidden" response would
    confirm the id exists, letting anyone enumerate sessions.
    """


class SQLAlchemySessionStore:
    """A :class:`SessionStorePort` bound to one ``AsyncSession``.

    Construct per request. ``commit()`` is explicit rather than left to
    dependency teardown, because a client that re-reads immediately after the
    response would otherwise race an uncommitted transaction.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # -- internal ----------------------------------------------------------

    async def _row(self, *, principal: Principal, session_id: str, for_update: bool = False):
        stmt = select(LibrarianSession).where(
            LibrarianSession.id == session_id,
            LibrarianSession.tenant_id == str(principal.tenant_id),
            LibrarianSession.user_id == str(principal.user_id),
            LibrarianSession.deleted_at.is_(None),
        )
        if for_update:
            stmt = stmt.with_for_update()
        row = (await self.db.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise SessionNotFound(session_id)
        return row

    @staticmethod
    def _to_session(row: LibrarianSession) -> StoredSession:
        return StoredSession(
            id=row.id,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
            archived_at=row.archived_at,
        )

    @staticmethod
    def _to_message(row: LibrarianMessage) -> StoredMessage:
        return StoredMessage(
            id=row.id,
            sequence=row.sequence,
            role=row.role,
            content=row.content,
            status=row.status,
            citations=row.citations,
            created_at=row.created_at,
        )

    # -- port --------------------------------------------------------------

    async def create_session(self, *, principal: Principal, title: str = "") -> StoredSession:
        row = LibrarianSession(
            id=str(uuid4()),
            tenant_id=str(principal.tenant_id),
            user_id=str(principal.user_id),
            title=title.strip()[:500],
        )
        self.db.add(row)
        await self.db.flush()
        return self._to_session(row)

    async def get_session(self, *, principal: Principal, session_id: str) -> StoredSession:
        return self._to_session(await self._row(principal=principal, session_id=session_id))

    async def list_sessions(self, *, principal: Principal, limit: int = 50) -> list[StoredSession]:
        result = await self.db.execute(
            select(LibrarianSession)
            .where(
                LibrarianSession.tenant_id == str(principal.tenant_id),
                LibrarianSession.user_id == str(principal.user_id),
                LibrarianSession.deleted_at.is_(None),
            )
            .order_by(LibrarianSession.updated_at.desc(), LibrarianSession.id.desc())
            .limit(limit)
        )
        return [self._to_session(row) for row in result.scalars()]

    async def list_messages(self, *, principal: Principal, session_id: str) -> list[StoredMessage]:
        await self._row(principal=principal, session_id=session_id)
        result = await self.db.execute(
            select(LibrarianMessage)
            .where(
                LibrarianMessage.session_id == session_id,
                LibrarianMessage.tenant_id == str(principal.tenant_id),
            )
            .order_by(LibrarianMessage.sequence.asc())
        )
        return [self._to_message(row) for row in result.scalars()]

    async def append_message(
        self,
        *,
        principal: Principal,
        session_id: str,
        role: str,
        content: str = "",
        status: str = "complete",
    ) -> StoredMessage:
        session = await self._row(principal=principal, session_id=session_id, for_update=True)
        if session.archived_at is not None:
            raise ValueError("cannot append to an archived session")
        sequence = session.next_sequence
        session.next_sequence += 1
        session.updated_at = _now()
        row = LibrarianMessage(
            session_id=session.id,
            tenant_id=str(principal.tenant_id),
            sequence=sequence,
            role=role,
            content=content,
            status=status,
        )
        self.db.add(row)
        await self.db.flush()
        return self._to_message(row)

    async def complete_message(
        self,
        *,
        message: StoredMessage,
        content: str,
        error: bool = False,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        row = await self.db.get(LibrarianMessage, message.id)
        if row is None:
            return
        row.content = content
        row.status = "error" if error else "complete"
        if citations is not None:
            row.citations = citations
        message.content = content
        message.status = row.status
        await self.db.flush()

    async def set_title(self, *, principal: Principal, session_id: str, title: str) -> None:
        row = await self._row(principal=principal, session_id=session_id)
        row.title = title.strip()[:500]
        row.updated_at = _now()
        await self.db.flush()

    async def archive_session(self, *, principal: Principal, session_id: str) -> None:
        row = await self._row(principal=principal, session_id=session_id)
        row.archived_at = _now()
        await self.db.flush()

    async def delete_session(self, *, principal: Principal, session_id: str) -> None:
        """Soft delete — the thread disappears but the audit trail survives."""
        row = await self._row(principal=principal, session_id=session_id)
        row.deleted_at = _now()
        await self.db.flush()

    async def commit(self) -> None:
        await self.db.commit()


__all__ = [
    "LibrarianBase",
    "LibrarianMessage",
    "LibrarianSession",
    "SQLAlchemySessionStore",
    "SessionNotFound",
]
