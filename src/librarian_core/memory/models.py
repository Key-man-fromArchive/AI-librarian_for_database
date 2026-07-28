"""Value objects and the state machine for governed memory.

"Governed" means no memory reaches a prompt without a human approving it. An
assistant that silently accumulates beliefs about your data is a liability: a
single bad inference becomes permanent context, silently poisoning every later
answer, with no record of where it came from.

Lifecycle::

    proposed ──approve──> active ──supersede──> superseded
       │                    │
       ├──reject──> rejected└──expire──> expired
                             └──tombstone──> tombstoned

``tombstoned`` is terminal and irreversible: it is the deletion path for content
that should never have been stored. Everything else keeps its revision history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MemoryStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    TOMBSTONED = "tombstoned"


#: Only these may be injected into a prompt.
INJECTABLE_STATUSES = frozenset({MemoryStatus.ACTIVE})

#: Allowed transitions. Anything absent raises :class:`LifecycleError`.
TRANSITIONS: dict[MemoryStatus, frozenset[MemoryStatus]] = {
    MemoryStatus.PROPOSED: frozenset({MemoryStatus.ACTIVE, MemoryStatus.REJECTED, MemoryStatus.TOMBSTONED}),
    MemoryStatus.ACTIVE: frozenset({MemoryStatus.SUPERSEDED, MemoryStatus.EXPIRED, MemoryStatus.TOMBSTONED}),
    MemoryStatus.REJECTED: frozenset({MemoryStatus.TOMBSTONED}),
    MemoryStatus.SUPERSEDED: frozenset({MemoryStatus.TOMBSTONED}),
    MemoryStatus.EXPIRED: frozenset({MemoryStatus.ACTIVE, MemoryStatus.TOMBSTONED}),
    MemoryStatus.TOMBSTONED: frozenset(),
}


class MemoryError_(Exception):
    """Base class for governed-memory failures."""


class ProvenanceError(MemoryError_):
    """Raised when a memory lacks the evidence its type requires."""


class LifecycleError(MemoryError_):
    """Raised on a transition the state machine does not allow."""


@dataclass(frozen=True)
class MemorySource:
    """Where a memory came from. Without this a memory is hearsay."""

    #: Identifier of the document/message the claim was drawn from.
    source_id: str
    #: Verbatim excerpt supporting the claim. Paraphrase defeats the purpose:
    #: a reviewer must be able to check the claim against the original words.
    excerpt: str
    kind: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    id: int | str
    tenant_id: int | str
    memory_type: str
    #: Stable key for the thing being remembered ("user.timezone",
    #: "protocol.buffer-recipe"). Supersession matches on this, so a new fact
    #: about the same subject retires the old one instead of coexisting.
    subject_key: str
    content: str
    status: MemoryStatus = MemoryStatus.PROPOSED
    sources: list[MemorySource] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    approved_at: datetime | None = None
    approved_by: int | str | None = None
    revision: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_injectable(self) -> bool:
        return self.status in INJECTABLE_STATUSES


@dataclass(frozen=True)
class MemoryRevision:
    """One entry in a memory's audit trail."""

    memory_id: int | str
    revision: int
    status: MemoryStatus
    content: str
    actor_id: int | str | None
    at: datetime
    reason: str = ""


__all__ = [
    "INJECTABLE_STATUSES",
    "TRANSITIONS",
    "LifecycleError",
    "Memory",
    "MemoryError_",
    "MemoryRevision",
    "MemorySource",
    "MemoryStatus",
    "ProvenanceError",
]
