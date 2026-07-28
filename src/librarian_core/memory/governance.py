"""Lifecycle rules for governed memory — storage-agnostic.

These functions validate and mutate :class:`Memory` objects in memory; they never
touch a database. Your store adapter persists the result and appends the
returned revision. Keeping the rules here means they are unit-testable without a
database and identical across every backend.

Two invariants are enforced and should not be relaxed:

* **No approval without provenance.** A memory whose type requires evidence
  cannot become ``active`` without at least one source carrying a real excerpt.
* **Approval is not self-service by default.** ``require_distinct_approver``
  stops the proposer rubber-stamping their own inference. Turn it off only for
  single-user deployments where there is nobody else to ask.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from .models import (
    TRANSITIONS,
    LifecycleError,
    Memory,
    MemoryRevision,
    MemorySource,
    MemoryStatus,
    ProvenanceError,
)

#: Types exempt from the provenance requirement — facts the user stated outright
#: rather than the assistant inferring them from documents.
DEFAULT_PROVENANCE_EXEMPT: frozenset[str] = frozenset({"user_preference", "user_directive"})

#: An excerpt shorter than this is not evidence a reviewer can check.
MIN_EXCERPT_CHARS = 12


def _now() -> datetime:
    return datetime.now(UTC)


def check_provenance(
    memory: Memory,
    *,
    exempt_types: frozenset[str] = DEFAULT_PROVENANCE_EXEMPT,
) -> None:
    """Raise :class:`ProvenanceError` unless the memory can be traced.

    Exempt types still pass — a stated preference has no document to cite.
    """
    if memory.memory_type in exempt_types:
        return
    usable = [s for s in memory.sources if s.source_id and len(s.excerpt.strip()) >= MIN_EXCERPT_CHARS]
    if not usable:
        raise ProvenanceError(
            f"memory {memory.id!r} of type {memory.memory_type!r} needs at least one source "
            f"with an excerpt of {MIN_EXCERPT_CHARS}+ characters before it can be approved"
        )


def _transition(
    memory: Memory,
    target: MemoryStatus,
    *,
    actor_id: int | str | None,
    reason: str,
) -> MemoryRevision:
    allowed = TRANSITIONS.get(memory.status, frozenset())
    if target not in allowed:
        raise LifecycleError(
            f"cannot move memory {memory.id!r} from {memory.status} to {target}; "
            f"allowed: {sorted(allowed) or 'none (terminal)'}"
        )
    memory.status = target
    memory.revision += 1
    memory.updated_at = _now()
    return MemoryRevision(
        memory_id=memory.id,
        revision=memory.revision,
        status=target,
        content=memory.content,
        actor_id=actor_id,
        at=memory.updated_at,
        reason=reason,
    )


def propose(
    *,
    memory_id: int | str,
    tenant_id: int | str,
    memory_type: str,
    subject_key: str,
    content: str,
    sources: Sequence[MemorySource] = (),
    metadata: dict | None = None,
) -> Memory:
    """Create a memory in ``proposed`` state. Never injected until approved."""
    text = content.strip()
    if not text:
        raise ValueError("memory content must not be empty")
    now = _now()
    return Memory(
        id=memory_id,
        tenant_id=tenant_id,
        memory_type=memory_type,
        subject_key=subject_key.strip(),
        content=text,
        status=MemoryStatus.PROPOSED,
        sources=list(sources),
        created_at=now,
        updated_at=now,
        metadata=dict(metadata or {}),
    )


def approve(
    memory: Memory,
    *,
    actor_id: int | str,
    proposer_id: int | str | None = None,
    require_distinct_approver: bool = True,
    exempt_types: frozenset[str] = DEFAULT_PROVENANCE_EXEMPT,
    reason: str = "",
) -> MemoryRevision:
    """Move a proposal to ``active``, making it eligible for injection."""
    if require_distinct_approver and proposer_id is not None and proposer_id == actor_id:
        raise LifecycleError(
            "the proposer cannot approve their own memory; set "
            "require_distinct_approver=False for single-user deployments"
        )
    check_provenance(memory, exempt_types=exempt_types)
    revision = _transition(memory, MemoryStatus.ACTIVE, actor_id=actor_id, reason=reason)
    memory.approved_at = revision.at
    memory.approved_by = actor_id
    return revision


def reject(memory: Memory, *, actor_id: int | str, reason: str = "") -> MemoryRevision:
    return _transition(memory, MemoryStatus.REJECTED, actor_id=actor_id, reason=reason)


def supersede(
    old: Memory,
    new: Memory,
    *,
    actor_id: int | str,
    reason: str = "",
) -> MemoryRevision:
    """Retire ``old`` in favour of ``new`` covering the same subject.

    Same-subject facts must not coexist: two contradictory "active" memories put
    the contradiction into the prompt and let the model pick, unpredictably.
    """
    if old.subject_key != new.subject_key:
        raise LifecycleError(
            f"supersede requires the same subject_key ({old.subject_key!r} != {new.subject_key!r})"
        )
    if old.tenant_id != new.tenant_id:
        raise LifecycleError("cannot supersede across tenants")
    return _transition(
        old,
        MemoryStatus.SUPERSEDED,
        actor_id=actor_id,
        reason=reason or f"superseded by {new.id!r}",
    )


def expire(memory: Memory, *, actor_id: int | str | None = None, reason: str = "") -> MemoryRevision:
    """Age a memory out of injection while keeping it recoverable."""
    return _transition(memory, MemoryStatus.EXPIRED, actor_id=actor_id, reason=reason or "expired")


def tombstone(memory: Memory, *, actor_id: int | str, reason: str = "") -> MemoryRevision:
    """Terminal removal. Content should be redacted by the store afterwards."""
    return _transition(memory, MemoryStatus.TOMBSTONED, actor_id=actor_id, reason=reason)


def find_expired(
    memories: Sequence[Memory],
    *,
    max_age_days: int,
    now: datetime | None = None,
) -> list[Memory]:
    """Active memories past their age limit.

    Age is measured from approval, not creation: a fact re-approved last week is
    fresh regardless of when it was first proposed.
    """
    if max_age_days <= 0:
        return []
    reference = now or _now()
    stale: list[Memory] = []
    for memory in memories:
        if memory.status != MemoryStatus.ACTIVE:
            continue
        approved = memory.approved_at
        if approved is None:
            continue
        if (reference - approved).days > max_age_days:
            stale.append(memory)
    return stale


__all__ = [
    "DEFAULT_PROVENANCE_EXEMPT",
    "MIN_EXCERPT_CHARS",
    "approve",
    "check_provenance",
    "expire",
    "find_expired",
    "propose",
    "reject",
    "supersede",
    "tombstone",
]
