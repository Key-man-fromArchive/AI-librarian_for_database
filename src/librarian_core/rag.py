"""Chunk-level retrieval with a threshold gate, per-source cap, and abstention.

Why chunk level, and why a threshold:

Document-level rank-fusion scores (RRF and friends) are rank-based and
structurally flat — every result lands within a hair of every other, so there is
no value at which you can say "below this, stop citing". They also cannot tell
you *which part* of a long document is relevant, and a single document routinely
holds several unrelated topics. Cosine similarity over chunks gives an
interpretable, thresholdable number and points at the exact passage.

The pipeline, in order:

1. Retrieve ``candidate_chunks`` chunks at or above ``min_similarity``.
2. Cap per source document, so one sprawling document cannot fill every slot.
3. Take the top ``max_passages``.
4. Render as an untrusted, citation-tagged reference block under a fair
   character budget.
5. If nothing survives, **abstain** — return empty and let the caller tell the
   model to say so. Abstaining is a feature: a confident wrong citation costs
   far more than an admitted gap.

Retrieval failures also abstain rather than propagate. A degraded answer beats a
500, and the caller has no better recovery available.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from .config import DEFAULT_CONFIG, LibrarianConfig
from .ports import Citation, Passage, Principal, RetrievalPort
from .prompts import PASSAGE_SECTION_HEADER

logger = logging.getLogger(__name__)

#: Characters that separate meaningful tokens inside a container name.
_TOKEN_SPLIT = re.compile(r"[\s\-_/·,]+")


def detect_scope(query: str, scope_names: Sequence[str]) -> list[str]:
    """Container names mentioned in the query, for automatic narrowing.

    A question that names its container ("the Q3 assay notebook", "in the
    onboarding folder") is telling you where to look. Honouring that beats
    semantic search over the whole corpus, which tends to bury a dense,
    obviously-relevant container under superficially similar documents.

    Matching is substring-based on the whole name and on each token of at least
    two characters, which handles compound names like ``project-alpha`` being
    referenced as just ``alpha``.
    """
    q = (query or "").lower()
    if not q:
        return []
    hits: list[str] = []
    for name in scope_names:
        if not name:
            continue
        low = name.lower()
        if low in q:
            hits.append(name)
            continue
        for token in _TOKEN_SPLIT.split(name):
            if len(token) >= 2 and token.lower() in q:
                hits.append(name)
                break
    return hits


def select_passages(
    chunks: Sequence[Passage],
    *,
    per_source_cap: int,
    max_passages: int,
) -> list[Passage]:
    """Apply the per-source cap and take the top N, preserving score order."""
    selected: list[Passage] = []
    used: dict[str, int] = {}
    for chunk in chunks:
        if len(selected) >= max_passages:
            break
        seen = used.get(chunk.source_id, 0)
        if seen >= per_source_cap:
            continue
        used[chunk.source_id] = seen + 1
        selected.append(chunk)
    return selected


def render_passages(
    passages: Sequence[Passage],
    *,
    max_chars: int,
    min_passage_chars: int = 1_200,
    header: str = PASSAGE_SECTION_HEADER,
) -> tuple[str, list[Citation]]:
    """Render selected passages as a citation-tagged block.

    The budget is divided evenly rather than first-come-first-served: an early
    long passage would otherwise consume everything and starve the rest, which
    both hurts answers and makes evaluation misleading (a passage the model
    never saw still counts as "retrieved" in your metrics).

    Note on merging: concatenating adjacent chunks was tried and reverted. For
    table-heavy sources it scrambles column alignment further and measurably
    degraded answers. One best chunk per passage is intentional.
    """
    if not passages or max_chars <= len(header):
        return "", []

    entries: list[str] = []
    citations: list[Citation] = []
    budget = max_chars - len(header)
    per_passage = max(min_passage_chars, budget // max(len(passages), 1))

    for chunk in passages:
        index = len(citations) + 1
        title = (chunk.title or "").strip()[:200]
        body = (chunk.text or "").strip()[:per_passage]
        prefix = (
            f"\n[{index}] {title} "
            f"(source {chunk.source_id}, chunk {chunk.chunk_index}, sim {chunk.score:.2f})\n"
        )
        room = budget - len(prefix)
        if room <= 0:
            break
        entry = prefix + body[:room]
        entries.append(entry)
        budget -= len(entry)
        citations.append(
            Citation(
                index=index,
                source_id=chunk.source_id,
                title=title,
                chunk_index=chunk.chunk_index,
                score=round(float(chunk.score), 4),
            )
        )
        if budget <= 0:
            break

    if not citations:
        return "", []
    return header + "".join(entries), citations


async def retrieve_context(
    retrieval: RetrievalPort,
    query: str,
    *,
    principal: Principal,
    config: LibrarianConfig = DEFAULT_CONFIG,
    scope: Sequence[str] | None = None,
    scope_candidates: Sequence[str] | None = None,
) -> tuple[str, list[Citation]]:
    """Retrieve, gate, cap and render passages for ``query``.

    ``scope`` narrows retrieval explicitly. When omitted and ``scope_candidates``
    is supplied, the query is scanned for container names and auto-narrowed.

    Returns ``("", [])`` to signal abstention — no passage cleared the bar, or
    retrieval failed. Callers treat both identically.
    """
    stripped = (query or "").strip()
    if not stripped or not config.rag_enabled:
        return "", []
    if config.max_passages <= 0 or config.max_context_chars <= 0:
        return "", []

    effective_scope = [s for s in (scope or []) if s and s.strip()]
    if not effective_scope and scope_candidates:
        effective_scope = detect_scope(stripped, scope_candidates)
    if effective_scope:
        logger.info("Retrieval scoped to: %s", effective_scope)

    try:
        chunks = await retrieval.search(
            stripped,
            principal=principal,
            limit=config.candidate_chunks,
            min_score=config.min_similarity,
            scope=effective_scope or None,
        )
    except Exception:
        # Fail open: the turn proceeds as an ungrounded general answer.
        logger.warning("Retrieval failed; continuing without passages", exc_info=True)
        return "", []

    selected = select_passages(
        chunks,
        per_source_cap=config.per_source_cap,
        max_passages=config.max_passages,
    )
    if not selected:
        return "", []

    return render_passages(
        selected,
        max_chars=config.max_context_chars,
        min_passage_chars=config.min_passage_chars,
    )


__all__ = ["detect_scope", "render_passages", "retrieve_context", "select_passages"]
