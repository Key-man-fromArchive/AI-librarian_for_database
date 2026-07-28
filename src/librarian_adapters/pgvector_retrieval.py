"""Chunk retrieval over a pgvector column, described by configuration.

You almost certainly already have a chunks-with-embeddings table. Rather than
imposing a schema, this adapter takes a :class:`ChunkTableSpec` naming your
columns and builds the query against them.

The ACL is applied *inside* the SQL, not after. Filtering in Python means the
database has already returned rows the user may not see — one logging statement
or error message away from a leak. It also fails closed: a principal with an
empty ``scope_ids`` matches nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from librarian_core.ports import Passage, Principal

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Awaitable[Sequence[float]]]


@dataclass(frozen=True)
class ChunkTableSpec:
    """Where your chunks live and what the columns are called."""

    table: str
    #: pgvector column holding the chunk embedding.
    embedding_column: str = "embedding"
    text_column: str = "chunk_text"
    #: Identifies the parent document — used for the per-source cap.
    source_id_column: str = "document_id"
    title_column: str = "title"
    chunk_index_column: str = "chunk_index"
    #: Tenant isolation column. Set to ``None`` for single-tenant schemas.
    tenant_column: str | None = "org_id"
    #: Container column matched against ``Principal.scope_ids``
    #: (folder/notebook/project). ``None`` disables container scoping.
    scope_column: str | None = None
    #: Human-readable container name, matched by :func:`librarian_core.rag.detect_scope`.
    scope_name_column: str | None = None
    #: Appended to WHERE verbatim — put soft-delete and policy flags here,
    #: e.g. ``"deleted_at IS NULL AND ai_policy = 'allowed'"``.
    extra_where: str = ""

    def validate(self) -> None:
        """Reject identifiers that are not plain names.

        Column names are interpolated into SQL (they cannot be bound
        parameters), so anything unusual is refused rather than escaped.
        ``extra_where`` is exempt by design — it is developer-authored SQL, so
        never build it from user input.
        """
        names = [
            self.table,
            self.embedding_column,
            self.text_column,
            self.source_id_column,
            self.title_column,
            self.chunk_index_column,
            self.tenant_column,
            self.scope_column,
            self.scope_name_column,
        ]
        for name in names:
            if name is None:
                continue
            if not all(part.replace("_", "").isalnum() for part in name.split(".") if part):
                raise ValueError(f"unsafe identifier in ChunkTableSpec: {name!r}")


class PgVectorRetrieval:
    """A :class:`RetrievalPort` over a pgvector chunk table.

    ``embed`` turns the query into a vector. Use the *same* model that produced
    the stored embeddings — mismatched models yield scores that look plausible
    and rank nonsense, which is far worse than an outright failure.
    """

    def __init__(self, db: AsyncSession, spec: ChunkTableSpec, embed: EmbedFn) -> None:
        spec.validate()
        self.db = db
        self.spec = spec
        self.embed = embed

    async def search(
        self,
        query: str,
        *,
        principal: Principal,
        limit: int,
        min_score: float,
        scope: Sequence[str] | None = None,
    ) -> list[Passage]:
        spec = self.spec
        vector = list(await self.embed(query))
        if not vector:
            return []

        params: dict[str, object] = {
            "embedding": "[" + ",".join(f"{v:.8f}" for v in vector) + "]",
            "min_score": min_score,
            "limit": limit,
        }
        where: list[str] = []

        if spec.tenant_column:
            where.append(f"{spec.tenant_column} = :tenant_id")
            params["tenant_id"] = principal.tenant_id

        if spec.scope_column:
            # Fail closed: no readable containers means no readable chunks.
            if not principal.scope_ids:
                return []
            where.append(f"{spec.scope_column} = ANY(:scope_ids)")
            params["scope_ids"] = list(principal.scope_ids)

        if scope and spec.scope_name_column:
            where.append(f"{spec.scope_name_column} = ANY(:scope_names)")
            params["scope_names"] = list(scope)

        if spec.extra_where:
            where.append(f"({spec.extra_where})")

        # 1 - cosine_distance == cosine similarity, so the threshold is an
        # interpretable "how alike" number rather than a raw distance.
        similarity = f"1 - ({spec.embedding_column} <=> CAST(:embedding AS vector))"
        where.append(f"{similarity} >= :min_score")

        sql = text(
            f"""
            SELECT {spec.source_id_column} AS source_id,
                   {spec.title_column}     AS title,
                   {spec.text_column}      AS chunk_text,
                   {spec.chunk_index_column} AS chunk_index,
                   {similarity}            AS score
            FROM {spec.table}
            WHERE {" AND ".join(where)}
            ORDER BY score DESC
            LIMIT :limit
            """
        )

        rows = (await self.db.execute(sql, params)).mappings().all()
        return [
            Passage(
                source_id=str(row["source_id"]),
                title=row["title"] or "",
                text=row["chunk_text"] or "",
                score=float(row["score"]),
                chunk_index=int(row["chunk_index"] or 0),
            )
            for row in rows
        ]

    async def scope_names(self, *, principal: Principal) -> list[str]:
        """Distinct container names the principal can read.

        Feed the result to ``LibrarianTurn.run(scope_candidates=...)`` to enable
        automatic narrowing when a question names its container.
        """
        spec = self.spec
        if not spec.scope_name_column:
            return []
        where: list[str] = [f"{spec.scope_name_column} IS NOT NULL"]
        params: dict[str, object] = {}
        if spec.tenant_column:
            where.append(f"{spec.tenant_column} = :tenant_id")
            params["tenant_id"] = principal.tenant_id
        if spec.scope_column:
            if not principal.scope_ids:
                return []
            where.append(f"{spec.scope_column} = ANY(:scope_ids)")
            params["scope_ids"] = list(principal.scope_ids)
        if spec.extra_where:
            where.append(f"({spec.extra_where})")

        sql = text(
            f"SELECT DISTINCT {spec.scope_name_column} AS name FROM {spec.table} WHERE {' AND '.join(where)}"
        )
        rows = (await self.db.execute(sql, params)).mappings().all()
        return [str(row["name"]) for row in rows if row["name"]]


__all__ = ["ChunkTableSpec", "PgVectorRetrieval"]
