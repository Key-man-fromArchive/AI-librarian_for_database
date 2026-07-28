"""Turning your application's user object into a scoped Principal.

Pick the adapter matching your isolation model, or write one — it is the
smallest port and the one most worth getting right, because everything the
librarian can read flows from it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from librarian_core.ports import Principal


class SingleUserACL:
    """No isolation: one user, everything visible.

    For personal tools and local deployments. Do not use where more than one
    person's data shares a database.
    """

    def __init__(self, *, user_id: int | str = 1, tenant_id: int | str = 0) -> None:
        self._user_id = user_id
        self._tenant_id = tenant_id

    async def resolve(self, user: Any = None) -> Principal:  # noqa: ARG002 — signature is the port
        return Principal(user_id=self._user_id, tenant_id=self._tenant_id)


class TenantACL:
    """Tenant isolation with an explicit list of readable containers.

    ``scope_loader`` returns the container ids this user may read. An empty
    result means *no access*, and every shipped retrieval adapter treats it that
    way — fail closed. If you write your own adapter, reproduce that: reading an
    empty scope as "unfiltered" is the classic way to leak an entire corpus.
    """

    def __init__(
        self,
        *,
        user_key: str = "user_id",
        tenant_key: str = "org_id",
        scope_loader: Callable[[Any], Awaitable[Sequence[int | str]]] | None = None,
    ) -> None:
        self._user_key = user_key
        self._tenant_key = tenant_key
        self._scope_loader = scope_loader

    async def resolve(self, user: Any) -> Principal:
        data = user if isinstance(user, dict) else getattr(user, "__dict__", {})
        user_id = data.get(self._user_key) if isinstance(data, dict) else None
        tenant_id = data.get(self._tenant_key) if isinstance(data, dict) else None
        if user_id is None:
            user_id = getattr(user, self._user_key, None)
        if tenant_id is None:
            tenant_id = getattr(user, self._tenant_key, 0)
        if user_id is None:
            raise ValueError(f"cannot resolve {self._user_key!r} from the supplied user object")

        scope: tuple[int | str, ...] = ()
        if self._scope_loader is not None:
            scope = tuple(await self._scope_loader(user))
        return Principal(user_id=user_id, tenant_id=tenant_id or 0, scope_ids=scope)


__all__ = ["SingleUserACL", "TenantACL"]
