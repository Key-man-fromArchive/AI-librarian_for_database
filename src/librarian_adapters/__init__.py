"""Ready-made implementations of the four core ports.

Import only what you need — FastAPI, SQLAlchemy and httpx are optional extras,
and each module imports its own dependency lazily at module level so an unused
adapter never forces an install.
"""

from .acl import SingleUserACL, TenantACL

__all__ = ["SingleUserACL", "TenantACL"]
