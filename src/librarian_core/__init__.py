"""A portable, database-grounded AI librarian.

Give it four adapters — retrieval, ACL, LLM, session store — and it answers
questions from your own data with citations, abstaining when it has no evidence.

Quick start::

    from librarian_core import LibrarianTurn, TurnRequest, LibrarianConfig

    turn = LibrarianTurn(store=store, llm=llm, retrieval=retrieval,
                         config=LibrarianConfig(answer_model="gpt-4o"))
    async for sse_block in turn.run(TurnRequest(session_id=sid, content=q),
                                    principal=principal):
        ...

See ``docs/PORTING.md`` for adapter implementation, or run the interactive
installer skill in ``skills/ai-librarian/``.
"""

from .config import DEFAULT_CONFIG, LibrarianConfig
from .ports import (
    ACLPort,
    ChatMessage,
    Citation,
    LLMChunk,
    LLMError,
    LLMPort,
    LLMRequest,
    Passage,
    Principal,
    RetrievalPort,
    SessionStorePort,
    StoredMessage,
    StoredSession,
)
from .prompts import DEFAULT_SYSTEM_PROMPT
from .rag import retrieve_context, select_passages
from .sse import DONE, SSE_HEADERS, encode_chunk, encode_error, encode_event, extract_error_message
from .turn import LibrarianTurn, TurnRequest

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SYSTEM_PROMPT",
    "DONE",
    "SSE_HEADERS",
    "ACLPort",
    "ChatMessage",
    "Citation",
    "LLMChunk",
    "LLMError",
    "LLMPort",
    "LLMRequest",
    "LibrarianConfig",
    "LibrarianTurn",
    "Passage",
    "Principal",
    "RetrievalPort",
    "SessionStorePort",
    "StoredMessage",
    "StoredSession",
    "TurnRequest",
    "__version__",
    "encode_chunk",
    "encode_error",
    "encode_event",
    "extract_error_message",
    "retrieve_context",
    "select_passages",
]
