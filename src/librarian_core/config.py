"""Tunables for the librarian turn.

Defaults are the values that survived production use on a multi-user research
notebook corpus. Where a default is non-obvious the reasoning is recorded next
to it — those comments are the most useful part of this file when you tune for
your own corpus.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LibrarianConfig:
    # --- Models -----------------------------------------------------------
    answer_model: str = "gpt-4o"
    #: Used only when the primary model fails *before producing any text*.
    #: Pick a different provider — the common failure is a provider-wide
    #: outage or a parameter one provider rejects, and a same-provider
    #: fallback fails identically.
    fallback_model: str = ""

    # --- Retrieval --------------------------------------------------------
    rag_enabled: bool = True
    #: Cosine similarity floor. Below ~0.45 unrelated passages start slipping
    #: in and the model dutifully cites them; above ~0.6 genuine answers get
    #: abstained away. Measure on your own goldset before moving it.
    min_similarity: float = 0.5
    #: Chunks pulled from the store before capping and selection.
    candidate_chunks: int = 40
    #: Ceiling on passages from any single document, so one long multi-topic
    #: document cannot crowd out every other source.
    per_source_cap: int = 2
    max_passages: int = 6
    #: Character budget for the whole retrieved-context block.
    max_context_chars: int = 12_000
    #: Floor on per-passage characters, so a generous budget cannot be eaten
    #: by the first long passage.
    min_passage_chars: int = 1_200

    # --- Conversation -----------------------------------------------------
    #: Character budget for replayed conversation history.
    max_history_chars: int = 24_000
    #: Characters of the first question used as the auto-generated title.
    auto_title_chars: int = 60

    # --- Generation -------------------------------------------------------
    max_output_tokens: int = 4_096
    #: Lower temperature when passages are present: grounded factual answering
    #: punishes creative drift. Without passages the answer is general
    #: knowledge, where some latitude reads better.
    temperature_grounded: float = 0.2
    temperature_ungrounded: float = 0.7

    # --- Governed memory --------------------------------------------------
    memory_enabled: bool = False
    memory_context_max_tokens: int = 1_500
    #: Approved memories older than this stop being injected. Long-lived facts
    #: should be re-approved rather than trusted indefinitely.
    memory_max_age_days: int = 365


DEFAULT_CONFIG = LibrarianConfig()

__all__ = ["DEFAULT_CONFIG", "LibrarianConfig"]
