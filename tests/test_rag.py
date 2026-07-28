"""Retrieval gating: the cap, the budget, the abstention."""

from __future__ import annotations

from librarian_core.config import LibrarianConfig
from librarian_core.ports import Passage, Principal
from librarian_core.rag import detect_scope, render_passages, retrieve_context, select_passages

PRINCIPAL = Principal(user_id=1, tenant_id=1)


def passage(source_id: str, score: float, text: str = "body", index: int = 0) -> Passage:
    return Passage(source_id=source_id, title=f"{source_id} title", text=text, score=score, chunk_index=index)


def test_per_source_cap_prevents_one_document_dominating():
    chunks = [passage("a", 0.9, index=i) for i in range(5)] + [passage("b", 0.6)]
    selected = select_passages(chunks, per_source_cap=2, max_passages=6)
    assert [p.source_id for p in selected] == ["a", "a", "b"]


def test_selection_stops_at_max_passages():
    chunks = [passage(f"d{i}", 0.9 - i / 100) for i in range(10)]
    assert len(select_passages(chunks, per_source_cap=2, max_passages=3)) == 3


def test_budget_is_shared_evenly_so_a_long_passage_cannot_starve_the_rest():
    chunks = [passage("a", 0.9, text="x" * 50_000), passage("b", 0.8, text="y" * 50_000)]
    section, citations = render_passages(chunks, max_chars=6_000, min_passage_chars=100)
    assert len(citations) == 2
    assert "y" in section  # the second passage survived


def test_rendering_numbers_citations_from_one():
    chunks = [passage("a", 0.9), passage("b", 0.8)]
    _, citations = render_passages(chunks, max_chars=4_000)
    assert [c.index for c in citations] == [1, 2]
    assert [c.source_id for c in citations] == ["a", "b"]


def test_empty_input_renders_nothing():
    section, citations = render_passages([], max_chars=4_000)
    assert section == ""
    assert citations == []


def test_scope_detection_matches_whole_names_and_tokens():
    names = ["roast-profiles", "cupping", "green inventory"]
    assert detect_scope("what is in the cupping notebook?", names) == ["cupping"]
    assert detect_scope("check roast settings", names) == ["roast-profiles"]
    assert detect_scope("unrelated question", names) == []


def test_single_character_tokens_do_not_match():
    """Otherwise a name like 'a-b' would match nearly every question."""
    assert detect_scope("something", ["a-b"]) == []


class ThresholdRetrieval:
    def __init__(self, passages):
        self.passages = passages
        self.last_scope = None

    async def search(self, query, *, principal, limit, min_score, scope=None):
        self.last_scope = scope
        return [p for p in self.passages if p.score >= min_score][:limit]


async def test_abstains_when_nothing_clears_the_threshold():
    retrieval = ThresholdRetrieval([passage("a", 0.2)])
    section, citations = await retrieve_context(
        retrieval, "question", principal=PRINCIPAL, config=LibrarianConfig(min_similarity=0.5)
    )
    assert section == ""
    assert citations == []


async def test_explicit_scope_is_passed_through_untouched():
    retrieval = ThresholdRetrieval([passage("a", 0.9)])
    await retrieve_context(retrieval, "q", principal=PRINCIPAL, scope=["cupping"])
    assert retrieval.last_scope == ["cupping"]


async def test_scope_is_auto_detected_from_the_question():
    retrieval = ThresholdRetrieval([passage("a", 0.9)])
    await retrieve_context(
        retrieval,
        "in the cupping notebook, what scored highest?",
        principal=PRINCIPAL,
        scope_candidates=["cupping", "roast-profiles"],
    )
    assert retrieval.last_scope == ["cupping"]


async def test_disabled_rag_skips_retrieval_entirely():
    retrieval = ThresholdRetrieval([passage("a", 0.99)])
    section, citations = await retrieve_context(
        retrieval, "q", principal=PRINCIPAL, config=LibrarianConfig(rag_enabled=False)
    )
    assert (section, citations) == ("", [])
    assert retrieval.last_scope is None
