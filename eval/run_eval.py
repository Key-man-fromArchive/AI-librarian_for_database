#!/usr/bin/env python3
"""Measure retrieval quality against a goldset.

This harness scores **retrieval only** — did the right passages reach the model?
That is deliberate. Retrieval is where threshold tuning pays off, it is
deterministic, and it costs nothing to run. Answer quality needs a judge and a
budget; get retrieval right first, because no prompt recovers from passages that
never arrived.

Two numbers matter, and only together:

``recall@k``
    Of the questions with a known correct source, how often did it get
    retrieved? Raise the threshold and this falls.

``abstain_rate``
    How often nothing cleared the bar. On ``must_abstain`` questions that is the
    correct answer; elsewhere it is a miss.

Optimising either alone gives a degenerate system: threshold 0 retrieves
everything (perfect recall, never abstains, cites nonsense); threshold 1
abstains always (perfect abstention, answers nothing).

Usage::

    python eval/run_eval.py                                  # built-in stub retrieval
    python eval/run_eval.py --goldset eval/mine.json
    python eval/run_eval.py --sweep 0.35,0.4,0.45,0.5,0.55,0.6

Plug in real retrieval with ``--retrieval mypkg.module:factory``, where the
factory returns an object implementing ``RetrievalPort``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from librarian_core.config import LibrarianConfig
from librarian_core.ports import Passage, Principal
from librarian_core.rag import retrieve_context

_WORD = re.compile(r"[\w가-힣]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text) if len(t) > 1}


class LexicalStubRetrieval:
    """Jaccard-overlap retrieval over the goldset's own documents.

    Not a serious retriever — it exists so the harness runs with no database,
    no embedding model and no API key, which makes the goldset format and the
    metrics inspectable in one command. Numbers from this stub describe the
    stub. Point ``--retrieval`` at your real adapter before drawing conclusions.
    """

    def __init__(self, documents: Sequence[dict]) -> None:
        self.documents = list(documents)

    async def search(
        self,
        query: str,
        *,
        principal: Principal,
        limit: int,
        min_score: float,
        scope: Sequence[str] | None = None,
    ) -> list[Passage]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        results: list[Passage] = []
        for doc in self.documents:
            if scope and doc.get("scope") not in scope:
                continue
            doc_tokens = _tokens(f"{doc['title']} {doc['text']}")
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens)
            score = overlap / len(query_tokens)
            if score < min_score:
                continue
            results.append(
                Passage(
                    source_id=doc["source_id"],
                    title=doc["title"],
                    text=doc["text"],
                    score=round(score, 4),
                    chunk_index=0,
                )
            )
        results.sort(key=lambda p: p.score, reverse=True)
        return results[:limit]


@dataclass
class Outcome:
    question_id: str
    must_abstain: bool
    abstained: bool
    retrieved: list[str]
    expected: list[str]

    @property
    def hit(self) -> bool:
        return bool(set(self.retrieved) & set(self.expected))

    @property
    def full_hit(self) -> bool:
        return bool(self.expected) and set(self.expected).issubset(self.retrieved)

    @property
    def correct(self) -> bool:
        return self.abstained if self.must_abstain else self.hit


async def evaluate(goldset: dict, retrieval, config: LibrarianConfig) -> list[Outcome]:
    principal = Principal(user_id=1, tenant_id=1)
    scope_names = sorted({d["scope"] for d in goldset.get("documents", []) if d.get("scope")})
    outcomes: list[Outcome] = []

    for item in goldset["questions"]:
        _, citations = await retrieve_context(
            retrieval,
            item["question"],
            principal=principal,
            config=config,
            scope=item.get("scope"),
            scope_candidates=scope_names,
        )
        outcomes.append(
            Outcome(
                question_id=item["id"],
                must_abstain=bool(item.get("must_abstain")),
                abstained=not citations,
                retrieved=[c.source_id for c in citations],
                expected=list(item.get("expected_source_ids") or []),
            )
        )
    return outcomes


def summarise(outcomes: Sequence[Outcome]) -> dict:
    answerable = [o for o in outcomes if not o.must_abstain]
    abstainable = [o for o in outcomes if o.must_abstain]
    return {
        "total": len(outcomes),
        "recall_at_k": round(sum(o.hit for o in answerable) / len(answerable), 3) if answerable else 0.0,
        "full_recall": round(sum(o.full_hit for o in answerable) / len(answerable), 3) if answerable else 0.0,
        "abstain_rate": round(sum(o.abstained for o in outcomes) / len(outcomes), 3) if outcomes else 0.0,
        "correct_abstentions": (
            round(sum(o.abstained for o in abstainable) / len(abstainable), 3) if abstainable else None
        ),
        "false_abstentions": sum(o.abstained for o in answerable),
        "accuracy": round(sum(o.correct for o in outcomes) / len(outcomes), 3) if outcomes else 0.0,
    }


def load_retrieval(spec: str | None, goldset: dict):
    if not spec:
        return LexicalStubRetrieval(goldset.get("documents", []))
    module_name, _, factory_name = spec.partition(":")
    if not factory_name:
        raise SystemExit("--retrieval must be 'package.module:factory'")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--goldset", type=Path, default=Path(__file__).parent / "goldset.synthetic.json")
    parser.add_argument(
        "--retrieval", default=None, help="'package.module:factory' returning a RetrievalPort"
    )
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--sweep", default=None, help="comma-separated thresholds to compare")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    goldset = json.loads(args.goldset.read_text(encoding="utf-8"))
    retrieval = load_retrieval(args.retrieval, goldset)

    base = LibrarianConfig()
    overrides: dict = {}
    if args.min_similarity is not None:
        overrides["min_similarity"] = args.min_similarity
    elif args.retrieval is None:
        # The lexical stub scores token overlap, whose useful range sits well
        # below the cosine default. Without this the stub abstains on everything
        # and the harness looks broken on first run.
        overrides["min_similarity"] = 0.25
    if args.max_passages is not None:
        overrides["max_passages"] = args.max_passages

    thresholds = (
        [float(v) for v in args.sweep.split(",")]
        if args.sweep
        else [overrides.get("min_similarity", base.min_similarity)]
    )

    report = {"goldset": str(args.goldset), "dataset_version": goldset.get("dataset_version"), "runs": []}
    for threshold in thresholds:
        config = LibrarianConfig(**{**overrides, "min_similarity": threshold})
        outcomes = asyncio.run(evaluate(goldset, retrieval, config))
        report["runs"].append({"min_similarity": threshold, **summarise(outcomes)})

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"goldset: {args.goldset.name}  ({goldset.get('dataset_version')})")
    if args.retrieval is None:
        print("retrieval: built-in lexical stub — format demo only, not a quality measurement")
    print()
    header = f"{'min_sim':>8}  {'recall@k':>9}  {'full':>6}  {'abstain':>8}  {'false_abs':>10}  {'acc':>6}"
    print(header)
    print("-" * len(header))
    for run in report["runs"]:
        print(
            f"{run['min_similarity']:>8.2f}  {run['recall_at_k']:>9.3f}  {run['full_recall']:>6.3f}  "
            f"{run['abstain_rate']:>8.3f}  {run['false_abstentions']:>10}  {run['accuracy']:>6.3f}"
        )
    print("\nRead recall@k and abstain together: a threshold that maximises one alone is degenerate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
