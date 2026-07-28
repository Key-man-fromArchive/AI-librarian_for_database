# Architecture, and the reasoning behind it

Every decision here was made twice: once the obvious way, then again after the
obvious way failed in production. This document records the second version and
what forced it.

## Chunk-level retrieval with an interpretable threshold

**The obvious approach.** Rank documents with hybrid search (full-text +
semantic, fused with Reciprocal Rank Fusion), take the top few, feed them in.

**Why it failed.** RRF scores are computed from *ranks*, not similarity. Every
result lands within roughly 0.01 of every other, so there is no value at which
you can say "below this, do not cite". Without a meaningful cut-off you must
either always include the top N — which guarantees irrelevant context on
questions your corpus cannot answer — or never abstain at all.

Document-level retrieval fails a second way: a long document routinely covers
several unrelated topics. Retrieving "the document" hands the model 20,000
characters of which 200 are relevant, and the model reliably finds something
plausible in the other 19,800.

**What works.** Cosine similarity over chunks. The score means something
("how alike are these two vectors"), it is comparable across queries, and it
points at the specific passage. That single property is what makes abstention
possible, and abstention is what makes citations trustworthy.

**Chunk merging was tried and reverted.** Concatenating adjacent chunks to
restore context sounded right and measurably hurt table-heavy sources: joining
neighbouring chunks of an extracted table scrambles column alignment further.
One best chunk per passage is deliberate.

## Abstention as a feature

`retrieve_context` returns `("", [])` when nothing clears the threshold, and the
turn then instructs the model to say so.

The pressure to remove this is constant, because a system that sometimes answers
"I do not have that" feels worse in a demo than one that always answers. It is
not worse. A librarian that fabricates one figure in twenty cannot be trusted
for any figure, so every answer must then be verified by hand — which is the
entire cost the tool was supposed to remove.

Retrieval *errors* abstain too, rather than propagating. A degraded general
answer beats a 500, and the caller has no better recovery available.

## The evidence gate, and why it was loosened

The first grounding prompt forbade any claim not present in the passages.
Answers became "that information is not in your documents", full stop. Users
stopped asking.

The current prompt splits the rule in two:

- Facts about *the user's own data* must come from passages and cite them.
- General domain knowledge is **encouraged**, presented plainly as general
  knowledge.

That distinction is the whole trick. Strictness applies where fabrication is
dangerous (your data) and not where the model is genuinely useful (the field's
knowledge). See the docstring in `prompts.py` for the third clause — DON'T
CONFLATE — which came from the model merging two products' figures into one
confidently wrong record.

## A turn fails only if it produced nothing

`emitted` tracks whether any text reached the client. It gates three decisions:

| Situation | Behaviour |
|---|---|
| Error, nothing emitted | fall back to a second model; if that also fails, surface the error |
| Error, text already emitted | keep the answer, log the error, mark the message **complete** |
| No error | complete normally |

The middle row is a bug fix, not a nicety. Users saw complete, correct answers
stamped "AI response failed" because a post-stream step had thrown. It teaches
people to distrust output that is in fact fine — the most expensive failure mode
in the system, because it is invisible in metrics.

Fallback is confined to the "nothing emitted" case for a different reason:
switching models mid-answer splices two different answers together.

## Errors are events, not exceptions

`LLMPort` yields `LLMError` instead of raising. Raising erases the distinction
between "failed before producing anything" and "failed midway", which is exactly
the distinction the turn needs.

The same principle applies at the SSE layer. Errors travel as
`event: error\ndata: {...}` blocks, and `extract_error_message` exists because a
consumer once checked only for `data:` prefixes: error blocks were silently
dropped, and a provider rejecting one request parameter surfaced to users as
"no content generated". Diagnosis took days. Anything consuming an SSE stream
from another producer should route error blocks through that helper.

## Persistence in `finally`, `DONE` outside it

A client that disconnects mid-stream must still leave a complete stored message —
otherwise the thread shows a truncated answer forever and rows stick in
`streaming`. So finalisation lives in `finally`.

But an async generator may not `yield` while closing; doing so raises
`RuntimeError: async generator ignored GeneratorExit`. Awaiting is fine. So the
`finally` block awaits persistence and the terminating `DONE` is emitted on the
normal path. There is a test for exactly this, because the first version had the
bug.

## Committing before responding

Every write path commits before returning, rather than relying on framework
teardown. Teardown may commit *after* the response is sent, and clients
routinely re-read immediately — a create followed by a read races an
uncommitted transaction and 404s. This was a production incident in the origin
system; the fix is one line and the diagnosis was not.

## Fail-closed authorisation

`Principal.scope_ids` empty means **no readable containers**, never "unfiltered".
`PgVectorRetrieval` returns `[]` immediately in that case.

The failure this prevents is a specific, common one: code that builds
`WHERE container_id = ANY(:ids)` and skips the clause when the list is empty,
turning "this user can see nothing" into "this user can see everything". The ACL
is applied inside the SQL for the same reason — filtering in Python means the
database already returned rows the user may not see, one log line away from a
leak.

## Governed memory

Long-term memory that accumulates silently is a liability: one bad inference
becomes permanent context, quietly poisoning later answers, with no record of
where it came from.

Hence three rules, enforced in `memory/governance.py`:

1. **Approval before injection.** Only `active` memories are eligible, and only
   a human moves a memory there.
2. **No approval without provenance.** Non-exempt types need a source with a
   verbatim excerpt of at least 12 characters. Paraphrase defeats the purpose: a
   reviewer must be able to check the claim against the original words.
3. **No contradictions.** Supersession matches on `subject_key`, so a new fact
   retires the old one. Two contradictory active memories would put the
   contradiction into the prompt and let the model pick.

Memory content is user-influenced text placed in a **system** message — the
highest-trust position in the prompt. `context.sanitize` is therefore blunt: a
false positive garbles one line of context, while a false negative hands an
attacker the system prompt.

## Layout

```
src/librarian_core/         no framework, no ORM, no SDK — pure logic
  ports.py                  the four protocols
  turn.py                   orchestration; the emitted/fallback rules
  rag.py                    threshold gate, per-source cap, rendering
  prompts.py                the evidence gate
  sse.py                    wire format + error extraction
  memory/                   governance state machine, ranking, sanitising
src/librarian_adapters/     everything that touches the outside world
```

The core imports nothing but the standard library. That is not purism: it means
the interesting logic is testable without a database, a network, or an API key,
and the 58-test suite runs in under half a second.
