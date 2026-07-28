# AI Librarian for Database

Ask questions of your own data and get answers that **cite their sources** — or
that say plainly when the evidence is not there.

This is the portable extraction of a librarian that has been running in
production against a multi-user research-notebook corpus. It is not a RAG
tutorial; it is the parts that only show up after real users hit them.

```python
turn = LibrarianTurn(store=store, llm=llm, retrieval=retrieval)

async for block in turn.run(TurnRequest(session_id=sid, content=question),
                            principal=principal):
    yield block   # Server-Sent Events: citations, then text, then done
```

## What you actually get

**Answers that abstain.** When nothing clears the relevance threshold, the
librarian says so instead of inventing a plausible answer from your documents.
A confident wrong citation costs more than an admitted gap.

**Citations resolved before the text arrives.** `[1]`, `[2]` markers are backed
by real passage references streamed *ahead* of the answer, so your UI can link
them as tokens appear.

**Failures that tell the truth.** A provider error after text has streamed is
not a failed answer, and is never displayed as one. A failure before any output
falls back to a second model — and the user is told it happened. Errors surface
the provider's own words, because "HTTP 400" is not a diagnosis.

**Retrieval that survives a real corpus.** Chunk-level cosine similarity with an
interpretable threshold, a per-document cap so one sprawling file cannot fill
every slot, and a fair character budget so the last passage is not starved by
the first.

**Memory nobody snuck in.** Optional governed memory: nothing reaches a prompt
without a human approving it, every approved fact traces to a quoted excerpt,
and contradictory facts about the same subject cannot coexist.

**No schema of its own to adopt.** Four small protocols. Implement them against
the tables you already have.

## Install

```bash
pip install "ai-librarian-for-database[all] @ git+https://github.com/Key-man-fromArchive/AI-librarian_for_database"
```

The core has **zero runtime dependencies**. SQLAlchemy, httpx and FastAPI are
optional extras used only by the adapters you choose.

## The four ports

| Port | Answers the question | Shipped adapter |
|---|---|---|
| `RetrievalPort` | What passages are relevant? | `PgVectorRetrieval` |
| `ACLPort` | What may this user see? | `SingleUserACL`, `TenantACL` |
| `LLMPort` | Generate the answer | `OpenAICompatLLM`, `CallableLLM` |
| `SessionStorePort` | Where do conversations live? | `SQLAlchemySessionStore` |

Bringing your own model router? Wrap it in `CallableLLM` — keep your budget
accounting, provider fallbacks and egress policy exactly as they are.

## Interactive install

The repository ships a Claude Code skill that installs the librarian into an
existing project by **inspecting it first**: schema, vector support, migration
tool, tenancy model — then asking only what inspection could not settle.

```
cp -r skills/ai-librarian ~/.claude/skills/
```

Then, in the target project: *"install the AI librarian"*. It reads your models,
proposes the knowledge source, and generates the adapters, migration and router
for your schema. See [`skills/ai-librarian/SKILL.md`](skills/ai-librarian/SKILL.md).

## Minimal wiring

```python
from librarian_core import LibrarianConfig, LibrarianTurn, TurnRequest
from librarian_adapters.acl import SingleUserACL
from librarian_adapters.openai_compat_llm import OpenAICompatLLM
from librarian_adapters.pgvector_retrieval import ChunkTableSpec, PgVectorRetrieval
from librarian_adapters.sqlalchemy_store import SQLAlchemySessionStore

retrieval = PgVectorRetrieval(
    db,
    ChunkTableSpec(
        table="document_chunks",
        embedding_column="embedding",
        text_column="chunk_text",
        source_id_column="document_id",
        title_column="title",
        tenant_column=None,            # single-tenant
        extra_where="deleted_at IS NULL",
    ),
    embed=my_embedding_function,
)

turn = LibrarianTurn(
    store=SQLAlchemySessionStore(db),
    llm=OpenAICompatLLM(api_key_env="OPENAI_API_KEY"),
    retrieval=retrieval,
    config=LibrarianConfig(answer_model="gpt-4o", fallback_model="gpt-4o-mini"),
)

principal = await SingleUserACL().resolve(None)
```

A runnable end-to-end example with no database and no API key lives in
[`examples/minimal_fastapi/`](examples/minimal_fastapi/).

## Tuning is the work

Defaults are a starting point, not an answer. The single most important number
is `min_similarity`, and you cannot pick it by intuition:

```bash
python eval/run_eval.py --sweep 0.35,0.4,0.45,0.5,0.55,0.6
```

```
 min_sim   recall@k    full   abstain   false_abs     acc
    0.15      1.000   1.000     0.000           0   0.800
    0.25      1.000   1.000     0.133           0   0.933
    0.35      0.500   0.500     0.600           6   0.600
```

Read recall and abstention **together**. A threshold of 0 retrieves everything
and never abstains — perfect recall, useless system. Write 10–15 real questions
naming the document each should cite, in the format of
[`eval/goldset.synthetic.json`](eval/goldset.synthetic.json), and sweep.

## Before you publish anything derived from private data

This project exists because a working system was extracted from a private
codebase. That extraction nearly published an evaluation goldset containing real
product formulations, a colleague's name inside a docstring, and internal
hostnames in configuration defaults. None of it was key-shaped, so a
secret-scanner would have passed all three.

```bash
python tools/scan_secrets.py            # working tree
python tools/scan_secrets.py --staged   # pre-commit
```

It checks credentials, personal data, and internal infrastructure, redacts
findings in its own output, and runs in this repository's CI so the repository
stays publishable.

## Scope of v0.1

**In:** the Ask turn, chunk RAG with abstention, citations, SSE, model fallback,
governed memory, Python/FastAPI/SQLAlchemy adapters, evaluation harness.

**Not in:** deep multi-step research agents, web-evidence gathering, reranking,
and a frontend. The origin system has all four; they are far more coupled to its
product decisions, and shipping them here would be guessing at yours.

**Python only.** The port contract in [`docs/PORTING.md`](docs/PORTING.md) is
what an implementation in another language would satisfy.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — why chunk-level retrieval,
  why abstention, why errors are events
- [`docs/PORTING.md`](docs/PORTING.md) — implementing the four ports against
  your own schema
- [`skills/ai-librarian/SKILL.md`](skills/ai-librarian/SKILL.md) — the guided
  installer

## Licence

MIT. See [`LICENSE`](LICENSE).
