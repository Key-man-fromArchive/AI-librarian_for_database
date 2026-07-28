# Porting the librarian to your project

Four protocols. Implement them against the tables you already have, and skip any
that a shipped adapter already covers.

Work in this order — each step is verifiable before the next.

## 1. ACLPort — start here

Everything the librarian can read flows from this. Get it wrong and the rest of
the work is built on a leak.

```python
from librarian_core.ports import Principal

class MyACL:
    async def resolve(self, user) -> Principal:
        return Principal(
            user_id=user.id,
            tenant_id=user.org_id,
            scope_ids=tuple(await readable_folder_ids(user)),
        )
```

**The rule that matters:** an empty `scope_ids` means *no access*. If your
existing query layer drops the `WHERE folder_id = ANY(...)` clause when the list
is empty, it converts "sees nothing" into "sees everything". Check for this
before you build on it — and if you find it, you have found a pre-existing bug
worth fixing independently.

Single-tenant projects can use `SingleUserACL` and move on.

## 2. RetrievalPort

If you have a chunk table with a pgvector column, describe it and you are done:

```python
from librarian_adapters.pgvector_retrieval import ChunkTableSpec, PgVectorRetrieval

retrieval = PgVectorRetrieval(
    db,
    ChunkTableSpec(
        table="document_chunks",
        embedding_column="embedding",
        text_column="chunk_text",
        source_id_column="document_id",
        title_column="title",
        chunk_index_column="chunk_index",
        tenant_column="org_id",           # None for single-tenant
        scope_column="folder_id",         # None to disable container scoping
        scope_name_column="folder_name",  # enables automatic scope detection
        extra_where="deleted_at IS NULL AND ai_indexable = true",
    ),
    embed=my_embedding_function,
)
```

Writing your own? The contract:

```python
async def search(self, query, *, principal, limit, min_score, scope=None) -> list[Passage]
```

- Apply the ACL **inside the query**. The core does not filter afterwards.
- Return results ordered by descending `score`, already filtered to
  `score >= min_score`.
- Scores must be comparable across queries. Cosine similarity works; raw
  rank-fusion scores do not, and the threshold gate becomes meaningless with
  them — see `docs/ARCHITECTURE.md`.
- Use the **same embedding model** that produced the stored vectors. A mismatch
  produces scores that look reasonable and rank nonsense.

### No chunks yet?

Chunk at 800–1200 characters with ~15% overlap as a starting point, store one
row per chunk with its embedding, and keep `chunk_index` so citations can
deep-link. Whole-document embeddings are not a shortcut — see the architecture
notes on why document-level retrieval was abandoned.

## 3. LLMPort

Any OpenAI-compatible endpoint (OpenAI, Azure, OpenRouter, Together, Groq,
vLLM, Ollama):

```python
llm = OpenAICompatLLM(base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
```

Already have a model router with budgets, provider fallback and egress policy?
Keep it:

```python
from librarian_adapters.openai_compat_llm import CallableLLM

llm = CallableLLM(lambda req: my_router.stream(req.model, req.messages))
```

The wrapped callable may yield plain strings. **Yield `LLMError` for provider
failures rather than raising** — the turn needs to distinguish a failure before
first output (fall back) from one after it (keep the answer), and an exception
erases that distinction.

## 4. SessionStorePort

`SQLAlchemySessionStore` covers SQLAlchemy 2.0 async. Create the tables through
your own migration tool:

```python
from librarian_adapters.sqlalchemy_store import LibrarianBase

# Alembic autogenerate: add LibrarianBase.metadata to target_metadata
target_metadata = [YourBase.metadata, LibrarianBase.metadata]
```

Writing your own store? Two requirements that are easy to miss:

- **`append_message` must allocate `sequence` atomically.** Two concurrent turns
  in one session otherwise read the same counter and write duplicate sequences,
  silently reordering the conversation. The shipped adapter uses
  `SELECT ... FOR UPDATE`.
- **`commit()` must actually commit.** The turn calls it before streaming and
  after finalising, because a client that polls the thread mid-stream must see
  the row, and one that re-reads immediately after the response must not race an
  uncommitted transaction.

## 5. Wire it up

```python
turn = LibrarianTurn(
    store=SQLAlchemySessionStore(db),
    llm=llm,
    retrieval=retrieval,
    config=LibrarianConfig(answer_model="gpt-4o", fallback_model="claude-3-5-haiku"),
)
```

Pick a fallback from a **different provider**. The common failure is a
provider-wide outage or a parameter one provider rejects, and a same-provider
fallback fails identically.

Mount the shipped FastAPI router, or write your own — it is thin, and everything
it does is public API:

```python
app.include_router(
    build_librarian_router(dependencies=LibrarianDeps(
        get_principal=..., get_turn=..., get_store=...,
        get_scope_candidates=lambda p: retrieval.scope_names(principal=p),
    )),
    prefix="/api",
)
```

`get_scope_candidates` is optional and worth supplying: it lets a question that
names its container ("in the onboarding folder…") narrow retrieval automatically.

## 6. Consume the stream

```javascript
const response = await fetch(`/api/librarian/sessions/${id}/turn`, {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
  body: JSON.stringify({ content: question }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "", answer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const blocks = buffer.split("\n\n");
  buffer = blocks.pop() ?? "";

  for (const block of blocks) {
    const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
    if (!dataLine) continue;
    const raw = dataLine.slice(6);
    if (raw === "[DONE]") continue;

    if (block.startsWith("event: citations")) {
      setCitations(JSON.parse(raw).citations);   // before the text arrives
    } else if (block.startsWith("event: error")) {
      setError(JSON.parse(raw).message);         // do not drop this branch
    } else {
      answer += JSON.parse(raw).chunk ?? "";
      setAnswer(answer);
    }
  }
}
```

Two client-side mistakes worth naming, both of which shipped in the origin
system:

1. **Dropping the `event: error` branch.** Checking only for `data:` prefixes
   makes real provider errors invisible and turns a one-line diagnosis into a
   multi-day one.
2. **Treating post-stream work as part of the answer.** If reloading the thread
   or refreshing a sidebar happens inside the same `try` as the stream, a
   failure there marks a perfectly good answer as failed. Keep them separate.

## 7. Tune before you trust it

```bash
python eval/run_eval.py --goldset eval/your_goldset.json --sweep 0.4,0.45,0.5,0.55
```

Write 10–15 real questions naming the document each should cite, plus a few that
*nothing* should answer (`must_abstain: true`) — those are what stop you tuning
the threshold to zero and calling it success.
