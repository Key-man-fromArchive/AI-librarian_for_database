---
name: ai-librarian
description: Install a database-grounded AI librarian into this project — cited answers over the project's own data, with abstention when evidence is missing. Use when the user asks to add an AI librarian, RAG chat, "ask my data" / "chat with my database" features, or mentions AI-librarian_for_database. Inspects the existing schema and stack first, then installs through a short interview.
---

# Installing the AI Librarian

You are installing a librarian into an **existing** project. The single biggest
failure mode is guessing: inventing a table that does not exist, assuming
pgvector when there is none, or wiring an ACL that leaks other tenants' rows.

So: **inspect first, ask second, generate third.** Never ask a question the
codebase already answers, and never assume an answer the codebase does not give.

## Step 0 — Inspect before asking anything

Run these before your first question to the user. Their results decide which
questions are worth asking.

```bash
# Stack
ls pyproject.toml requirements*.txt setup.py package.json go.mod 2>/dev/null
grep -riE "fastapi|django|flask|sqlalchemy|sqlmodel" --include="*.txt" --include="*.toml" . | head

# Existing models / tables
find . -name "models*.py" -o -name "models" -type d | grep -v node_modules | head
grep -rn "__tablename__" --include="*.py" . | grep -v node_modules | head -40

# Vector support
grep -rniE "pgvector|vector\(|embedding|sqlite-vec|faiss|qdrant|chroma|weaviate" \
  --include="*.py" --include="*.sql" --include="*.toml" . | grep -v node_modules | head -20

# Migration tool
ls alembic.ini migrations/ 2>/dev/null

# Multi-tenancy signal
grep -rn "org_id\|tenant_id\|workspace_id\|team_id" --include="*.py" . | grep -v node_modules | head
```

If a live database is reachable, prefer it over source inspection — the schema
in the database is the truth:

```bash
psql "$DATABASE_URL" -c "\dt"
psql "$DATABASE_URL" -c "\d+ <candidate_table>"
psql "$DATABASE_URL" -c "SELECT extname FROM pg_extension;"   # is vector installed?
```

Summarise what you found in three or four lines before asking anything. The
user should see you have read their project.

## Step 1 — The interview

Ask **only** what inspection left genuinely open. Present findings as defaults
to confirm, not as blank questions. Use the AskUserQuestion tool with your
detected value as the first, recommended option.

### 1. Stack — usually already answered
State it: "FastAPI + SQLAlchemy 2.0 async + Alembic. Installing the Python
adapters." Ask only if genuinely ambiguous. If the project is **not** Python,
stop and say so plainly: v0.1 ships Python adapters only, and the port contract
in `docs/PORTING.md` is what a port to another language would implement.

### 2. Knowledge source — the decisive question
Which table holds the text to answer from, and which column is the body?

List the candidate tables you found with row counts and a likely text column, and
let the user pick. Get this wrong and every answer is grounded in the wrong data.

```sql
SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT 15;
```

### 3. Chunks and embeddings
Three cases, in order of preference:

- **A chunk table with embeddings already exists** → write a `ChunkTableSpec`
  naming its columns. Nothing to build. Confirm the embedding model that
  produced them; a mismatched query model ranks nonsense while looking fine.
- **Documents exist, no chunks** → generate a chunking + embedding backfill
  script. Ask for chunk size (default 800–1200 chars, overlap ~15%) and the
  embedding model.
- **No pgvector** → check `CREATE EXTENSION vector;` is permitted. If not,
  offer an alternative retrieval adapter and be explicit that the user must
  implement `RetrievalPort` against it.

### 4. Permission model — fail closed
Ask directly: *"Can every user see every document, or is access scoped?"*

- Single user / everything shared → `SingleUserACL`.
- Scoped → find the existing ACL. Reuse it; do not invent a parallel one.
  Identify the tenant column and the container column, then write the
  `scope_loader`.

State the rule out loud when you generate it: **an empty scope means no access,
never all access.** If the project's existing ACL treats empty as unfiltered,
flag it — you have found a pre-existing leak, and it is worth reporting even
though it is outside this installation.

### 5. LLM provider
Ask which provider and model. Then:

- Write config to `.env` and `.env.example`, **keys in `.env` only**.
- Confirm `.env` is git-ignored. If it is not, fix that first and say so.
- Never print a key back to the user, never write one into a source file, never
  commit one.
- If the project already has a model router, wrap it with `CallableLLM` rather
  than adding a second path to the same providers.

### 6. Surface
API only, or API plus a minimal chat UI? Default to API only — most projects
have their own component conventions, and a generated UI usually gets rewritten.

## Step 2 — Generate

Write, in this order, showing a diff summary as you go:

1. `librarian_config.py` — `LibrarianConfig` with the chosen models and
   thresholds.
2. `librarian_adapters_local.py` — the project's `ChunkTableSpec`, ACL wiring,
   embedding function, and dependency providers.
3. Migration creating `librarian_sessions` and `librarian_messages` (via the
   project's own migration tool — never hand-edit a live schema).
4. Router mounting `build_librarian_router(...)`.
5. `.env.example` entries, with placeholder values only.
6. A smoke test that runs one turn against a stub LLM and asserts the answer
   cites a real passage.

Then verify, and report what actually happened:

```bash
alembic upgrade head          # or the project's migration command
pytest tests/test_librarian_smoke.py -v
```

## Step 3 — Tune on real questions

An untuned threshold is the difference between a useful librarian and one that
either abstains constantly or cites unrelated passages. Ask the user for 10–15
real questions with the document each should cite, put them in the goldset
format from `eval/goldset.synthetic.json`, and run:

```bash
python eval/run_eval.py --goldset eval/your_goldset.json
```

Read `recall@k` and `abstain_rate` together, and move `min_similarity` in steps
of 0.05:

| Symptom | Meaning | Move |
|---|---|---|
| Low recall, low abstention | citing the wrong passages | raise the threshold |
| Low recall, high abstention | genuine answers filtered out | lower the threshold |
| High recall, high abstention | threshold is near-right; look at chunk size | leave it |

## Rules

- **Never invent schema.** If you cannot find a table, ask. A generated adapter
  pointing at a nonexistent column fails at the first question, in production.
- **Never write a secret to a tracked file.** Keys live in `.env`.
- **Never weaken the ACL to make a demo work.** A librarian that answers from
  another tenant's data is worse than one that answers nothing.
- **Keep abstention on.** The instinct to lower `min_similarity` to zero so it
  "always answers" converts an honest system into a confident liar.
- **Report verification honestly.** If the smoke test fails, say so with the
  output; do not describe the installation as complete.
