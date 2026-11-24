# askmystuff: design

## Question

Ask a question about one of your devices, get an answer that cites the manual and the page.
Behind it, one measured question: **does a device manual still need RAG when the whole manual fits in the prompt?**
askmystuff answers it with numbers (correctness, latency, cost) on a fixed set of reference questions.

## Two answer modes

| Mode | What goes into the prompt |
|---|---|
| `rag` | The 5 best chunks of the selected device's manual (hybrid search) |
| `full` | The whole manual of the selected device, no retrieval |

Both must answer only from the manual, cite `[manual, page]`, and say "not found in the manual" otherwise.

## Devices

Eight real devices with a French PDF manual from the manufacturer:
phone, laptop, TV, Chromecast, Bluetooth speaker, Bluetooth headphones, printer, car.
Two pairs are close on purpose (TV / Chromecast, speaker / headphones): they test that retrieval stays within the selected device.
The car manual (300+ pages) is where `full` gets expensive or overflows.

The repository holds the manuals' links (`data/manuals.yaml`) and a download script, never the PDFs.

## Architecture

```
  frontend (Vite, React, TS)     backend (FastAPI)              data
 ┌─────────────────────────┐   ┌──────────────────────────┐   ┌───────────────────┐
 │ home: prompt + devices  │──▶│ POST /ask                │──▶│ Postgres+pgvector │
 │ inventory, benchmark    │◀──│ 1. filter by device      │◀──│ devices           │
 └─────────────────────────┘   │ 2. hybrid search (RRF)   │   │ documents         │
                               │ 3. LLM answer + citations│   │ chunks            │
                               └────────────┬─────────────┘   └───────────────────┘
                                            ▼ one trace per question
                                        Langfuse

 offline: fetch_manuals → ingest → eval
```

- **Ingestion**: PyMuPDF text per page, language detected per page, French pages kept. One chunk per page (a long page is split and keeps its number), section title prepended when found.
- **Retrieval**: pgvector similarity + Postgres full-text search (`french` config, needed for error codes such as `E21`), fused with reciprocal rank fusion, filtered by device, top 5.
- **Models**: one interface, `llm.complete()`, with two backends: the Claude CLI (development, CI evaluation) and the Gemini API (public demo, reference benchmark run). Embeddings come from Google's multilingual embedding model.
- **Storage**: Postgres + pgvector in Docker locally and in CI; Supabase hosts the demo database. Same SQL everywhere.

## Evaluation

- **Reference set**: 40 questions, 5 per device: single-page, two-page, no-answer (must refuse), and cross-device traps.
  Questions are drafted by an LLM together with an exact quote from the expected page; a script rejects any question whose quote is not on that page; a human checks a random sample of 10.
- **Metrics**: recall@5 (expected page among retrieved chunks), correctness and faithfulness (LLM judge from another model family than the answering model, checked by hand on 10 answers), correct refusals, median and p95 latency per stage (embedding, search, model), median cost.
  `ask()` returns these timings; Langfuse keeps them per question. A later agent mode is one more `mode`, measured the same way.
- **In CI**: every pull request that touches ingestion, retrieval or prompts runs a 15-question subset and posts the difference with `main`. A drop in recall@5 or correct refusals fails the check.
- **Reference run**: one full run of both modes with the demo's model, published in `eval/results/` and in the README.
- With 40 questions, a 1 or 2 point difference is noise; results say so.

## Demo

Three pages, in a clean, airy style with automatic dark mode:

- **Home**: one large prompt; device chips under it pick the manual (the last one is remembered).
  The answer streams in; each citation is a `p. 42` chip that opens the manual at that page.
  A quiet line under it gives the timings and cost: `Search 105 ms · Answer 1.9 s · $0.001`.
- **Inventory**: one card per device (line icon by kind, name, model, pages, manual link); a card opens Home with that device selected.
- **Benchmark**: `rag` against `full` (correctness, latency, cost), read from `eval/results/`.

- Fixed devices, `rag` mode only; `full` is shown through its measured results.
- Rate limit per visitor (10 questions per hour), token cap per answer, budget alerts.
- Deployed on Azure Container Apps on each merge to `main`, with a smoke test and rollback.

## How it is built

Coding agents build askmystuff from GitHub issues: each issue has acceptance criteria, an agent writes tests then code and opens a pull request, a reviewer agent reviews it, and CI, including the evaluation, decides what can merge.

## Out of scope for v1

Books, adding devices from the public demo, live `full` mode, an agent mode (LangGraph), an MCP server.
