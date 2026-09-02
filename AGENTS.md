# AGENTS.md

Read this before changing anything in `src/` or `tests/`.

## What this project does

Builds **grounded** Rolodex profiles: for each entity, an `EntityBio` whose every
emitted primitive value carries a span pointing back at the text that supports
it. A profile with no grounding is not a cheaper profile, it is a different and
worthless artifact — the grounding is the product.

```sh
uv run python -m rolodex_v1.build_profiles --dry-run
```

Two **context regimes** decide what evidence a profile is built from, and they
are not interchangeable:

- `memory` — the entity's aggregated memories. This is what production does.
- `chunks` — ±250-token windows of the source documents around each mention of
  the entity, overlapping windows merged, each surviving window wrapped as one
  synthetic memory so the grounding contract is unchanged. This is what the
  agentic run and the source-doc arm of the Luna experiment consume.

The same entity yields different profiles under each, so the regime is in the
output filename rather than a flag you have to remember you passed.

## Where this came from, and where the data is

The predecessor is `~/dev/z-r-eval_prompt_and_autotune/src/tune_prompt/rolodex_v1/`
(and its `scripts/rolodex_v1/`). It was a prompt-tuning adapter that grew a
pipeline inside it, so its corpus locations were module constants. Read it for
the algorithms — `task_model/grounding.py` and `task_model/extraction.py` are
the parts worth carrying over — not for its structure.

Nothing in this repo hardcodes a corpus. The two locations are configuration:

| | flag | env |
| --- | --- | --- |
| memories bundle | `--memories-bundle` | `ROLODEX_V1_MEMORIES_BUNDLE` |
| source documents | `--source-docs` | `ROLODEX_V1_SOURCE_DOCS` |

What those pointed at in the predecessor, none of which this repo depends on:

- memories bundle — `~/dev/rolodex/test_artifacts/obama_entity_memory_map.json`
  (12 profiles), extended to 22 by `prepare_context_experiment.py`.
- source documents — `/tmp/obama_srcdocs`, a cache keyed by content digest.
- human labels — `~/dev/rolodex/biographies_human_annotated/obama.json`,
  validated into `.ground_truth/obama_22.json` in the predecessor.
- splits — `obama_splits_22.json`, the original 12 assignments verbatim plus a
  deterministic 6/2/2 over the 10 added profiles. Holding the original twelve
  fixed is the only reason the new numbers compare to the tuned prompts.

`prompts/extraction_addendum.jinja` is the tuned best prompt, copied out of the
predecessor's `runs/luna-high/logs/rolodex_v1-20260803-094455/` as
`initial_prompt_addendum-best-iter6.jinja` — the best of a 10-iteration
`gpt-5.6-luna` high-effort run. Every rule in it
was bought with an editor-model iteration against held-out labels, so treat an
edit to it as an experiment that needs a score, not a wording change.

Extraction runs on `gpt-5.6-luna` and needs `OPENAI_API_KEY_SMALL`; the editor
model is `gpt-5.6-sol` and needs `OPENAI_API_KEY_BIG`. The predecessor read both
from the environment first and then from the Rolodex checkout's `.env.local`.

## Traps

Nothing has shipped from this repo yet, so these are inherited — each one
already cost someone a run in the predecessor.

**1. The context budget is measured on the rendered request, not the excerpts.**
`build_grounding_context` adds a per-window header and re-wraps every line to 200
characters with a line-number prefix. That inflated one ~700K-token excerpt set
into a **1,168,995-token** request. Fit the budget against the rendered messages.

**2. The deployed input limit is lower than the documented one.** `BIO_MAX_INPUT_TOKENS`
derives from a 1,050,000-token context window, but a 922,518-token request came
back rejected with "Input tokens exceed the configured limit of 922000 tokens".
Budget against 922,000, and leave headroom for the per-run prompt addendum — it
is not free and is not known at fit time (`prompts/extraction_addendum.jinja` is
~2.2K tokens).

**3. The token fit must use the same system message the request will carry.**
Counting against anything other than the exact string the extraction body sends
undercounts, and a profile lands just over the limit at request time — after the
expensive part.

**4. Grounding is subject-specific or it is wrong.** The failure the tuned prompt
spends most of its length on: a value supported by nearby text *about someone
else* — an employer's address read as the person's city, a colleague's phone
read as theirs. A span that resolves is not the same as a span about the
subject. Any evaluation that only checks span resolution will score this bug 100%.

**5. Name matching is not `\b`.** A bare word boundary matches after an
apostrophe, so `Malley` hits `O'Malley`. The predecessor's `mention_pattern`
carries a leading guard for exactly this; keep it if you port the matcher.

## Conventions

- Dependencies are added with `uv add`, never `pip install`.
- `uv run ruff format && uv run ruff check --fix` before anything else runs.
- Determinism: module-level `SEED = 0`, every sampler takes `random.Random(SEED)`.
  A rebuild reproduces the previous numbers exactly.
- Output filenames encode config, `profiles-{model_code}-{context}-{variant}`.
  Add a knob that changes the output and it belongs in the name.
- Output is `.jsonl.gz`, not this workspace's usual `.csv.gz`: a profile is a
  nested object carrying `_grounding` and `_sources` sidecars, and flattening it
  into columns loses the spans.
- Scripts are incremental — re-running skips outputs already on disk unless
  `--force`.
- A third corpus location means a config file, not a third environment variable.
- Comments explain *why*, not *what*.

## Tests

```sh
uv run pytest
```

Every test names a property the code must hold. Two of them are load-bearing on
a fresh machine: `--dry-run` stays green with no corpus configured, and a missing
corpus exits non-zero **before** creating the output directory. Half a profile
set is worse than none, and looks the same on disk.

## Working safely

- Generating profiles costs real money, per entity, per regime. The labels, the
  splits and the tuned prompt are cheap to lose and expensive to rebuild — treat
  them as precious.
- Large data lives outside the repo and is gitignored. Look at what is in an
  output directory before writing into it.
- `--dry-run` reports the resolved input and output paths and writes nothing.
  Use it to confirm a machine is configured before spending a run.
