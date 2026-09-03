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

**Evidence is source documents, and only source documents.** For each entity,
±400-character windows around every mention, plus narrower ±200/260-character
windows around any `FIELD_CUES` hit sitting within 600 characters of one of
those mentions; overlapping windows merged, capped per document so one dense
briefing cannot crowd out the corpus. Windows are addressed by `grounding`'s
line-numbering rules — they are not wrapped as synthetic memories, and there is
no memory regime; it was removed rather than left as a configurable nothing
reads.

The cue pass exists because the facts a profile is scored on rarely sit inside
the same 400 characters as the name. A phone number, a `he`/`her`, a `Deputy`,
a `graduated`, a LinkedIn URL — the sentence that carries the value often names
the subject only by pronoun, several sentences downstream. So `FIELD_CUES` is
narrowed to the scored fields on purpose: a cue for an unscored field opens a
window that spends pack budget a scored one needed. It is a *recall* device and
buys nothing on precision — a cue window is still only opened near a mention,
and the text it drags in may be about somebody else entirely, which is trap 4
arriving with more surface area. The pack's job is to put the supporting
sentence in front of the model; deciding whether it is about the subject is the
prompt's, and the grounding's.

A profile is then written by a Claude Agent SDK agent that gets the pack as a
file plus read-only `Read`/`Grep`/`Glob` over the corpus, so its evidence set is
not fixed when the run starts.

## Who gets profiled

Entities come from one **resolved-entities bundle** (`--entities-bundle`): it
decides who gets a profile and what names to search for. Nothing else does —
there is no roster file and no name list. See
`src/rolodex_v1/resolved_entities.py`, whose docstring carries the schema
reasoning. Two properties drive everything downstream:

- **The id is the key, not the name.** 160 canonical names in the Obama bundle
  are used by more than one entity — two distinct `The White House`, and so on —
  so a name-keyed dict silently drops profiles.
- **Alias probability is a distribution, not a confidence.** Where several
  entities claim the same alias string, their claims sum to exactly 1.0. So
  `--min-alias-probability` above 0.5 leaves at most one owner per string, which
  is what stops one entity's evidence being retrieved under another's name. It
  is in the output filename (`p50`) because it changes which evidence a profile
  was built from.

`selectable()` drops entities left with nothing to match on — 133 of the sample
bundle's 9,180 at the 0.5 default, `BP` at 0.0427 among them. Dropping them is
correct; dropping them quietly is not, which is why `report_bundle` logs the
count on every run.

`surname_spread` is the guard no threshold provides: a real person's surface
forms share a surname, and one sample entity holds Amy Hall, Andrea Palm, Lauren
Aronson and Barbara Smith under a single name with **every alias at probability
1.0**. That is measured and warned about, not filtered — see trap 6.

## Spending money

`build_profiles` calls a model once per entity, and the corpus holds 6,203
in-scope people. **A full run is a four-figure invoice**, so an invocation with
no scope is refused: pass one of `--limit N`, `--top-mentions N`, `--entity ID`,
`--entity-list FILE`, or `--all`. `--dry-run` prices the work without committing
to it, and `--max-spend-usd` stops a run rather than an invoice.

The five are not five ways of saying the same thing, and `chosen()` ranks them:
explicit ids first (`--entity` and `--entity-list` pooled, first occurrence
wins), then `--top-mentions`, then `--limit`. A named id is always what the run
meant, so it beats a count that happens to be on the same command line rather
than intersecting with it and quietly buying fewer profiles than were asked for.

- `--entity-list FILE` is `--entity` for a set too big for shell history: one id
  per line, `#` comments and blanks skipped, so a curated cohort can record why
  each id is on it and be re-run months later against the same names. Ids that
  are unknown or out of scope are named individually before anything is bought —
  a typo and a filtered organization need different fixes, and an id that
  vanishes silently buys nine profiles where ten were asked for while the run
  reports success.
  - Cohorts worth keeping live in `entity_lists/`, which is **gitignored** for
    the same reason the machine configs are: a cohort is a list of real people
    derived from one corpus, so it belongs beside the corpus rather than in the
    repo. Record in comments how each was picked — the local
    `umb-top-people.txt` is the worked example, and the reason `--top-mentions`
    is not a substitute: its header notes that the raw top-10 by mention count
    is unusable because resolution filed a street address, a bare phone number
    and an email greeting under `participant_person`. It is also a UMB cohort,
    not Obama; the corpus provenance in this document is Obama throughout, but
    nothing in the code is.
- `--top-mentions N` takes the N most-mentioned in-scope entities, ties broken on
  entity id so the same N is picked every run. **It is the honest top-N, not a
  curated one**: resolution files mailing lists, a street address and a bare
  phone number under `participant_person`, so they rank. `--limit N` is the
  cheap smoke test instead — the first N by id, which is arbitrary but stable.

- Every profile is appended to `{out}/…​.work/checkpoints.jsonl` as it lands, so
  a crash on entity 40 does not re-buy the first 39. Deleting that file is how
  you ask for a genuinely fresh — and re-paid — run. It is one append-only log
  rather than a file per entity: the resume check is a set of ids read once, and
  a directory whose entry count is the only record of progress invites the
  half-finished-set confusion the output naming exists to prevent. A line that does not
  parse, is not an object, or carries no `entity_id` — all of which a crash
  mid-write can leave — is skipped with a warning rather than making the whole
  log unreadable and re-buying everything in it.
- **The checkpoint log carries what the artifact does not.** Each record, empty
  packs included, holds the regime, the model, the usage, the cost and
  `pack_recipe` — the frozen `PackRecipe` the evidence was built from, so two
  runs at different window or cue settings are distinguishable on disk. The
  finished artifact is profiles and nothing else, so that provenance lives only
  here: keep the `.work` directory of a run whose numbers you intend to quote.
- The agent is sandboxed by `can_use_tool`: read-only tools, and only inside the
  pack directory and the corpus. It is a hard deny, not a prompt — a batch run
  has nobody to answer one.
- `prompts/extraction_addendum.jinja` is appended to the agent's system prompt.
  It ships with the checkout and is resolved against the installed package, not
  the working directory, so a run from `src/` uses the same prompt as one from
  the root. `--dry-run` reports it alongside the corpus and **exits non-zero if
  it is missing or empty**: a broken addendum is a broken checkout rather than
  an unconfigured machine, and a dry run is worth nothing if green does not mean
  the next run can spend. `--no-addendum` and a different `--addendum` both
  change the output filename, because each is a different artifact rather than a
  cheaper one.
- It costs roughly $0.30–$1.10 per entity, which is where the printed price
  range comes from.
- Both regimes are adapters behind one `Regime` interface, `run(pack) -> Attempt`
  (`build_profiles.py`). The loop reads the `Attempt` — profile, usage, cost,
  log summary — rather than asking which regime produced it, and both adapters
  attach grounding through `grounded_profile.build_profile`, so that decision
  lives in one place. `in_context.request_profile` returns the raw response and
  its usage; it no longer attaches grounding itself.
- **`--max-spend-usd` is refused for a regime that cannot price itself.** An
  `Attempt` with `cost_usd = None` is *unpriced*, a distinct state and not a
  cheap `0.0` — the in-context endpoint reports tokens, and this repo has no
  per-model rate table to turn those into dollars. So `--max-spend-usd` with
  `--regime in-context` exits 2 before the bundle is even loaded, rather than
  accumulating zero however long the run lasts. A ceiling nobody can enforce is
  the same risk as no ceiling with a false assurance attached. Bound an
  in-context run with the scope flags (`--limit`, `--top-mentions`, `--entity`,
  `--entity-list`), which is what actually bounds it. Pricing in-context usage —
  a rate table, a new source of truth — is what would lift the refusal.
- The ceiling is checked **after** each attempt, so the run stops the moment it
  is over rather than one entity later. It can still overshoot by one profile:
  what an attempt costs is only known once it has been bought, and refusing to
  start one on an estimate would let a fabricated number decide a real run.
- **A self-hosted run is not free, it is billed differently.** `--dry-run` prints
  "GPU time, not per-request" for a Qwen run rather than a per-entity range,
  because the bill is the deployment's GPU-hours and does not divide by entity
  count. Nothing in this repo can stop that meter: the limits that apply are the
  deployment's `scaledown_window` and your own eye on the Modal dashboard, so a
  Qwen batch is the one case where walking away is expensive.

Whether the tuned addendum helps *under this regime* is unmeasured — it was
tuned against a single-request regime. That is a number to produce, not an
assumption to build on.

## What a corpus is

A corpus is a directory — `data/obama/`, `data/umb/` — holding exactly two
things. Neither is built by this repo, and a run cannot start without both.

**1. the source documents — the text.** Every `.txt` **at any depth** under the
configured directory, one document per file, plain text. The export arrives
grouped by where a document came from (`email/`, `pdf/`, `plaud/`, `text/`) and
that grouping is deliberately not carried into the evidence: a fact is as
citable from a transcript as from a message, so `corpus_documents` walks
recursively and keeps the subdirectory only in the path. Anything that is not
`.txt` is silently not evidence.

The filename stem is the document's identity in every citation the run emits, so
name them stably — renaming a file after a run detaches the profiles already
built from the text that supports them. A stem claimed by two documents in
different subdirectories would make that identity a lie, so `source_id_map`
demotes *both* claimants to their corpus-relative path rather than letting the
second silently overwrite the first in the source dict. Ids stay short in the
common case; the Obama export has no collisions among its 5,425 files.

This is the *only* evidence: there is no memory regime and no second store, so a
fact that is not in these files cannot be grounded and will not be emitted.

**2. `resolved_entities_bundle.json` — who gets profiled.** It decides both the
entity set and the surface forms retrieval searches for (see *Who gets
profiled* above). It comes from the memorymachines API, which runs resolution
over the same documents. **There is currently no way to fetch one through that
API**, so the bundles on this machine are the ones somebody handed over, a
corpus cannot be re-resolved on demand, and this file deliberately records no
endpoint. Take a bundle as it arrives and do not edit it. Its schema is documented in
`src/rolodex_v1/resolved_entities.py`; a bundle whose alias probabilities are
not a per-string distribution summing to 1.0 breaks what
`--min-alias-probability` is for, which is trap 6.

`source_docs` and `entities_bundle` are separate `[paths]` keys, so whatever
shape the export arrived in is named directly rather than rearranged on disk to
satisfy the default layout — the working Obama config puts the bundle *inside*
the document directory, which is harmless because the walk only collects `.txt`.
Setting only `data` still works for a corpus that does hang off one root.

The two must be built from the same document set. A bundle resolved over a
corpus the `source_docs/` directory does not contain yields entities with
nothing to retrieve — empty packs, checkpointed and paid for.

## Where the data is configured

Nothing in this repo hardcodes a corpus. Every location is configuration:

| | flag | env | `[paths]` key |
| --- | --- | --- | --- |
| entities bundle | `--entities-bundle` | `ROLODEX_V1_ENTITIES_BUNDLE` | `entities_bundle` |
| source documents | `--source-docs` | `ROLODEX_V1_SOURCE_DOCS` | `source_docs` |
| output directory | `--out-dir` | — | `out_dir` |

All three default under one `data/` directory, so the setting worth reaching for
is usually the root they hang off — but only the defaults assume that shape, and
an export that arrived with the bundle beside or inside it is named directly
instead. Copy `configs/rolodex-v1.toml.example` to
`configs/rolodex-v1.toml` and set one key:

```toml
[paths]
data = "/Volumes/corpus/obama"
```

The real file is gitignored — where the data sits is a property of the machine,
not the project. Discovery walks up from the working directory looking for
`configs/rolodex-v1.toml`, so one file serves runs started anywhere inside the
checkout. A bare `rolodex-v1.toml` beside the repo root is still read, because a
config that is present but unfound would fall through to `data/` and report
success against a corpus nobody chose.

Discovery covers the machine's *one* corpus. To run against a different one — a
second bundle, a scratch copy, a colleague's export — name the file instead:

```sh
uv run python -m rolodex_v1.build_profiles --dry-run --config configs/rolodex-v1.umb.toml

ROLODEX_V1_CONFIG=configs/rolodex-v1.umb.toml \
    uv run python -m rolodex_v1.build_profiles --dry-run
```

`--config` beats `$ROLODEX_V1_CONFIG`, and both beat discovery. A named file
that does not exist **exits non-zero** rather than falling through to discovery:
you asked for that file, and silently reading a corpus you did not name is the
expensive kind of surprise. `ROLODEX_V1_CONFIG=""` disables discovery outright,
which is how `tests/conftest.py` keeps the suite off whatever corpus this
machine is configured for.

A **relative path in the config resolves against the config file**, not the
working directory. That is the one behavioural difference from the bare
defaults, and it is the point: `uv run …` from `src/` reads the same corpus as
from the root. It also means the file's own location is part of its meaning —
from inside `configs/`, the repo-root `data/` is `"../data"`. This bites hardest
with `--config`: a config kept outside the checkout resolves its relative paths
against wherever *it* sits, so give an out-of-tree config absolute paths.

Precedence per location, highest first: the flag, then that location's
environment variable, then its own `[paths]` key, then `[paths].data` joined
with the built-in name, then `data/`. A machine already configured with
environment variables keeps working untouched. `--dry-run` names the config file
in effect on its first line, because a path can now arrive from a file the
command line never mentions.

Only paths are configurable. Model, context, variant and the alias threshold
stay on the command line: they are encoded in the output filename, and a config
file that quietly changed one would break the promise that the name describes
the artifact.

## Where this came from

The predecessor is `~/dev/z-r-eval_prompt_and_autotune/src/tune_prompt/rolodex_v1/`
(and its `scripts/rolodex_v1/`). It was a prompt-tuning adapter that grew a
pipeline inside it, so its corpus locations were module constants. Read it for
the algorithms — `task_model/grounding.py` and `task_model/extraction.py` are
the parts worth carrying over — not for its structure.

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

## Providers: where an in-context request goes

`--regime in-context` is one chat-completions request, and **which endpoint
receives it is chosen by `--model`**. A model id starting with `qwen` resolves to
the self-hosted provider; anything else is OpenAI. There is no second flag that
could disagree with the first.

```sh
# OpenAI, the tuned baseline
uv run python -m rolodex_v1.build_profiles --regime in-context --limit 5

# Self-hosted Qwen3.8-27B, same regime, same pack, same schema
export ROLODEX_V1_QWEN_BASE_URL=https://<workspace>--rolodex-qwen-serve.modal.run/v1
export QWEN_API_KEY=<the serving token>
uv run python -m rolodex_v1.build_profiles \
    --regime in-context --model qwen3.8-27b --effort xhigh --limit 5
```

`src/rolodex_v1/providers.py` owns the four things the two endpoints genuinely
disagree on, and its docstring carries the reasoning. Two are traps:

- **`high` is not a Qwen effort.** Qwen3.8's dial is `low`/`medium`/`xhigh`.
  `--effort` offers the union because argparse fixes choices before `--model` is
  known, so the real check is `Provider.validate_effort`, which runs before a
  pack is built. Accepting `high` for Qwen would buy a different amount of
  thinking than was asked for while the filename still claimed `high`.
- **`service_tier` is an OpenAI concept and vLLM rejects the field.** The tier
  fallback loop walks the provider's tiers; an empty tuple means one pass and no
  field.

The endpoint URL is **not** in the output filename, unlike every other knob.
Which server happens to be up is a property of the machine, not of the artifact —
the model code already says what produced the profiles. The serving cost is
quoted as "GPU time, not per-request" rather than a per-entity range, because a
self-hosted run is billed by the hour and a fabricated per-entity rate in the
pre-flight estimate would read like a measurement.

`deploy/modal_qwen.py` is the deployment, and the only thing here that knows how
to host a model. Modal is not a project dependency — the `modal` CLI brings its
own environment, so a machine that never serves a local model never installs it.
Qwen3.8's native window is 262,144 tokens and a dense entity's pack exceeds it,
so the deployment applies the YaRN rope override and serves at 960,000.
`MAX_MODEL_LEN` there and `providers.QWEN_MAX_MODEL_LEN` here must stay in step
or the client fits requests against a ceiling the server does not have;
`tests/test_providers.py` fails if they drift.

## Schema parity

Every regime must emit the same `EntityBio`/`Biography`. Inside this repo there
is one `src/rolodex_v1/profile_schema.py` and everything that produces a profile
imports it, so a regime cannot quietly grow its own shape. Both regimes are
wired and both import it, rather than validating against a local shape.

**Prompt text is under the same rule**, for the same reason and one more: the
regimes are scored against *each other*, so wording that drifts makes the
comparison invalid rather than merely untidy. `src/rolodex_v1/prompt_rules.py`
holds what is word-for-word common — the owner-context cascade and the shared
base rules — and each regime keeps only what genuinely differs (the agentic
prompt describes a pack file and tools; the in-context one describes numbered
source lines). Those divergences are deliberate and marked at the seam. Both
regimes pin their rendered prompt with a digest test, so an accidental reword
fails rather than quietly rebasing the numbers.

The provider split does not touch this. A self-hosted endpoint sends the same
strict `json_schema` with the same `GroundedBiographyResponse`, and
`tests/test_providers.py` asserts it: which server answered is not licence to
emit a different artifact. What a provider may vary is the envelope around the
schema, never the schema.

What is *not* structural is parity with the tuned Luna/Sol runs, whose scores
are the baseline every new number is compared against. Those ran in
`z-r-eval_prompt_and_autotune` against their own copy of the schema. Their
Both attach with `require_complete=False`, and the schema is pinned by
`tests/fixtures/tuned_runs_grounded_schema.json`, a verbatim snapshot of what
those runs sent — `test_profile_schema.py` compares descriptions and all, so a
reworded docstring fails the suite.

**The citation contract, though, is no longer pinned.** This file used to argue
that `grounding.py` was byte-identical to the predecessor's, which is what made
the spans comparable. It has since diverged, and not cosmetically: the predecessor
merged only *overlapping* line ranges (`start <= end`), where this copy also
merges *abutting* ones (`start <= end + 1`), so a citation of lines 1-2 and one
of lines 3-4 arrive as a single quote here and as two there. Resolution also
splits runs on non-contiguous source text, and an unresolvable line is tolerated
rather than raised. Those change
which selectors come out of the same cited ranges, so the comparison to the
tuned Luna/Sol scores rests on nothing measured. Re-establish it against the
predecessor's output or stop quoting those scores as a baseline — this is a
number to produce, not an assumption to build on.

**A docstring on a schema class is prompt text.** Pydantic emits it as the JSON
Schema's `description`, which the model reads. `Biography`'s docstring carries
the null/`[]` contract and is the tuned wording verbatim but for one clause
(theirs named OpenAI, which is false under the agentic regime);
`GroundedBiographyResponse` deliberately has none, because the tuned runs sent
none. Maintainer notes go in comments above the class, where they cannot reach
the model. `tests/test_profile_schema.py` fails on a changed field, a loosened
type, or a reworded description.

If that test fails, a profile built today is not the same artifact as one that
was scored, and the comparison is invalid until somebody decides which side is
right — regenerate the fixture only when that decision has been made.

## Traps

Nothing has shipped from this repo yet, so these are inherited — each one
already cost someone a run in the predecessor.

**1. The context budget is measured on the rendered request, not the excerpts.**
`build_grounding_context` adds a per-window header and re-wraps every line to 200
characters with a line-number prefix. That inflated one ~700K-token excerpt set
into a **1,168,995-token** request. Fit the budget against the rendered messages.

**2. The deployed input limit is lower than the documented one.** The
predecessor's ceiling derived from a 1,050,000-token context window, but a
922,518-token request came back rejected with "Input tokens exceed the
configured limit of 922000 tokens". Budget against 922,000, and leave headroom
for the per-run prompt addendum — it is not free and is not known at fit time
(`prompts/extraction_addendum.jinja` is ~2.2K tokens). That number is
`providers.OPENAI_OBSERVED_INPUT_LIMIT`, and the budget derived from it is
`providers.OPENAI.max_input_tokens`.

**The ceiling is per-provider, not global.** A self-hosted server's is whatever
`--max-model-len` it was launched with. Fitting a Qwen request against OpenAI's
number would refuse requests that would have fit, and fitting an OpenAI request
against Qwen's would spend one to be rejected. `chat_completion_body` asks the
provider it was handed; nothing reads a module-level constant.

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
`resolved_entities._LEADING_GUARD` is that guard, and every pattern this repo
builds goes through `resolved_entities.form_pattern`.

**6. A surname is not an identification.** Matching bare surnames is how the
handoff reported the same 692 documents for Mike Allen and Jessica Allen. On
this corpus it was worse: bare `Obama` put 1,656 of 3,665 documents into
*Michelle* Obama's evidence, and bare `Malley` gave Robert Malley 174 documents
of which 168 were Martin O'Malley's.

There is no surname pass any more. `select_windows` matches only the entity's
resolved surface forms and emails, every one of which identifies, and ambiguity
is settled upstream by probability: a string two entities claim splits its mass
between them, so the threshold picks an owner before retrieval runs. Only 29 of
1,301 single-token forms survive under more than one entity, and all of them are
organizations, which are out of scope by default.

The way back in is the threshold, not the matcher. `--min-alias-probability 0`
admits every contested string to every claimant, which is the `Malley`/`O'Malley`
collision arriving through a different door.

**7. Minimizing surface forms is not substring containment.** Dropping a form
because a shorter kept form is lexically inside it looks equivalent and is not:
the bundle holds the truncation artifact `axelro` beside `David Axelrod`, and
`axelro` is inside `david axelrod`, so the entity was left searching for a
fragment that matches nothing. That silently destroyed the usable forms of 216
entities. `surface_forms` tests with the *same word-boundary pattern retrieval
uses*, under which `\baxelro\b` does not match `Axelrod`. Keep the two rules
the same or this returns.

**8. Two providers, two effort vocabularies.** `high` is the top of OpenAI's
range and does not exist for Qwen3.8, whose top is `xhigh`. A run that sends the
wrong one either errors at the far end or, worse, is silently accepted at a
different thinking budget than the filename records. The provider validates it;
do not move that check into the request builder, where it would fire after the
pack is built.

**9. A crashed server on Modal looks exactly like a slow one.** The predecessor's
deployment built its command with `Popen(" ".join(cmd), shell=True)`.
`--hf-overrides` carries JSON containing spaces, so the shell word-split it and
stripped its quotes; vLLM exited immediately with `unrecognized arguments`.
Modal then sat waiting for port 8000 until `startup_timeout`, and from the
client every request simply hung — indistinguishable from a 30GB cold start,
for 45 minutes. `deploy/modal_qwen.py` passes a list to `Popen` with no shell,
echoes it with `shlex.join` so the logged line is the line that ran, and
`tests/test_providers.py` asserts both. When a deployment seems slow to boot,
read `modal app logs` before waiting on it.

## Conventions

- Dependencies are added with `uv add`, never `pip install`.
- The Claude Agent SDK spawns the `claude` CLI as a subprocess, so a
  pip-only environment fails at *runtime*, not at install time. Node ≥18 and
  `@anthropic-ai/claude-code` must be present. Only a real run needs them:
  `agentic` imports the SDK lazily, so `--dry-run` and the whole test suite run
  without it.
- `uv run ruff format && uv run ruff check --fix` before anything else runs.
- Determinism: module-level `SEED = 0`, every sampler takes `random.Random(SEED)`.
  A rebuild reproduces the previous numbers exactly.
- Output filenames encode config,
  `profiles-{run_code}-{regime}-{threshold}-{addendum}-{variant}`, where
  `run_code` is the model code plus the reasoning effort for `in-context`, and
  `{addendum}` is `noaddendum` when suppressed or `ad{8 hex}` of the addendum's
  **content** otherwise. Add a knob that changes the output and it belongs in
  the name — the alias threshold is there because it changes which evidence a
  profile was built from, and the addendum digest is content-keyed rather than
  path-keyed because renaming the prompt must not fork the artifact while
  editing it must.
  - The pack recipe is the deliberate exception (see above), and it is now the
    expensive one: it is in neither the name nor the artifact, only in the run's
    `.work/checkpoints.jsonl`. So two runs whose recipes differ collide on one
    path, the second silently loses to the incremental skip, and deleting the
    `.work` directory of the survivor leaves a file nothing on disk explains.
    Change a pack knob, change `--out-dir` with it.
  The converse is why `--qwen-base-url` is absent: two runs of the same model at
  the same effort are the same artifact whichever host answered, and putting the
  URL in the name would split one result across two files.
- Output is **one `.json` object, entity id to profile** — not this workspace's
  usual `.csv.gz`, because a profile is a nested object carrying `_grounding`
  and `_sources` sidecars and flattening it into columns loses the spans; and
  not one file per entity, because the artifact is the set and "what did this
  run produce for X" should be a key lookup rather than a directory scan. Keyed
  by entity id for the reason the rest of the pipeline is: 160 canonical names
  in the sample bundle belong to more than one entity, so a name-keyed map
  silently drops profiles. An entity whose pack held no evidence is checkpointed
  but is **not** a key — an id mapping to null would read as a profile that came
  back empty, which is a different and more alarming thing than one nobody
  bought.
- Scripts are incremental — re-running skips outputs already on disk unless
  `--force`.
- A new corpus **location** is a key in `settings.LAYOUT`, not a new environment
  variable. Listing it there gives it a flag-beats-env-beats-config resolution,
  a `[paths]` key, relocation with the root, and a line in `--dry-run` — the
  three `ROLODEX_V1_*` path variables predate the config file and are kept only
  so a configured machine keeps working.
- `ROLODEX_V1_QWEN_BASE_URL` is the one `ROLODEX_V1_*` variable that is **not** a
  path, and it is deliberately not in `settings.LAYOUT`: it is a URL, so it does
  not relocate with the data root, has no `[paths]` key, and resolving it against
  a config file's directory — which is what `LAYOUT` does to a relative value —
  would be nonsense. It keeps the flag-beats-env half of the precedence rule
  (`--qwen-base-url` wins), which is the half that applies. Do not "fix" it into
  `LAYOUT`.
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
