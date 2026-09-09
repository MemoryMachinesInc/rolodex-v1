# rolodex-v1

Generates **grounded** Rolodex profiles. For each person in a corpus, a model
writes a biography in which every emitted value carries a span pointing back at
the text that supports it. A run produces one JSON file mapping entity id to
profile.

A run needs two inputs: a directory of `.txt` source documents, and a
`resolved_entities_bundle.json` that says who gets profiled. You can point the
tool at an export somebody handed you, or download both from the memorymachines
API.

> **This costs real money.** Profiles are bought one model call at a time, at
> roughly $0.30–$1.10 per person. The sample corpus holds 6,203 people, so a
> full run is a four-figure invoice. Two things guard against that: `--dry-run`
> prices any command without spending, and a real run refuses to start unless
> you have said how many people to profile.

## How to Set Up

### Step 1: Install uv

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and its
dependencies. You do not need to install Python separately; uv fetches the
right version (3.13 or newer).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Check out the sibling repository

This project depends on `z-r-research_memorome` for minting the API token used
by the corpus downloader. It is published to no package index, so the path is
the only way to install it: **both repos must sit side by side under the same
parent directory.**

```text
~/[parent directory]/
├── rolodex-v1/            # this repo
└── z-r-research_memorome/ # must be here, or `uv sync` fails
```

**Check out the `firebase-token-importable` branch**, not `main`. That branch
is what packages the token client so another project can install it; `main`
does not have it yet. It is [PR #11][pr11], and once that merges, plain `main`
will work and the `--branch` flag can go.

```bash
# from the parent directory that holds rolodex-v1
git clone --branch firebase-token-importable \
    https://github.com/MemoryMachinesInc/z-r-research_memorome.git
```

Already cloned it? Switch the existing checkout over:

```bash
cd ../z-r-research_memorome
git fetch origin firebase-token-importable
git checkout firebase-token-importable
```

[pr11]: https://github.com/MemoryMachinesInc/z-r-research_memorome/pull/11

### Step 3: Install dependencies

```bash
cd rolodex-v1
uv sync
```

### Step 4: Configure where your data lives

This is the step that matters most. Start by copying the template, which is
gitignored because where your data sits is a property of your machine:

```bash
cp configs/rolodex-v1.toml.example configs/rolodex-v1.toml
```

#### The owner directory

A corpus belongs to one person — the **owner** whose documents these are. All
of that owner's data lives in a single parent folder, and **that folder is
normally the only path you have to configure**:

```toml
[paths]
data = "../data/blake"
```

Everything else is derived from it. Lay the folder out like this:

```text
data/blake/                        ←  the owner directory  ([paths].data)
├── resolved_entities_bundle.json  ←  input: who gets profiled
├── source_docs/                   ←  input: the evidence
│   ├── email/
│   ├── pdf/
│   ├── plaud/
│   └── text/
├── profiles-opus5-agentic-p50-adad5978bb-v1.json    ←  output
└── profiles-opus5-agentic-p50-adad5978bb-v1.work/   ←  checkpoints
```

Four things to know about that layout:

- **The bundle must have exactly that filename**, sitting directly in the
  owner directory.
- **`source_docs/` is searched recursively for `*.txt`.** The subfolder names
  are just how the export happened to group things and carry no meaning, so
  leave them as they arrived. Anything that is not a `.txt` file is ignored.
- **Results are written back into the same directory**, beside the inputs.
- **Both inputs must come from the same export.** A bundle built over
  different documents produces people whose names appear nowhere in the text,
  which looks like a working run that profiles nobody.

#### One directory per owner

Keep owners side by side and give each one its own config file:

```text
data/
├── blake/
├── obama/
└── umb/
```

```bash
uv run python -m rolodex_v1.build_profiles \
    --dry-run --config configs/rolodex-v1.blake.toml
```

Without `--config`, the tool walks up from your working directory looking for
`configs/rolodex-v1.toml`. The flag beats the `$ROLODEX_V1_CONFIG` variable,
and both beat that search.

The owner directory only names the corpus; it does not tell the model whose
Rolodex it is. Pass `--profile-owner "Blake Moody"` for that, and every
profile's `relationship_to_user` is written relative to that person.

#### If your export is not in that shape

Name the locations individually. Any you leave out still default under `data`:

```toml
[paths]
source_docs     = "/Volumes/corpus/obama"
entities_bundle = "/Volumes/corpus/obama/resolved_entities_bundle.json"
out_dir         = "/Volumes/corpus/obama"
```

Relative paths resolve against **the config file's own directory**, not your
working directory. Give absolute paths to a config kept outside the checkout.

If you do not have a corpus yet, point `data` at where you want the download
to land and continue to the download section below.

### Step 5: Add your API keys

Copy the template and fill in the keys for what you actually intend to run.
The file is gitignored.

```bash
cp .env.local.example .env.local
```

| Variable | Needed for |
| --- | --- |
| `OPENAI_API_KEY_SMALL` | `--regime in-context` with an OpenAI model |
| `QWEN_API_KEY` | `--regime in-context` with a self-hosted Qwen model |

`OPENAI_API_KEY_SMALL` falls back to `OPENAI_API_KEY` if unset. Environment
variables win over the file, and each value is read only when the command that
needs it actually runs, so you never need keys for a path you are not using.

Two things are **not** configured here. The agentic regime authenticates
through the Claude CLI (Step 6), so run `claude` once and log in. And
downloading a corpus uses your Engramme sign-in, covered in its own section
below.

### Step 6: Install Node (only for the agentic regime)

The default regime runs a Claude agent, which the SDK launches as a
command-line subprocess. If you plan to use it, install Node 18 or newer and
the Claude CLI:

```bash
npm install -g @anthropic-ai/claude-code
```

Skip this if you will only use `--regime in-context`. Nothing here is needed
for `--dry-run` or for the test suite.

### Step 7: Confirm the setup

```bash
uv run python -m rolodex_v1.build_profiles --dry-run --limit 3
```

This writes nothing and costs nothing. It reports the config file in effect,
whether each input was found, and where output would go:

```text
config   /Users/you/dev/rolodex-v1/configs/rolodex-v1.toml; data root .../data
ok       .../data/obama/resolved_entities_bundle.json
ok       .../data/obama
ok       .../prompts/extraction_addendum.jinja (ships with this repo)
would profile 3 entities via agentic, about $0.90-$3.30
would write .../data/profiles-opus5-agentic-p50-adad5978bb-v1.json
```

Any line starting with `missing` means that input was not found. Fix the path
in your config and run it again before spending anything.

## How to Download a Corpus (Optional)

Skip this section if you already have source documents and a bundle.

### Add a `[fetch]` table

Downloading is refused unless your config asks for it explicitly, so that no
run can guess its way into asking a production API for somebody's documents.
Add this to `configs/rolodex-v1.toml`:

```toml
[fetch]
environment = "prod"   # required: prod, staging or dev
```

`environment` has no default, because it decides whose API is asked. Use
`prod` unless you have been told otherwise. You can also set `sources` to limit
which document types are pulled; the default is all of them.

### Sign in to Engramme

The download authenticates as you, against the Engramme API. **The credential
comes from your macOS keychain, where the Engramme desktop app puts it when you
sign in.** So the setup is just: install the Engramme desktop app and log in.
The tool finds the credential on its own — there is nothing to copy or paste.

The first time it reads the keychain, macOS asks your permission. Click
**Always Allow** so later runs are not interrupted.

If you are not on a Mac, or you would rather not use the keychain, put the
token in `.env.local` instead and the tool will prefer it:

```bash
MM_REFRESH_TOKEN="your-engramme-refresh-token"
```

### Run the download

```bash
# what it would pull, and where it would land
uv run python -m rolodex_v1.fetch_corpus --dry-run

# the bundle and the documents
uv run python -m rolodex_v1.fetch_corpus
```

Both halves are incremental: documents already on disk are skipped, so an
interrupted download resumes rather than starting over. An existing bundle is
left alone unless you pass `--force`.

Useful variations (very optional):

```bash
# just one half
uv run python -m rolodex_v1.fetch_corpus --only bundle
uv run python -m rolodex_v1.fetch_corpus --only source-docs

# convert a directory an earlier shell dump already downloaded
uv run python -m rolodex_v1.fetch_corpus --from-dump ./source_docs_dump
```

If any document cannot be read, it is named in the log and the command exits
non-zero, so an incomplete corpus never passes for a finished one.

## How to Build Profiles

### Choose a regime (`--regime`)

There are two ways to write a profile. Both emit the same schema, start from
the same evidence pack, and attach the same grounding. What differs is what
the model can do while writing.

| | `--regime agentic` (default) | `--regime in-context` |
| --- | --- | --- |
| How it works | a Claude agent gets the pack as a file, plus read-only access to the corpus | one request carrying the whole pack |
| Evidence | can search the corpus for more | fixed when the request is sent |
| Rough cost | $0.30–$1.10 per person | $0.03–$0.15 per person |
| `--max-spend-usd` | works | refused; it reports tokens, not dollars |

**Agentic** is the default and the more thorough of the two. That live corpus
access is the entire difference: the agent can confirm what the pack shows, or
go find the sentence the pack missed. It is sandboxed while it does so — every
tool except `Read`, `Grep` and `Glob` is denied outright, and it cannot look
outside the pack directory and the corpus.

**In-context** is around ten times cheaper and easier to reason about, because
the evidence is exactly the pack and nothing more. Use it for wide runs, for
comparing prompts, or when you want a self-hosted model, which is the one thing
only this regime can do.

### Choose a model (`--model`)

You can leave this alone. Each regime has a sensible default:

| Regime | Default model |
| --- | --- |
| `agentic` | `claude-opus-5` |
| `in-context` | `gpt-5.6-luna` |

Any model your key can reach will work. The model's short code goes into the
output filename — `claude-opus-5` becomes `opus5` — so two runs on different
models never overwrite each other.

**The model id also decides which server receives the request.** An id
starting with `qwen` goes to a self-hosted deployment; anything else goes to
OpenAI. There is no second flag that could disagree with it.

To use a self-hosted model, deploy one with `modal deploy
deploy/modal_qwen.py`, then point the run at it:

```bash
export ROLODEX_V1_QWEN_BASE_URL=https://<workspace>--rolodex-qwen-serve.modal.run/v1
export QWEN_API_KEY=<the serving token>

uv run python -m rolodex_v1.build_profiles \
    --regime in-context --model qwen3.8-27b --effort xhigh --limit 5
```

The URL can also come from `--qwen-base-url`. It is deliberately not part of
the output filename: which server happened to be up is a property of your
machine, not of the profiles. Note that a self-hosted run is billed as GPU time
by the hour, so nothing here can cap it — watch the Modal dashboard and shut
the deployment down when you are done.

### Set the reasoning effort (`--effort`)

How much thinking to buy per profile. It applies only to `--regime in-context`;
the agentic regime has no such dial and rejects the flag.

The accepted values depend on the model, because the two providers use
different vocabularies:

| Provider | Accepted | Default |
| --- | --- | --- |
| OpenAI | `low`, `medium`, `high` | `high` |
| Qwen | `low`, `medium`, `xhigh` | `xhigh` |

`high` is not a Qwen setting and `xhigh` is not an OpenAI one. Passing the
wrong one is rejected up front, before any evidence is built or anything is
spent. Effort is part of the output filename too, since the same model at two
efforts produces two different sets of profiles.

### Choose how many people to profile

Because this stage spends money per person, **a real run with no scope is
refused.** A dry run without one still works; it just reports the whole corpus
and what profiling all of it would cost. Pick exactly one:

| Flag | What it does |
| --- | --- |
| `--limit N` | the first N entities by id — the cheap smoke test |
| `--top-mentions N` | the N most-mentioned entities |
| `--entity ID` | one specific entity; repeatable |
| `--entity-list FILE` | ids listed in a file, one per line |
| `--all` | everyone in scope — the four-figure run |

Always dry-run first. It prints an estimated price range for the scope you
picked:

```bash
uv run python -m rolodex_v1.build_profiles --dry-run --limit 5
```

### Run it

```bash
# the default: a sandboxed Claude agent that can search the corpus itself
uv run python -m rolodex_v1.build_profiles --regime agentic --limit 5

# one OpenAI request carrying the whole evidence pack
uv run python -m rolodex_v1.build_profiles --regime in-context --limit 5
```

Add a hard ceiling to any agentic run. It is checked after each profile, so the
run stops as soon as the total is exceeded:

```bash
uv run python -m rolodex_v1.build_profiles \
    --regime agentic --all --max-spend-usd 50
```

`--max-spend-usd` is rejected for `--regime in-context`, which reports tokens
rather than dollars. Bound those runs with the scope flags instead.

### Where the output goes

One JSON file in your output directory, mapping entity id to profile:

```text
profiles-opus5-agentic-p50-adad5978bb-v1.json
         │      │       │   │          └── variant tag
         │      │       │   └── digest of the prompt addendum
         │      │       └── alias probability threshold
         │      └── regime
         └── model
```

An in-context run carries its effort in the model slot as well, since the same
model at two efforts is two different results:

```text
profiles-gpt56luna-high-in-context-p50-adad5978bb-v1.json
```

The name encodes the settings, so two runs configured differently never
overwrite each other. Re-running skips a file that already exists unless you
pass `--force`.

### Resuming an interrupted run

Every profile is checkpointed as it lands, in a `.work` directory beside the
output. A crash on person 40 does not re-buy the first 39 — just run the same
command again.

To force a genuinely fresh, re-paid run, delete that `.work` directory. Keep it
for any run whose numbers you intend to quote: it records the model, the cost
and the exact evidence settings behind each profile, none of which appear in
the finished file.

## Running the Tests

```bash
uv run pytest
```

The suite needs no corpus, no API keys and no network.

## Project Structure

```text
rolodex-v1/
├── configs/
│   └── rolodex-v1.toml.example   # copy to rolodex-v1.toml and edit
├── deploy/
│   └── modal_qwen.py             # optional self-hosted model deployment
├── prompts/
│   └── extraction_addendum.jinja # the tuned prompt; ships with the checkout
├── src/rolodex_v1/
│   ├── build_profiles.py         # the profile builder (main entry point)
│   ├── fetch_corpus.py           # the corpus downloader (optional step)
│   ├── memorymachines.py         # API client the downloader drives
│   ├── settings.py               # where the data lives, resolved
│   ├── resolved_entities.py      # reads the bundle: who gets profiled
│   ├── evidence_pack.py          # builds each person's evidence
│   ├── grounding.py              # spans back to the source text
│   ├── profile_schema.py         # the one profile shape every regime emits
│   ├── agentic.py                # regime: a sandboxed Claude agent
│   ├── in_context.py             # regime: one request with the whole pack
│   └── providers.py              # OpenAI vs self-hosted differences
├── tests/
├── .env.local.example            # copy to .env.local and add keys
└── pyproject.toml
```

## Troubleshooting

**`uv sync` cannot find or build `z-r-research-memorome`.** Either the sibling
checkout is missing, or it is on the wrong branch. Both repos must share a
parent directory, and the sibling must be on `firebase-token-importable` —
`main` is not installable as a package yet. See Step 2.

**A dry run says `missing` for an input.** The path in your config is wrong.
Remember that relative paths resolve against the config file's directory, not
your shell's.

**A dry run reads the wrong corpus.** Config discovery walks up from your
working directory. Name one explicitly to be sure:

```bash
uv run python -m rolodex_v1.build_profiles --dry-run --config path/to/config.toml
```

**A run exits immediately asking for scope.** That is deliberate, and it only
happens on a real run. Pass one of `--limit`, `--top-mentions`, `--entity`,
`--entity-list` or `--all`.

**A download hangs with no output on macOS.** The keychain is waiting on a
consent dialog this process cannot show. The command times out after 20 seconds
and tells you how to approve it once, or set `MM_REFRESH_TOKEN` to skip the
keychain entirely.

**Profiles come back empty.** The bundle and the documents were probably built
from different corpora, so the names being searched for do not appear in the
text. Both inputs must come from the same export.

## Further Reading

[AGENTS.md](AGENTS.md) is the design document: how evidence is selected, how
the two regimes compare, how entities are chosen, and the traps that have
already cost someone a run.
