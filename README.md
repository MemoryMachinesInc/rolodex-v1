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
~/dev/
├── rolodex-v1/            # this repo
└── z-r-research_memorome/ # must be here, or `uv sync` fails
```

```bash
# from the parent directory that holds rolodex-v1
git clone https://github.com/MemoryMachinesInc/z-r-research_memorome.git
```

### Step 3: Install dependencies

```bash
cd rolodex-v1
uv sync
```

### Step 4: Install Node (only for the agentic regime)

The default regime runs a Claude agent, which the SDK launches as a
command-line subprocess. If you plan to use it, install Node 18 or newer and
the Claude CLI:

```bash
npm install -g @anthropic-ai/claude-code
```

Skip this if you will only use `--regime in-context`. Nothing here is needed
for `--dry-run` or for the test suite.

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
| `MM_REFRESH_TOKEN` | downloading a corpus with `fetch_corpus` |

`OPENAI_API_KEY_SMALL` falls back to `OPENAI_API_KEY` if unset. Environment
variables win over the file, and each value is read only when the command that
needs it actually runs, so you never need keys for a path you are not using.

The agentic regime is the exception: it needs no key here, because it
authenticates through the Claude CLI you installed in Step 4. Run `claude`
once and log in, or set `ANTHROPIC_API_KEY` for the CLI's own benefit. This
project never reads that variable.

`MM_REFRESH_TOKEN` is optional even for downloads: if it is unset, the tool
looks in `~/.engramme/engramme_refresh_token.txt` and then in the macOS
keychain item the Engramme desktop app writes.

### Step 6: Point the checkout at a corpus

Copy the config template. This file is gitignored, because where your data
sits is a property of your machine.

```bash
cp configs/rolodex-v1.toml.example configs/rolodex-v1.toml
```

If both inputs live under one directory, set a single key:

```toml
[paths]
data = "/Volumes/corpus/obama"
```

If they arrived separately, which is the usual case, name them individually:

```toml
[paths]
source_docs     = "/Volumes/corpus/obama"
entities_bundle = "/Volumes/corpus/obama/resolved_entities_bundle.json"
```

Relative paths resolve against **the config file's own directory**, not your
working directory. Give absolute paths to a config kept outside the checkout.

If you do not have a corpus yet, leave the paths pointing where you want the
download to land and continue to the next section.

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

`environment` has no default because it decides whose API is asked. It also
selects the Firebase project your refresh token must have been minted against,
so a credential does not carry between environments.

Optional keys: `base_url` overrides the environment's API base, `sources`
limits which document types are pulled (the default is all of them), and
`refresh_token_env` names a different variable to read the credential from.
No secret ever goes in this file — it names the variable, not the value.

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

Useful variations:

```bash
# just one half
uv run python -m rolodex_v1.fetch_corpus --only bundle
uv run python -m rolodex_v1.fetch_corpus --only source-docs

# convert a directory an earlier shell dump already downloaded
uv run python -m rolodex_v1.fetch_corpus --from-dump ./source_docs_dump
```

Documents arrive as JSON and are written as `.txt` as they land, so an
interrupted download leaves usable evidence rather than a directory the profile
builder reads as an empty corpus. If any document is refused, it is named in
the log and the command exits non-zero, so a corpus quietly missing documents
cannot pass for a complete one.

## How to Build Profiles

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

### Using a self-hosted model (optional)

A model id starting with `qwen` sends the request to your own server instead of
OpenAI. Deploy one with `modal deploy deploy/modal_qwen.py`.

```bash
export ROLODEX_V1_QWEN_BASE_URL=https://<workspace>--rolodex-qwen-serve.modal.run/v1
export QWEN_API_KEY=<the serving token>

uv run python -m rolodex_v1.build_profiles \
    --regime in-context --model qwen3.8-27b --effort xhigh --limit 5
```

Self-hosted runs are billed by GPU time rather than per request, so nothing in
this repo can cap them. Watch the Modal dashboard and shut the deployment down
when you are finished.

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

**`uv sync` fails to find `z-r-research-memorome`.** The sibling checkout is
missing. See Step 2 — both repos must share a parent directory.

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
