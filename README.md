# rolodex-v1

Grounded Rolodex profile generation. Every emitted primitive value carries a
span pointing back at the text that supports it. A run writes one JSON file
mapping entity id to profile.

```sh
# Where a run would read and write, on this machine. Writes nothing, costs
# nothing, and is the only end-to-end path today.
uv run python -m rolodex_v1.build_profiles --dry-run
```

Three ways to write a profile, one schema and one grounding contract between
them:

```sh
# A sandboxed Claude agent with live corpus access
uv run python -m rolodex_v1.build_profiles --regime agentic --limit 5

# One OpenAI request holding the whole evidence pack
uv run python -m rolodex_v1.build_profiles --regime in-context --limit 5

# The same request, sent to a self-hosted Qwen3.8-27B
export ROLODEX_V1_QWEN_BASE_URL=https://<workspace>--rolodex-qwen-serve.modal.run/v1
export QWEN_API_KEY=<the serving token>
uv run python -m rolodex_v1.build_profiles \
    --regime in-context --model qwen3.8-27b --effort xhigh --limit 5
```

`--model` alone picks the endpoint: an id starting with `qwen` goes to the
self-hosted server, anything else to OpenAI. Serve one with
`modal deploy deploy/modal_qwen.py`.

A corpus directory holds exactly two things, and a run needs both:

```text
data/obama/
  email/  pdf/  plaud/  text/     # source documents; walked recursively
  resolved_entities_bundle.json   # who gets profiled, and what names to search for
```

The source documents are the only evidence: every `*.txt` at any depth under the
configured directory, other extensions ignored. Filenames are the document
identity in every citation, so keep them stable. The two locations are separate
`[paths]` keys, so neither has to live inside the other.

`resolved_entities_bundle.json` comes from the memorymachines API, resolved over
that same document set. **There is currently no way to retrieve one through that
API**, so a corpus is only as new as the last bundle somebody handed over; the
endpoint is deliberately not documented here. The bundle is used as returned.

Point the checkout at a corpus by copying `configs/rolodex-v1.toml.example`
to `configs/rolodex-v1.toml`. One key moves everything together:

```toml
[paths]
data = "/Volumes/corpus/obama"
```

Name the two inputs individually when the export did not arrive in the default
shape — which is the usual case:

```toml
[paths]
source_docs     = "../data/obama"
entities_bundle = "../data/obama/resolved_entities_bundle.json"
```

That file is found by walking up from the working directory. To run against a
different corpus, name a config explicitly — the flag wins over the variable,
and both win over discovery:

```sh
uv run python -m rolodex_v1.build_profiles --dry-run --config configs/rolodex-v1.umb.toml

ROLODEX_V1_CONFIG=configs/rolodex-v1.umb.toml \
    uv run python -m rolodex_v1.build_profiles --dry-run
```

`--dry-run` names the config that won on its first line. A named file that does
not exist exits non-zero instead of falling back to discovery. Relative paths
inside a config resolve against **that file's** directory, so give a config kept
outside the checkout absolute paths.

See [AGENTS.md](AGENTS.md) for the context regimes, the provider split, how
entities are selected, the corpus configuration, and the traps.
