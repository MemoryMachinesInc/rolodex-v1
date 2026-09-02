"""Generate grounded Rolodex profiles for a set of entities.

This is the clean rewrite of the generation + grounding path that grew inside
``z-r-eval_prompt_and_autotune/src/tune_prompt/rolodex_v1/``. That code was a
tuning adapter first and a pipeline second, so its inputs were hardcoded module
constants -- the memories bundle at ``~/dev/rolodex/test_artifacts/...`` and the
source-document cache at ``/tmp/obama_srcdocs``. A tmp path is not a place to
keep a corpus that takes real money to rebuild, and a hardcoded home directory
means only one machine can run the stage. Both locations are therefore
configuration here, and nothing else about the corpus is baked into the code.

The other thing that module conflated is *which evidence a profile is built
from*. Two regimes were run against the same entities and scored against the
same labels:

``memory``
    Each entity's aggregated memories are the context. This is what production
    does.
``chunks``
    Windows of the source documents around each mention of the entity are the
    context. This is what the agentic run and the source-doc arm of the Luna
    experiment consume, and it needs ``--source-docs``.

They produce different profiles from the same entity, so ``context`` is part of
the output filename rather than a flag whose value you have to remember.

Run it:

    uv run python -m rolodex_v1.build_profiles --help
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

# A rebuild must reproduce the previous numbers exactly, so every sampler is
# handed random.Random(SEED) rather than reaching for the global module state.
SEED = 0

# The two corpus locations are machine-specific and live outside the repo, so
# they are read from the environment when no flag is given. Anything that needs
# a third location should graduate to a config file rather than grow a third
# variable here.
ENV_MEMORIES_BUNDLE = "ROLODEX_V1_MEMORIES_BUNDLE"
ENV_SOURCE_DOCS = "ROLODEX_V1_SOURCE_DOCS"

CONTEXTS = ("memory", "chunks")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """The knobs that change the output, and therefore its filename."""

    model_code: str
    variant: str
    context: str
    memories_bundle: Path
    source_docs_dir: Path
    out_dir: Path

    def __post_init__(self) -> None:
        """Reject an unknown context here rather than at write time.

        A typo in --context would otherwise produce a plausibly-named file
        holding whichever regime the code happened to fall through to.
        """
        if self.context not in CONTEXTS:
            raise ValueError(f"context must be one of {CONTEXTS}, got {self.context!r}")

    @property
    def out_path(self) -> Path:
        """Output filenames encode the config: {type}-{model_code}-{ctx}-{variant}.

        Two runs with different settings can then coexist on disk, and the
        incremental skip below can tell which one it is looking at. The context
        regime is in the name because memory and chunk runs of the same model
        are different profiles, not two attempts at one.

        The extension is .jsonl.gz, not the workspace-default .csv.gz: a profile
        is a nested object carrying _grounding and _sources sidecars, and
        flattening that into columns loses the spans.
        """
        stem = f"profiles-{self.model_code}-{self.context}-{self.variant}"
        return self.out_dir / f"{stem}.jsonl.gz"

    @property
    def evidence_path(self) -> Path:
        """The input this context regime actually reads.

        Named once here so the missing-input error and the dry-run report
        cannot disagree about which file was expected.
        """
        if self.context == "chunks":
            return self.source_docs_dir
        return self.memories_bundle

    @property
    def evidence_env(self) -> str:
        """The environment variable that configures evidence_path."""
        return ENV_SOURCE_DOCS if self.context == "chunks" else ENV_MEMORIES_BUNDLE


def run(cfg: Config, rng: random.Random) -> None:
    """Do the work and write cfg.out_path."""
    raise NotImplementedError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-code", default="g5m", help="model code for the filename")
    p.add_argument("--variant", default="v1", help="config tag for the filename")
    p.add_argument(
        "--context",
        choices=CONTEXTS,
        default="memory",
        help="evidence each profile is built from",
    )
    p.add_argument(
        "--memories-bundle",
        type=Path,
        default=Path(os.environ.get(ENV_MEMORIES_BUNDLE, "data/memories_bundle.json")),
        help=f"entity-memory bundle; defaults to ${ENV_MEMORIES_BUNDLE}",
    )
    p.add_argument(
        "--source-docs",
        type=Path,
        default=Path(os.environ.get(ENV_SOURCE_DOCS, "data/source_docs")),
        help=f"source-document corpus; defaults to ${ENV_SOURCE_DOCS}",
    )
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--force",
        action="store_true",
        help="rewrite the output even if it already exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written and write nothing",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = Config(
        model_code=args.model_code,
        variant=args.variant,
        context=args.context,
        memories_bundle=args.memories_bundle,
        source_docs_dir=args.source_docs,
        out_dir=args.out_dir,
    )

    # Re-running skips work already on disk. This is what makes the stage safe
    # to invoke repeatedly, and --force the only way to overwrite.
    if cfg.out_path.exists() and not args.force:
        log.info("%s exists; nothing to do (--force to rewrite)", cfg.out_path)
        return 0

    if args.dry_run:
        # A dry run reports the configured corpus rather than requiring it, so
        # it stays the cheap way to check where a run would read and write on a
        # machine that has not been given the data yet.
        log.info("context %s reads %s", cfg.context, cfg.evidence_path)
        if not cfg.evidence_path.exists():
            log.info("  (not present; set $%s)", cfg.evidence_env)
        log.info("would write %s", cfg.out_path)
        return 0

    if not cfg.evidence_path.exists():
        log.error(
            "context %s needs %s, which is missing (set $%s)",
            cfg.context,
            cfg.evidence_path,
            cfg.evidence_env,
        )
        return 1

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    run(cfg, random.Random(SEED))
    log.info("wrote %s", cfg.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
