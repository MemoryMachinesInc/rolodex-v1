"""Generate grounded Rolodex profiles for the entities in a resolved bundle.

This is the clean rewrite of the generation + grounding path that grew inside
``z-r-eval_prompt_and_autotune/src/tune_prompt/rolodex_v1/``. That code was a
tuning adapter first and a pipeline second, so its inputs were hardcoded module
constants -- a source-document cache at ``/tmp/obama_srcdocs``, and a three-file
chain for entity aliases of which one lived in a directory named
``merges_tmp_workspace``. A tmp path is not a place to keep a corpus that costs
real money to rebuild, and a hardcoded home directory means only one machine can
run the stage. Every location is configuration here; see :mod:`settings`.

The deeper change is *which entities can be profiled at all*. The predecessor
started from a human-annotated label set and resolved each label through
annotations, merges and a summary bundle -- so an entity nobody had annotated
had no aliases, and a lookup that missed degraded silently to the bare label.
Entities now come from one resolved-entities bundle covering the whole corpus,
and nothing on this path asks whether a resolution was human-made or
machine-made. See :mod:`rolodex_v1.resolved_entities`.

Evidence is source documents, and only source documents. The memory regime is
gone rather than left as an unread configurable.

**This stage spends money, per entity, and the corpus has thousands of them.**
So it refuses to run without an explicit scope: ``--limit``,
``--top-mentions``, ``--entity``, ``--entity-list``, or ``--all``. An unscoped
invocation prices the work and exits non-zero. The
same reasoning drives the rest of the safety here -- a per-entity checkpoint so
a crash does not re-buy what already landed, and a spend ceiling that stops the
run rather than the invoice.

Run it:

    uv run python -m rolodex_v1.build_profiles --help
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rolodex_v1 import agentic, grounded_profile, in_context, providers, settings
from rolodex_v1.evidence_pack import EvidencePack, build_pack
from rolodex_v1.resolved_entities import (
    DEFAULT_MIN_PROBABILITY,
    ResolvedEntity,
    load_bundle,
)

# A rebuild must reproduce the previous numbers exactly, so any sampler added
# here takes random.Random(SEED) rather than reaching for global module state.
# Nothing on this path samples today; the constant records the convention.
SEED = 0

# A person's surface forms share a surname. Above this many distinct ones the
# entity is likely several people merged into one, and its evidence would be
# retrieved as though it were a single subject.
SUSPECT_SURNAME_SPREAD = 3

REGIMES = ("agentic", "in-context")

# Observed on the predecessor's runs, and the reason nothing here runs unscoped.
# The in-context regime is one request rather than a whole agent session, so it
# is roughly an order of magnitude cheaper -- but it sees only the pack, where
# the agent can go back to the corpus.
# The in-context rate is the provider's, not a copy: providers.py owns each
# endpoint's numbers, and a duplicate here would go stale silently.
AGENTIC_COST_PER_ENTITY_USD = (0.30, 1.10)

# The tuned addendum ships with this checkout, not with the corpus, so it is
# resolved against the installed package rather than the working directory: a
# CWD-relative default made `--dry-run` report a ready machine from src/ and
# then die in load_addendum after the expensive part had started.
PACKAGE_DIR = Path(__file__).resolve().parent
ADDENDUM_NAME = "extraction_addendum.jinja"


def _default_addendum() -> Path:
    """Where the tuned prompt sits, whatever directory the run started in.

    `prompts/` lives at the repo root today (the wheel packages only
    `src/rolodex_v1`), so the checkout layout is the first candidate; a copy
    inside the package wins if one is ever shipped as package data. Falling
    back to the checkout path rather than to None keeps the error message
    pointing at where the file is supposed to be.
    """
    candidates = (
        PACKAGE_DIR / "prompts" / ADDENDUM_NAME,
        PACKAGE_DIR.parents[1] / "prompts" / ADDENDUM_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


DEFAULT_ADDENDUM = _default_addendum()

# Pack sizing. The budget is characters rather than tokens because the pack is
# read by the agent as a file, not rendered into one request whose size has to
# be predicted -- which is what made the predecessor's token accounting fragile.
DEFAULT_PACK_BUDGET_CHARS = 90_000
DEFAULT_MAX_DOC_CHARS = 6_000

log = logging.getLogger(__name__)


def model_code(model: str) -> str:
    """Shorten a model id into the token that goes in the filename.

    The filename has to say which model produced the profiles, and the full id
    is too long and too punctuated to read in a directory listing.
    """
    return re.sub(r"[^a-z0-9]", "", model.lower().replace("claude-", "")) or "model"


def addendum_tag(path: Path | None) -> str:
    """The addendum's identity in the output filename.

    A boolean was not enough: two runs with *different* tuned addenda wrote to
    the same path, and the incremental skip then reported the first run's
    artifact as the second's success. Content is the honest identity rather
    than the path -- AGENTS.md treats an edit to the tuned prompt as an
    experiment that needs its own score, and an edit changes the digest while a
    rename does not. The suppressed case stays spelled out, because "built
    without the tuned rules" is the one distinction a human reads for.
    """
    if path is None:
        return "-noaddendum"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return f"-ad{digest}"


@dataclass(frozen=True)
class RequiredInput:
    """One path a run reads, and how to point it somewhere else.

    ``env_var`` is None for a location that is not machine configuration: the
    tuned addendum ships with the code, so "set $VAR" is the wrong advice for
    it, and a pair type had nowhere to say so.
    """

    path: Path
    env_var: str | None

    @property
    def hint(self) -> str:
        return f" (set ${self.env_var})" if self.env_var else " (ships with this repo)"


@dataclass(frozen=True)
class Config:
    """The knobs that change the output, and therefore its filename."""

    model: str
    variant: str
    entities_bundle: Path
    source_docs_dir: Path
    out_dir: Path
    addendum: Path | None = DEFAULT_ADDENDUM
    min_alias_probability: float = DEFAULT_MIN_PROBABILITY
    include_organizations: bool = False
    profile_owner: str | None = None
    regime: str = "agentic"
    #: None means the command line asked for nothing; the in-context provider
    #: fills in its own default in __post_init__, and the agentic regime, which
    #: has no dial, refuses the flag rather than ignoring it.
    effort: str | None = None
    # Which server is up right now, not a property of the artifact, so it is
    # deliberately absent from `stem`.
    qwen_base_url: str | None = None

    def __post_init__(self) -> None:
        """Reject a threshold outside [0, 1] here rather than at write time."""
        if not 0.0 <= self.min_alias_probability <= 1.0:
            raise ValueError("min_alias_probability must be within [0, 1]")
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")
        # Resolving against the provider rather than a fixed tuple is what
        # stops "high" being accepted for Qwen, whose top effort is xhigh:
        # sending it would buy a different amount of thinking than was asked
        # for, and the filename would still say the run was high. It is also
        # what gives a Qwen run with no --effort Qwen's default instead of
        # OpenAI's, which was rejected. Resolved here, before a pack is built.
        if self.regime == "in-context":
            object.__setattr__(
                self, "effort", self.provider.resolve_effort(self.effort)
            )
        elif self.effort is not None:
            # The agentic regime runs the SDK agent at a fixed effort, so a
            # value here would be silently discarded and the filename -- which
            # omits effort for this regime -- could not record it either.
            raise ValueError(
                "--effort applies to --regime in-context; the agentic regime "
                f"has no reasoning-effort dial (got {self.effort!r})"
            )

    @property
    def provider(self) -> providers.Provider:
        """The endpoint this run's model implies.

        Resolved from the model id so one flag decides it. Raises here, before
        a pack is built, when a Qwen run has no server to talk to.
        """
        return providers.provider_for(self.model, qwen_base_url=self.qwen_base_url)

    @property
    def stem(self) -> str:
        """Output filenames encode the config, so two runs coexist on disk.

        The model and the alias threshold are both in the name because both
        change what the profiles are: the threshold decides which entities had
        a name to search for at all.
        """
        threshold = f"p{round(self.min_alias_probability * 100):02d}"
        # Profiles built without the tuned rules are a different artifact, not a
        # cheaper one, so they must not land in a file whose name claims
        # otherwise.
        tuned = addendum_tag(self.addendum)
        return (
            f"profiles-{self.run_code}-{self.regime}-{threshold}{tuned}-{self.variant}"
        )

    @property
    def run_code(self) -> str:
        """The model token in the filename.

        The in-context regime carries its reasoning effort too: low and high are
        different runs of the same model, and the predecessor scored them as
        separate arms ("luna-low", "luna-high").
        """
        if self.regime == "in-context":
            return f"{model_code(self.model)}-{self.effort}"
        return model_code(self.model)

    def price_estimate(self, count: int) -> str:
        """What ``count`` entities would cost under this run's configuration.

        A method rather than a function because the three inputs -- regime,
        provider and rates -- are all this object's, and a call site that
        assembles them can assemble them wrongly: passing a provider on an
        agentic run was silently ignored, and dodging the provider property's
        RuntimeError was a hand-rolled local at the call site. Only an
        in-context run touches the provider, and one that could not resolve
        its endpoint never got as far as being constructed.

        A self-hosted provider quotes zero per entity because its bill is GPU
        time by the hour, not requests. Saying so is better than printing a
        fabricated per-entity rate that reads like a measurement.
        """
        if self.regime == "in-context":
            rates = self.provider.cost_per_entity_usd
            if rates == (0.0, 0.0):
                return "GPU time, not per-request"
        else:
            rates = AGENTIC_COST_PER_ENTITY_USD
        low, high = (count * rate for rate in rates)
        return f"${low:,.2f}-${high:,.2f}"

    @property
    def out_path(self) -> Path:
        """The finished artifact.

        The extension is .jsonl.gz, not this workspace's usual .csv.gz: a
        profile is a nested object carrying _grounding and _sources sidecars,
        and flattening it into columns loses the spans.
        """
        return self.out_dir / f"{self.stem}.jsonl.gz"

    @property
    def work_dir(self) -> Path:
        """Where each profile lands as it is paid for.

        Beside the output and named after it, so which run a half-finished
        directory belongs to is never a guess.
        """
        return self.out_dir / f"{self.stem}.work"

    @property
    def required_inputs(self) -> tuple[RequiredInput, ...]:
        """Every path this run reads.

        The addendum is on this list because --dry-run is the rail AGENTS.md
        tells you to trust before spending money, and an addendum that is
        missing at request time kills the run after the expensive part.
        """
        inputs = [
            RequiredInput(self.entities_bundle, settings.ENV_ENTITIES_BUNDLE),
            RequiredInput(self.source_docs_dir, settings.ENV_SOURCE_DOCS),
        ]
        if self.addendum is not None:
            inputs.append(RequiredInput(self.addendum, None))
        return tuple(inputs)


def checkpoint_name(entity_id: str) -> str:
    """A filename for one entity's checkpoint that cannot collide.

    Entity ids carry colons, so they are not filenames as they stand. The digest
    is what makes the sanitized form safe: two ids differing only in a character
    that sanitizes away would otherwise share a file, and a resumed run would
    report a paid result belonging to somebody else.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", entity_id).strip("-")[:80]
    digest = hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}.json"


def selectable(
    entities: dict[str, ResolvedEntity],
    min_probability: float,
) -> dict[str, ResolvedEntity]:
    """Entities that still have something to search for at this threshold.

    A name or an email counts. Both matter: an alias claimed by several entities
    splits its probability across them, so an entity whose only alias is mostly
    owned by someone else keeps no name -- 133 of the sample bundle's 9,180 fall
    out that way at the 0.5 default, ``BP`` at 0.0427 among them. A further 56
    never had a name to begin with, because their canonical name *is* an address
    (``politicoplaybook@politico.com``); those stay in, since an address is the
    least ambiguous matcher there is.

    Dropping an entity is correct; dropping it quietly is not, which is why this
    is a named step and the caller logs the count.
    """
    return {
        entity_id: entity
        for entity_id, entity in entities.items()
        if entity.surface_forms(min_probability) or entity.emails(min_probability)
    }


def in_scope(
    entities: dict[str, ResolvedEntity],
    cfg: Config,
) -> dict[str, ResolvedEntity]:
    """Narrow to the entities this configuration would actually profile.

    Organizations are excluded by default. The profile schema asks for age,
    birthday, spouse and education; on ``Department of Homeland Security`` every
    one of those must be null, so the spend buys a record of absences.
    """
    usable = selectable(entities, cfg.min_alias_probability)
    if cfg.include_organizations:
        return usable
    return {eid: e for eid, e in usable.items() if e.is_person}


def report_bundle(entities: dict[str, ResolvedEntity], cfg: Config) -> None:
    """Log what the bundle offers at this threshold, including what it loses."""
    usable = selectable(entities, cfg.min_alias_probability)
    scoped = in_scope(entities, cfg)
    people = sum(1 for entity in usable.values() if entity.is_person)
    log.info(
        "%d entities (%d person, %d other) usable at p>=%.2f; %d have nothing to "
        "search for; %d in scope",
        len(usable),
        people,
        len(usable) - people,
        cfg.min_alias_probability,
        len(entities) - len(usable),
        len(scoped),
    )

    spreads = {
        eid: entity.surname_spread(cfg.min_alias_probability)
        for eid, entity in scoped.items()
    }
    suspect = [
        eid for eid, spread in spreads.items() if spread > SUSPECT_SURNAME_SPREAD
    ]
    if suspect:
        worst = max(suspect, key=lambda eid: spreads[eid])
        log.warning(
            "%d in-scope entities carry more than %d distinct surnames and are "
            "probably several people merged into one; worst is %r at %d",
            len(suspect),
            SUSPECT_SURNAME_SPREAD,
            scoped[worst].canonical_name[:60],
            spreads[worst],
        )


@dataclass(frozen=True)
class Attempt:
    """One entity's finished profile, and everything the loop does with it.

    The two regimes used to hand back different shapes -- ``(dict, AgentRun)``
    against ``(dict, usage)`` -- so the checkpoint, the spend accounting and the
    log line each had to re-derive which one had run. They all read this
    instead, which is what lets the loop stop knowing.

    ``cost_usd`` is ``None`` for an *unpriced* attempt, and that is a third
    state rather than a cheap zero: an in-context request is billed in tokens
    this repo has no rate table for, and calling that 0.0 is what made
    ``--max-spend-usd`` silently inert. :meth:`is_priced` is the question the
    ceiling asks.
    """

    profile: dict[str, Any]
    usage: dict[str, Any]
    cost_usd: float | None
    #: Appended to the per-entity log line; the regime says what it spent in
    #: its own units, so the loop needs no branch to phrase it.
    summary: str
    #: Extra fields for the checkpoint, e.g. how an agent run ended.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_priced(self) -> bool:
        return self.cost_usd is not None


class Regime(Protocol):
    """Turning one evidence pack into one :class:`Attempt`.

    The seam the twoness of regimes always implied and never had. Both
    implementations attach grounding through
    :func:`grounded_profile.build_profile`, so that decision lives in one place
    rather than one per arm -- the same argument that module's docstring makes
    for holding it apart from either regime.

    Shallow deliberately: a fake implementation is three lines, which is what
    makes :func:`generate` testable without a network call or an API key.
    """

    # Declared as read-only properties, not as mutable attributes: both
    # implementations are frozen dataclasses, and a frozen attribute does not
    # structurally satisfy a settable protocol member.

    @property
    def name(self) -> str:
        """The regime tag on the checkpoint and in the filename."""
        ...

    @property
    def prices_itself(self) -> bool:
        """Whether this implementation can report dollars.

        False means ``--max-spend-usd`` cannot be enforced and is refused up
        front rather than accepted and ignored.
        """
        ...

    async def run(self, pack: EvidencePack, pack_path: Path) -> Attempt: ...


@dataclass(frozen=True)
class AgenticRegime:
    """A sandboxed Claude agent with read-only access to the corpus.

    The system prompt is built once, here, rather than by the loop: it is this
    implementation's business that the base agent rules wrap the addendum, and
    the in-context prompt has its own rules section and takes the addendum raw.
    """

    cfg: Config
    system_prompt: str
    name: str = "agentic"
    prices_itself: bool = True

    async def run(self, pack: EvidencePack, pack_path: Path) -> Attempt:
        response, agent_run = await agentic.run_agent(
            pack,
            pack_path,
            self.cfg.source_docs_dir,
            system_prompt=self.system_prompt,
            profile_owner=self.cfg.profile_owner,
            model=self.cfg.model,
        )
        cost = agent_run.cost_usd
        return Attempt(
            profile=grounded_profile.build_profile(response, pack),
            usage={},
            cost_usd=cost,
            # Pre-formatted: %-style logging has no comma flag, and "%,.2f"
            # raises at emit time rather than at import.
            summary=f"${cost:,.2f}" if cost is not None else "cost not reported",
            detail={
                "turns": agent_run.turns,
                "duration_ms": agent_run.duration_ms,
                "stop_reason": agent_run.stop_reason,
            },
        )


@dataclass(frozen=True)
class InContextRegime:
    """One chat-completions request carrying the whole pack.

    Unpriced, and says so. The endpoint reports tokens, and turning tokens into
    dollars needs a per-model rate table this repo does not have; inventing one
    would put a fabricated number where a measurement belongs. So the absence
    is declared rather than papered over with 0.0.
    """

    cfg: Config
    addendum: str | None
    name: str = "in-context"
    prices_itself: bool = False

    async def run(self, pack: EvidencePack, pack_path: Path) -> Attempt:
        del pack_path  # this regime sends the pack, it does not point at it
        # One blocking request, run off the event loop rather than stalling it.
        response, usage = await asyncio.to_thread(
            in_context.request_profile,
            pack,
            model=self.cfg.model,
            effort=self.cfg.effort,
            profile_owner=self.cfg.profile_owner,
            addendum=self.addendum,
            provider=self.cfg.provider,
        )
        return Attempt(
            profile=grounded_profile.build_profile(response, pack),
            usage=usage,
            cost_usd=None,
            summary=(
                f"{usage.get('prompt_tokens', '?')} tokens in, "
                f"{usage.get('completion_tokens', '?')} out (unpriced)"
            ),
        )


#: Which implementation can report dollars, read off the implementations
#: themselves so the answer has one source. The command line needs it before a
#: regime is built, because a ceiling nobody can enforce should be refused
#: before the bundle is even loaded.
REGIME_PRICES_ITSELF = {
    AgenticRegime.name: AgenticRegime.prices_itself,
    InContextRegime.name: InContextRegime.prices_itself,
}


def unenforceable_ceiling(regime: str) -> str | None:
    """Why ``--max-spend-usd`` cannot bound this regime, or None if it can.

    A ceiling that cannot be enforced is not a smaller risk than no ceiling; it
    is the same risk with a false assurance attached, which is how an
    in-context run came to accumulate zero however long it ran. So it is
    refused rather than accepted and ignored.
    """
    if REGIME_PRICES_ITSELF.get(regime, False):
        return None
    return (
        f"--max-spend-usd cannot bound the {regime} regime: it reports tokens, "
        "not dollars, and this repo has no rate table to price them. Bound the "
        "run with --limit, --top-mentions, --entity or --entity-list instead."
    )


def make_regime(cfg: Config) -> Regime:
    """The implementation this configuration names."""
    addendum = agentic.load_addendum(cfg.addendum) if cfg.addendum else None
    if cfg.regime == "agentic":
        return AgenticRegime(cfg, agentic.build_system_prompt(addendum))
    return InContextRegime(cfg, addendum)


async def generate(
    cfg: Config,
    entities: dict[str, ResolvedEntity],
    *,
    max_spend_usd: float | None,
    regime: Regime | None = None,
) -> tuple[int, float]:
    """Build a pack and buy a profile for each entity, checkpointing as it goes.

    Returns the number of profiles now on disk and what this invocation spent.
    Anything already checkpointed is skipped without a request, which is what
    makes an interrupted run resumable rather than repayable.

    ``regime`` is injected for tests; a real run lets :func:`make_regime`
    resolve it from the configuration.
    """
    regime = regime or make_regime(cfg)
    if max_spend_usd is not None and not regime.prices_itself:
        raise ValueError(unenforceable_ceiling(regime.name) or "")

    profiles_dir = cfg.work_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    spent = 0.0
    written = 0
    for entity_id, entity in sorted(entities.items()):
        checkpoint = profiles_dir / checkpoint_name(entity_id)
        if checkpoint.exists():
            written += 1
            continue

        pack = build_pack(
            cfg.source_docs_dir,
            entity,
            budget_chars=DEFAULT_PACK_BUDGET_CHARS,
            max_doc_chars=DEFAULT_MAX_DOC_CHARS,
            min_probability=cfg.min_alias_probability,
        )
        # The recipe rides on every checkpoint, empty packs included: what the
        # windows and cues were is as much a property of "no evidence" as of a
        # profile, and two runs at different cue settings are otherwise
        # indistinguishable on disk.
        recipe = asdict(pack.recipe)
        if pack.is_empty:
            # Nothing citable means nothing worth paying for. Recorded so a
            # resumed run does not rebuild the same empty pack.
            checkpoint.write_text(
                json.dumps(
                    {
                        "entity_id": entity_id,
                        "profile": None,
                        "reason": "no evidence",
                        "pack_recipe": recipe,
                    }
                ),
                encoding="utf-8",
            )
            written += 1
            continue

        pack_path = cfg.work_dir / "packs" / f"{checkpoint.stem}.md"
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        pack_path.write_text(pack.text, encoding="utf-8")

        attempt = await regime.run(pack, pack_path)

        checkpoint.write_text(
            json.dumps(
                {
                    "entity_id": entity_id,
                    "canonical_name": entity.canonical_name,
                    "regime": regime.name,
                    "model": cfg.model,
                    "profile": attempt.profile,
                    "cost_usd": attempt.cost_usd,
                    "usage": attempt.usage,
                    "pack_recipe": recipe,
                    **attempt.detail,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        spent += attempt.cost_usd or 0.0
        written += 1
        log.info(
            "%s (%s) -- %s, $%s so far",
            entity.canonical_name[:48],
            entity_id,
            attempt.summary,
            f"{spent:,.2f}",
        )
        # Checked after the attempt rather than before the next one, so the run
        # stops the moment it is over rather than one iteration later. It can
        # still overshoot by one profile: what an attempt costs is only known
        # once it has been bought, and refusing to start one on an estimate
        # would be a fabricated number deciding a real run.
        if max_spend_usd is not None and spent >= max_spend_usd:
            log.warning("spend ceiling $%.2f reached; stopping", max_spend_usd)
            break
    return written, spent


def collect(cfg: Config) -> int:
    """Assemble the checkpoints into the finished artifact.

    Sorted by entity id so the artifact is byte-identical for a given set of
    checkpoints, whatever order they were bought in.
    """
    profiles_dir = cfg.work_dir / "profiles"
    rows = []
    for path in sorted(profiles_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("profile") is not None:
            rows.append(row)
    rows.sort(key=lambda row: row["entity_id"])
    with gzip.open(cfg.out_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def run(
    cfg: Config,
    entities: dict[str, ResolvedEntity],
    *,
    max_spend_usd: float | None = None,
) -> None:
    """Do the work and write cfg.out_path."""
    written, spent = asyncio.run(generate(cfg, entities, max_spend_usd=max_spend_usd))
    kept = collect(cfg)
    log.info(
        "%d checkpoints, %d profiles with evidence, $%.2f spent this run",
        written,
        kept,
        spent,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--regime",
        choices=REGIMES,
        default="agentic",
        help=(
            "agentic: a sandboxed Claude agent with live corpus access. "
            "in-context: one request holding the whole pack, sent to OpenAI or "
            "to a served Qwen model depending on --model."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "model that writes the profiles; its short code enters the "
            f"filename. Defaults to {agentic.DEFAULT_MODEL} for --regime "
            f"agentic and {in_context.DEFAULT_MODEL} for --regime in-context. "
            "Any other model the key can reach works; none is default. "
            "A model id starting with 'qwen' selects the self-hosted provider "
            "and needs --qwen-base-url."
        ),
    )
    p.add_argument(
        "--effort",
        # The union, because the valid set depends on the model and argparse
        # fixes choices before --model is known. The provider rejects an effort
        # it does not implement, which is where the real check belongs.
        choices=sorted({*in_context.EFFORTS, *providers.QWEN_EFFORTS}),
        default=None,
        help=(
            "reasoning effort for --regime in-context; part of the filename. "
            "OpenAI takes low/medium/high and defaults to high; Qwen takes "
            "low/medium/xhigh and defaults to xhigh. Rejected for --regime "
            "agentic, which has no dial."
        ),
    )
    p.add_argument(
        "--qwen-base-url",
        default=None,
        help=(
            "the /v1 URL of a served Qwen model, e.g. from deploy/modal_qwen.py. "
            f"Falls back to ${providers.ENV_QWEN_BASE_URL}. Required when "
            "--model names a Qwen model; not part of the filename, because "
            "which server is up is a property of the machine, not the artifact."
        ),
    )
    p.add_argument("--variant", default="v1", help="config tag for the filename")
    # These default to None rather than to a path so that "not passed" is
    # distinguishable from "passed the same value the default would have been".
    # settings.resolve needs that distinction to let a config file beat a
    # default while still losing to a flag somebody actually typed.
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            f"path config; defaults to the nearest "
            f"{settings.CONFIG_DIR}/{settings.CONFIG_FILENAME} "
            f"at or above the working directory, or ${settings.ENV_CONFIG}"
        ),
    )
    p.add_argument(
        "--entities-bundle",
        type=Path,
        default=None,
        help=f"resolved-entities bundle; ${settings.ENV_ENTITIES_BUNDLE} or [paths]",
    )
    p.add_argument(
        "--source-docs",
        type=Path,
        default=None,
        help=f"source-document corpus; ${settings.ENV_SOURCE_DOCS} or [paths]",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--addendum",
        type=Path,
        default=DEFAULT_ADDENDUM,
        help=(
            "tuned prompt addendum prepended to the system prompt; its content "
            f"digest enters the filename. Defaults to {DEFAULT_ADDENDUM}, which "
            "ships with this checkout"
        ),
    )
    p.add_argument(
        "--no-addendum",
        action="store_true",
        help="run without the tuned rules; the profiles are a different artifact",
    )
    p.add_argument(
        "--min-alias-probability",
        type=float,
        default=DEFAULT_MIN_PROBABILITY,
        help=(
            "keep an alias only when it is at least this likely to refer to the "
            "entity claiming it; above 0.5 at most one entity owns a string"
        ),
    )
    p.add_argument(
        "--include-organizations",
        action="store_true",
        help="also profile entities whose case is entity_organization",
    )
    p.add_argument(
        "--profile-owner",
        default=None,
        help="the Rolodex owner, against whom relationship_to_user is defined",
    )

    scope = p.add_argument_group(
        "scope",
        "This stage spends money per entity. One of these is required.",
    )
    scope.add_argument("--limit", type=int, default=None, help="profile the first N")
    scope.add_argument(
        "--top-mentions",
        type=int,
        default=None,
        metavar="N",
        help="profile the N most-mentioned entities in scope",
    )
    scope.add_argument(
        "--entity",
        action="append",
        default=None,
        metavar="ID",
        help="profile this resolved_entity_id; repeatable",
    )
    scope.add_argument(
        "--entity-list",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "profile the resolved_entity_ids listed in FILE, one per line; "
            "blank lines and # comments are ignored"
        ),
    )
    scope.add_argument(
        "--all", action="store_true", help="profile every entity in scope"
    )
    scope.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help=(
            "stop the run once this much has been spent; refused for a regime "
            "that reports tokens rather than dollars, because a ceiling that "
            "cannot be enforced is a false assurance"
        ),
    )

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


def read_entity_list(path: Path) -> list[str]:
    """Read entity ids to profile, one per line.

    Blank lines and ``#`` comments are skipped, so a curated list can record why
    each id is on it -- which is the whole reason to keep one in a file rather
    than in shell history. Ids repeat harmlessly; order is preserved, and the
    first occurrence wins, because that order is what the run reports.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entity_id = line.split("#", 1)[0].strip()
        if entity_id and entity_id not in seen:
            seen.add(entity_id)
            ids.append(entity_id)
    if not ids:
        raise ValueError(f"{path} lists no entity ids")
    return ids


def requested_ids(args: argparse.Namespace) -> list[str]:
    """Every id named explicitly, by --entity or --entity-list."""
    ids = list(args.entity or [])
    if args.entity_list:
        ids.extend(eid for eid in read_entity_list(args.entity_list) if eid not in ids)
    return ids


def chosen(
    scoped: dict[str, ResolvedEntity],
    args: argparse.Namespace,
) -> dict[str, ResolvedEntity]:
    """Apply the explicit ids, --top-mentions or --limit to the in-scope entities."""
    explicit = requested_ids(args)
    if explicit:
        return {eid: scoped[eid] for eid in explicit if eid in scoped}
    if args.top_mentions is not None:
        # Ties break on entity id so the same N is picked every run. Note what
        # this actually selects: the corpus's most-mentioned "people" include
        # mailing lists, a street address and a bare phone number, because
        # resolution files them all under participant_person. It is the honest
        # top-N, not a curated one.
        ranked = sorted(
            scoped.items(), key=lambda item: (-item[1].mention_count, item[0])
        )
        return dict(ranked[: args.top_mentions])
    if args.limit is not None:
        return dict(sorted(scoped.items())[: args.limit])
    return scoped


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        paths = settings.resolve(
            config=args.config,
            flags={
                "entities_bundle": args.entities_bundle,
                "source_docs": args.source_docs,
                "out_dir": args.out_dir,
            },
        )
    except ValueError as exc:
        # A config file that cannot be read is not a reason to fall back to the
        # defaults: it would read some other corpus and report success.
        log.error("%s", exc)
        return 1

    # The default model depends on the regime, so it cannot be an argparse
    # default -- picking one there would silently send an OpenAI model id to the
    # Agent SDK, or the reverse.
    model = args.model or (
        agentic.DEFAULT_MODEL if args.regime == "agentic" else in_context.DEFAULT_MODEL
    )

    try:
        cfg = Config(
            model=model,
            variant=args.variant,
            entities_bundle=paths.entities_bundle,
            source_docs_dir=paths.source_docs,
            out_dir=paths.out_dir,
            addendum=None if args.no_addendum else args.addendum,
            min_alias_probability=args.min_alias_probability,
            include_organizations=args.include_organizations,
            profile_owner=args.profile_owner,
            regime=args.regime,
            effort=args.effort,
            qwen_base_url=args.qwen_base_url,
        )
    except (ValueError, RuntimeError) as exc:
        # A Qwen run with no server, or an effort the chosen model does not
        # implement. Both are typos at the command line, not stack traces.
        log.error("%s", exc)
        return 2
    # Refused here, before the bundle is loaded and before --dry-run reports a
    # plan the ceiling would not actually have bounded.
    if args.max_spend_usd is not None:
        problem = unenforceable_ceiling(cfg.regime)
        if problem:
            log.error("%s", problem)
            return 2

    # Checked before anything reads cfg.out_path, whose name now carries the
    # addendum's content digest. An addendum that is missing or empty is a
    # broken checkout rather than an unconfigured machine, so it is fatal even
    # under --dry-run: the point of a dry run is that a green one means the
    # next invocation can spend money.
    if cfg.addendum is not None:
        try:
            agentic.load_addendum(cfg.addendum)
        except (OSError, ValueError) as exc:
            log.error("prompt addendum unusable: %s", exc)
            log.error("pass --addendum FILE, or --no-addendum to run untuned")
            return 1

    # Re-running skips work already on disk. This is what makes the stage safe
    # to invoke repeatedly, and --force the only way to overwrite.
    if cfg.out_path.exists() and not args.force:
        log.info("%s exists; nothing to do (--force to rewrite)", cfg.out_path)
        return 0

    missing = [item for item in cfg.required_inputs if not item.path.exists()]

    if args.dry_run:
        # A dry run reports the configured corpus rather than requiring it, so
        # it stays the cheap way to check where a run would read and write on a
        # machine that has not been given the data yet. It names the config file
        # first: a path can now arrive from a file the command line never
        # mentions, and "why is it reading there" needs an answer on one line.
        log.info("config   %s", paths.origin)
        for item in cfg.required_inputs:
            state = "missing" if not item.path.exists() else "ok"
            log.info("%-8s %s%s", state, item.path, item.hint)
        if not missing:
            entities = load_bundle(cfg.entities_bundle)
            report_bundle(entities, cfg)
            scoped = in_scope(entities, cfg)
            for eid in requested_ids(args):
                if eid not in scoped:
                    log.warning(
                        "%s is not in scope and would be skipped",
                        eid,
                    )
            picked = chosen(scoped, args)
            log.info(
                "would profile %d entities via %s, about %s",
                len(picked),
                cfg.regime,
                cfg.price_estimate(len(picked)),
            )
        log.info("would write %s", cfg.out_path)
        return 0

    if missing:
        for item in missing:
            log.error("%s is missing%s", item.path, item.hint)
        return 1

    entities = load_bundle(cfg.entities_bundle)
    report_bundle(entities, cfg)
    scoped = in_scope(entities, cfg)

    # An unscoped run is refused rather than defaulted. The whole corpus is a
    # four-figure invoice, and a flag nobody typed is not consent to it.
    if not (
        args.all
        or args.limit is not None
        or args.top_mentions is not None
        or args.entity
        or args.entity_list
    ):
        log.error(
            "refusing to profile %d entities via %s (about %s) without a "
            "scope: pass --limit N, --top-mentions N, --entity ID, "
            "--entity-list FILE, or --all",
            len(scoped),
            cfg.regime,
            cfg.price_estimate(len(scoped)),
        )
        return 1

    # An id that silently vanishes buys nine profiles where ten were asked for,
    # and the run reports success. Say which ids, and say which of the two
    # reasons applies: a typo and an out-of-scope organization need different
    # fixes.
    explicit = requested_ids(args)
    unknown = [eid for eid in explicit if eid not in entities]
    filtered = [eid for eid in explicit if eid in entities and eid not in scoped]
    if unknown:
        log.error("not in the bundle: %s", ", ".join(unknown))
    if filtered:
        log.error(
            "in the bundle but out of scope at p>=%.2f%s: %s",
            cfg.min_alias_probability,
            "" if cfg.include_organizations else " (organizations excluded)",
            ", ".join(filtered),
        )
    if unknown or filtered:
        return 1

    picked = chosen(scoped, args)
    if not picked:
        log.error("no entities selected")
        return 1

    log.info(
        "profiling %d entities via %s, about %s",
        len(picked),
        cfg.regime,
        cfg.price_estimate(len(picked)),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    run(cfg, picked, max_spend_usd=args.max_spend_usd)
    log.info("wrote %s", cfg.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
