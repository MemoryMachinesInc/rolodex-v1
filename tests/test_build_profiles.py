"""What the pipeline guarantees before it is allowed to spend anything.

Every test passes explicit paths. Defaults point into ``data/``, which is real
on a configured machine, and a test that read it would pass or fail depending on
whose laptop it ran on.

The scope guard, the filename and the selection rules are all decided before
the first request, which is exactly why they are the parts worth testing
without spending. ``generate`` itself is tested too, and costs nothing to test:
the regime is a seam, so a fake implementation stands in for an agent session
or a chat-completions request without either.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rolodex_v1.build_profiles import (
    DEFAULT_ADDENDUM,
    DEFAULT_PACK_BUDGET_CHARS,
    AgenticRegime,
    Attempt,
    Config,
    InContextRegime,
    chosen,
    collect,
    generate,
    in_scope,
    main,
    make_regime,
    model_code,
    pack_name,
    read_checkpoints,
)
from rolodex_v1.evidence_pack import CUE_NEAR_NAME, NAME_WINDOW
from rolodex_v1.resolved_entities import Alias, ResolvedEntity

PERSON = "participant_person"
ORGANIZATION = "entity_organization"


def config(tmp_path: Path, **overrides: Any) -> Config:
    """A Config whose corpus paths are inside tmp_path unless overridden."""
    kwargs: dict[str, Any] = {
        "model": "claude-opus-5",
        "variant": "v1",
        "entities_bundle": tmp_path / "resolved_entities_bundle.json",
        "source_docs_dir": tmp_path / "source_docs",
        "out_dir": tmp_path,
    }
    kwargs.update(overrides)
    return Config(**kwargs)


def entity(entity_id: str, name: str, case: str = PERSON) -> ResolvedEntity:
    return ResolvedEntity(
        entity_id=entity_id,
        canonical_name=name,
        case=case,
        canonical_type=None,
        mention_count=1,
        memory_count=1,
        aliases=(Alias(text=name, probability=1.0),),
    )


def write_bundle(tmp_path: Path, records: list[ResolvedEntity] | None = None) -> Path:
    records = records or [entity("proto:participant_person:20", "Josh Earnest")]
    path = tmp_path / "resolved_entities_bundle.json"
    path.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "resolved_entity_id": e.entity_id,
                        "canonical_name": e.canonical_name,
                        "canonical_type": e.canonical_type,
                        "case": e.case,
                        "mention_count": e.mention_count,
                        "memory_count": e.memory_count,
                        "aliases": [
                            {"alias": a.text, "probability": a.probability}
                            for a in e.aliases
                        ],
                    }
                    for e in records
                ]
            }
        )
    )
    return path


def addendum_file(tmp_path: Path, text: str, name: str = "addendum.jinja") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--out-dir",
        str(tmp_path),
        "--entities-bundle",
        str(tmp_path / "resolved_entities_bundle.json"),
        "--source-docs",
        str(tmp_path / "source_docs"),
        *extra,
    ]


def test_the_filename_names_the_model_that_wrote_the_profiles() -> None:
    """A directory listing has to say which model produced an artifact."""
    assert model_code("claude-opus-5") == "opus5"
    assert model_code("claude-sonnet-5") == "sonnet5"


def test_config_encodes_settings_in_the_filename(tmp_path: Path) -> None:
    """Two configs must not collide on disk."""
    a = config(tmp_path, model="claude-opus-5")
    b = config(tmp_path, model="claude-sonnet-5")
    assert a.out_path != b.out_path
    assert "opus5" in a.out_path.name


def test_the_alias_threshold_is_in_the_filename(tmp_path: Path) -> None:
    """It decides which entities had a name to search for, so it changes output."""
    strict = config(tmp_path, min_alias_probability=0.9)
    loose = config(tmp_path, min_alias_probability=0.5)
    assert "p90" in strict.out_path.name
    assert "p50" in loose.out_path.name


def test_dropping_the_tuned_rules_is_recorded_in_the_filename(tmp_path: Path) -> None:
    """Profiles built without the addendum are a different artifact."""
    tuned = config(tmp_path, addendum=addendum_file(tmp_path, "Rules."))
    untuned = config(tmp_path, addendum=None)
    assert "noaddendum" in untuned.out_path.name
    assert "noaddendum" not in tuned.out_path.name


def test_two_different_addenda_do_not_share_an_output_file(tmp_path: Path) -> None:
    """A boolean let the incremental skip report one run's artifact as another's."""
    first = config(tmp_path, addendum=addendum_file(tmp_path, "Rules.", "a.jinja"))
    second = config(tmp_path, addendum=addendum_file(tmp_path, "Other.", "b.jinja"))
    assert first.out_path != second.out_path


def test_the_filename_follows_the_addendums_content_not_its_name(
    tmp_path: Path,
) -> None:
    """An edit to the tuned prompt is an experiment that needs its own score."""
    path = addendum_file(tmp_path, "Rules.")
    before = config(tmp_path, addendum=path).out_path
    copy = addendum_file(tmp_path, "Rules.", "renamed.jinja")
    assert config(tmp_path, addendum=copy).out_path == before
    path.write_text("Rules, revised.", encoding="utf-8")
    assert config(tmp_path, addendum=path).out_path != before


def test_the_default_addendum_does_not_depend_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CWD-relative default made --dry-run pass and then the paid run die."""
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_ADDENDUM.is_absolute()
    assert DEFAULT_ADDENDUM.is_file()


def test_a_dry_run_checks_the_addendum_it_would_send(tmp_path: Path) -> None:
    """--dry-run is the rail AGENTS.md says to trust before spending money."""
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    missing = tmp_path / "gone.jinja"
    code = main(argv(tmp_path, "--dry-run", "--addendum", str(missing)))
    assert code == 1


def test_an_unreadable_addendum_is_caught_before_the_run_spends(
    tmp_path: Path,
) -> None:
    """load_addendum used to raise inside generate(), after the pack was built."""
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    empty = addendum_file(tmp_path, "   ", "empty.jinja")
    assert main(argv(tmp_path, "--all", "--addendum", str(empty))) == 1


def test_a_probability_outside_the_unit_interval_is_rejected(tmp_path: Path) -> None:
    """A threshold above 1.0 silently selects nothing at all."""
    with pytest.raises(ValueError, match="within"):
        config(tmp_path, min_alias_probability=1.5)


def test_the_work_directory_is_named_after_its_output(tmp_path: Path) -> None:
    """Which run a half-finished directory belongs to must never be a guess."""
    cfg = config(tmp_path)
    assert cfg.work_dir.name.startswith(cfg.stem)


def test_a_pack_name_cannot_collide() -> None:
    """Two ids differing only in a sanitized character must not share a file.

    Without the digest the agent would be handed evidence belonging to somebody
    else.
    """
    assert pack_name("proto:person:1") != pack_name("proto/person:1")
    assert ":" not in pack_name("proto:person:1")


def test_organizations_are_out_of_scope_by_default(tmp_path: Path) -> None:
    """The schema asks for age, spouse and education; an agency has none of them."""
    entities = {
        "p": entity("p", "Josh Earnest"),
        "o": entity("o", "Department of Homeland Security", ORGANIZATION),
    }
    assert set(in_scope(entities, config(tmp_path))) == {"p"}
    included = config(tmp_path, include_organizations=True)
    assert set(in_scope(entities, included)) == {"p", "o"}


def test_an_unscoped_run_is_refused(tmp_path: Path) -> None:
    """The whole corpus is a four-figure invoice; an untyped flag is not consent."""
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    assert main(argv(tmp_path)) == 1
    assert not list(tmp_path.glob("profiles-*.json"))


def test_an_unscoped_dry_run_is_allowed(tmp_path: Path) -> None:
    """Pricing the work must not require committing to it."""
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    assert main(argv(tmp_path, "--dry-run")) == 0


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """A --dry-run that touched the output clobbered real data once."""
    out_dir = tmp_path / "out"
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    assert main(argv(tmp_path, "--dry-run") + ["--out-dir", str(out_dir)]) == 0
    assert not out_dir.exists()


def test_dry_run_survives_an_unconfigured_corpus(tmp_path: Path) -> None:
    """It is the cheap check on a machine that has not been given the data."""
    assert main(argv(tmp_path, "--dry-run")) == 0


def test_a_missing_corpus_fails_instead_of_writing(tmp_path: Path) -> None:
    """Half a profile set is worse than none, and looks the same on disk."""
    out_dir = tmp_path / "out"
    assert main(argv(tmp_path, "--all") + ["--out-dir", str(out_dir)]) == 1
    assert not out_dir.exists()


def test_a_missing_entities_bundle_fails_even_with_a_corpus(tmp_path: Path) -> None:
    """Evidence with nobody to profile is not a runnable configuration."""
    (tmp_path / "source_docs").mkdir()
    assert main(argv(tmp_path, "--all")) == 1


def test_existing_output_is_not_rewritten(tmp_path: Path) -> None:
    """Re-running is a no-op, which is what makes the stage incremental."""
    cfg = config(tmp_path)
    cfg.out_path.write_bytes(b"sentinel")
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    assert main(argv(tmp_path, "--all")) == 0
    assert cfg.out_path.read_bytes() == b"sentinel"


def test_collect_assembles_checkpoints_in_a_stable_order(tmp_path: Path) -> None:
    """Two runs holding the same checkpoints must produce the same artifact."""
    cfg = config(tmp_path)
    write_checkpoints(
        cfg,
        {"entity_id": "proto:b", "profile": {"full_name": "B"}},
        {"entity_id": "proto:a", "profile": {"full_name": "A"}},
    )
    assert collect(cfg) == 2
    profiles = json.loads(cfg.out_path.read_text(encoding="utf-8"))
    assert list(profiles) == ["proto:a", "proto:b"]
    assert profiles["proto:a"] == {"full_name": "A"}


def test_an_entity_with_no_evidence_is_checkpointed_but_not_emitted(
    tmp_path: Path,
) -> None:
    """It must not be re-bought on resume, and it must not become an empty profile."""
    cfg = config(tmp_path)
    write_checkpoints(
        cfg, {"entity_id": "proto:x", "profile": None, "reason": "no evidence"}
    )
    assert collect(cfg) == 0
    # Not a null-valued key: that reads as a profile that came back empty.
    assert json.loads(cfg.out_path.read_text(encoding="utf-8")) == {}


def test_the_regime_and_effort_are_in_the_filename(tmp_path: Path) -> None:
    """Low and high are separate arms of the same model, scored separately."""
    agentic_cfg = config(tmp_path, regime="agentic")
    low = config(tmp_path, regime="in-context", model="gpt-5.6-luna", effort="low")
    high = config(tmp_path, regime="in-context", model="gpt-5.6-luna", effort="high")
    assert "agentic" in agentic_cfg.out_path.name
    assert low.out_path != high.out_path
    assert "luna-low-in-context" in low.out_path.name
    assert "luna-high-in-context" in high.out_path.name


def test_an_unknown_regime_is_rejected(tmp_path: Path) -> None:
    """A typo would otherwise name a file after a regime that did not run."""
    with pytest.raises(ValueError, match="regime must be one of"):
        config(tmp_path, regime="incontext")


def test_an_unknown_effort_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="effort must be one of"):
        config(tmp_path, regime="in-context", effort="highest")


def test_top_mentions_picks_by_mention_count_not_by_id(tmp_path: Path) -> None:
    """--limit takes the first N alphabetically, which is not what was asked for."""
    import argparse

    scoped = {}
    for entity_id, name, mentions in (
        ("z:1", "Loud Person", 500),
        ("a:1", "Quiet Person", 2),
        ("m:1", "Middle Person", 50),
    ):
        e = entity(entity_id, name)
        scoped[entity_id] = ResolvedEntity(
            entity_id=e.entity_id,
            canonical_name=e.canonical_name,
            case=e.case,
            canonical_type=e.canonical_type,
            mention_count=mentions,
            memory_count=mentions,
            aliases=e.aliases,
        )

    args = argparse.Namespace(entity=None, entity_list=None, top_mentions=2, limit=None)
    assert list(chosen(scoped, args)) == ["z:1", "m:1"]

    args = argparse.Namespace(entity=None, entity_list=None, top_mentions=None, limit=2)
    assert list(chosen(scoped, args)) == ["a:1", "m:1"]


def test_top_mentions_is_a_valid_scope(tmp_path: Path) -> None:
    """It must satisfy the spend guard, or it cannot be used at all."""
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    assert main(argv(tmp_path, "--dry-run", "--top-mentions", "3")) == 0


def test_an_entity_list_is_read_one_id_per_line(tmp_path: Path) -> None:
    """Comments and blanks are skipped so a curated list can say why."""
    from rolodex_v1.build_profiles import read_entity_list

    path = tmp_path / "ids.txt"
    path.write_text(
        "# curated\n"
        "proto:participant_person:55   # Lucille Nguyen\n"
        "\n"
        "proto:participant_person:39\n"
        "proto:participant_person:55\n",  # duplicate
        encoding="utf-8",
    )
    assert read_entity_list(path) == [
        "proto:participant_person:55",
        "proto:participant_person:39",
    ]


def test_an_empty_entity_list_is_refused(tmp_path: Path) -> None:
    """A file of only comments would otherwise select nothing and look fine."""
    from rolodex_v1.build_profiles import read_entity_list

    path = tmp_path / "ids.txt"
    path.write_text("# nothing here\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lists no entity ids"):
        read_entity_list(path)


def test_an_entity_list_is_a_valid_scope(tmp_path: Path) -> None:
    """It must satisfy the spend guard, or the flag cannot be used at all."""
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    path = tmp_path / "ids.txt"
    path.write_text("proto:participant_person:20\n", encoding="utf-8")
    assert main(argv(tmp_path, "--dry-run", "--entity-list", str(path))) == 0


def test_an_unknown_id_fails_instead_of_shrinking_the_run(tmp_path: Path) -> None:
    """Nine profiles where ten were asked for, reported as success, is the bug."""
    write_bundle(tmp_path)
    (tmp_path / "source_docs").mkdir()
    path = tmp_path / "ids.txt"
    path.write_text("proto:participant_person:20\nproto:typo:999\n", encoding="utf-8")
    assert main(argv(tmp_path, "--entity-list", str(path))) == 1


def test_the_spend_log_line_renders() -> None:
    """`%,.2f` is f-string syntax; %-logging raises on it only when it fires.

    The agentic arm emitted a traceback per entity for exactly this, after the
    money was already spent.
    """
    template = "%s (%s) -- $%.2f, $%s so far"
    args = ("Someone", "proto:1", 1.31, f"{11622.046:,.2f}")
    # %-formatting is what logging does, so it is what this must exercise.
    assert template % args == "Someone (proto:1) -- $1.31, $11,622.05 so far"  # noqa: UP031


def test_a_qwen_run_with_no_effort_gets_qwens_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: it inherited OpenAI's "high", which Qwen rejects, and exited 2.

    Constructing the Config is the whole test -- the resolution happens there,
    before a pack is built, which is where trap 8 says the check belongs.
    """
    monkeypatch.setenv("ROLODEX_V1_QWEN_BASE_URL", "https://example.invalid/v1")
    cfg = config(tmp_path, regime="in-context", model="qwen3.8-27b")
    assert cfg.effort == "xhigh"
    assert "xhigh-in-context" in cfg.out_path.name


def test_an_effort_qwen_does_not_implement_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROLODEX_V1_QWEN_BASE_URL", "https://example.invalid/v1")
    with pytest.raises(ValueError, match="effort must be one of"):
        config(tmp_path, regime="in-context", model="qwen3.8-27b", effort="high")


def test_an_effort_on_an_agentic_run_is_refused_not_ignored(tmp_path: Path) -> None:
    """It reached nothing and the agentic filename could not have recorded it.

    Accepting it silently would let two runs asked for at different efforts
    land in one file, both of them actually run at the agent's fixed effort.
    """
    with pytest.raises(ValueError, match="no reasoning-effort dial"):
        config(tmp_path, regime="agentic", effort="low")


def test_an_agentic_run_needs_no_effort(tmp_path: Path) -> None:
    cfg = config(tmp_path, regime="agentic")
    assert cfg.effort is None


def test_the_estimate_prices_the_regime_that_will_run(tmp_path: Path) -> None:
    """Agentic and OpenAI in-context differ by an order of magnitude per entity."""
    agentic_cfg = config(tmp_path, regime="agentic")
    openai_cfg = config(tmp_path, regime="in-context", model="gpt-5.6-luna")
    assert agentic_cfg.price_estimate(10) == "$3.00-$11.00"
    assert openai_cfg.price_estimate(10) == "$0.30-$1.50"


def test_a_self_hosted_run_is_quoted_as_gpu_time_not_as_dollars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-entity rate for an hourly bill would read like a measurement."""
    monkeypatch.setenv("ROLODEX_V1_QWEN_BASE_URL", "https://example.invalid/v1")
    cfg = config(tmp_path, regime="in-context", model="qwen3.8-27b")
    quote = cfg.price_estimate(10)
    assert quote == "GPU time, not per-request"
    assert "$" not in quote


def test_the_estimate_takes_only_a_count(tmp_path: Path) -> None:
    """Regime and provider are the Config's; a call site cannot get them wrong.

    The bound method is taken through an untyped name so the deliberate misuse
    is a runtime assertion rather than a type error the checker reports.
    """
    cfg = config(tmp_path, regime="agentic")
    estimate: Any = cfg.price_estimate
    with pytest.raises(TypeError):
        estimate(10, "in-context")


# --- generate() -------------------------------------------------------------
#
# These do call the loop that spends, which is only testable because the regime
# is a seam: FakeRegime satisfies the same interface as the two real
# implementations with no network call, no API key and no Node subprocess.


class FakeRegime:
    """A regime that records what it was asked for and charges a fixed price."""

    def __init__(self, *, cost: float | None = 1.0, prices_itself: bool = True):
        self.name = "fake"
        self.prices_itself = prices_itself
        self.cost = cost
        self.calls: list[str] = []

    async def run(self, pack: Any, pack_path: Path) -> Attempt:
        self.calls.append(pack.name)
        return Attempt(
            profile={"full_name": pack.name},
            usage={"prompt_tokens": 10},
            cost_usd=self.cost,
            summary="fake",
        )


def corpus_with(tmp_path: Path, **documents: str) -> Path:
    directory = tmp_path / "source_docs"
    directory.mkdir(exist_ok=True)
    for stem, text in documents.items():
        (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
    return directory


def generated(
    cfg: Config,
    entities: dict[str, ResolvedEntity],
    regime: FakeRegime,
    *,
    max_spend_usd: float | None = None,
) -> tuple[int, float]:
    return asyncio.run(
        generate(cfg, entities, max_spend_usd=max_spend_usd, regime=regime)
    )


def checkpoints(cfg: Config) -> list[dict[str, Any]]:
    return list(read_checkpoints(cfg.work_dir / "checkpoints.jsonl").values())


def write_checkpoints(cfg: Config, *records: dict[str, Any]) -> None:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "checkpoints.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_a_checkpointed_entity_is_not_bought_again(tmp_path: Path) -> None:
    """Resuming is the whole reason the checkpoint exists; re-buying is the bug."""
    cfg = config(tmp_path, source_docs_dir=corpus_with(tmp_path, a="Ada Byron wrote."))
    people = {"e1": entity("e1", "Ada Byron")}
    regime = FakeRegime()

    written, _ = generated(cfg, people, regime)
    assert (written, regime.calls) == (1, ["Ada Byron"])

    written, spent = generated(cfg, people, regime)
    assert written == 1
    assert spent == 0.0
    assert regime.calls == ["Ada Byron"]  # no second request


def test_an_entity_with_no_evidence_is_checkpointed_without_a_request(
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path, source_docs_dir=corpus_with(tmp_path, a="Nobody here."))
    regime = FakeRegime()

    written, spent = generated(cfg, {"e1": entity("e1", "Ada Byron")}, regime)

    assert (written, spent, regime.calls) == (1, 0.0, [])
    (row,) = checkpoints(cfg)
    assert row["profile"] is None
    assert row["reason"] == "no evidence"
    # The log carries what the artifact does not, empty packs included: which
    # regime and model the run would have bought with, and that it spent nothing.
    assert row["regime"] == regime.name
    assert row["model"] == cfg.model
    assert row["cost_usd"] == 0.0
    assert row["usage"] is None


def test_the_pack_recipe_reaches_the_checkpoint(
    tmp_path: Path,
) -> None:
    """Two runs at different cue settings are otherwise identical on disk."""
    cfg = config(tmp_path, source_docs_dir=corpus_with(tmp_path, a="Ada Byron wrote."))
    generated(cfg, {"e1": entity("e1", "Ada Byron")}, FakeRegime())

    (row,) = checkpoints(cfg)
    recipe = row["pack_recipe"]
    assert recipe["name_window"] == list(NAME_WINDOW)
    assert recipe["field_cues_digest"].startswith("sha256:")
    assert recipe["min_alias_probability"] == cfg.min_alias_probability
    assert recipe["budget_chars"] == DEFAULT_PACK_BUDGET_CHARS

    # The recipe stays on the checkpoint: the artifact is profiles, and the log
    # beside it is the record of how they were built.
    collect(cfg)
    assert list(json.loads(cfg.out_path.read_text(encoding="utf-8"))) == ["e1"]


def test_an_empty_pack_records_its_recipe_too(tmp_path: Path) -> None:
    """How the windows were drawn is as much a property of "no evidence"."""
    cfg = config(tmp_path, source_docs_dir=corpus_with(tmp_path, a="Nobody here."))
    generated(cfg, {"e1": entity("e1", "Ada Byron")}, FakeRegime())
    (row,) = checkpoints(cfg)
    assert row["pack_recipe"]["cue_near_name"] == CUE_NEAR_NAME


def test_the_spend_ceiling_stops_the_run(tmp_path: Path) -> None:
    """It stops on the attempt that crosses, not one entity later."""
    cfg = config(
        tmp_path,
        source_docs_dir=corpus_with(
            tmp_path,
            a="Ada Byron wrote.",
            b="Bea Cross wrote.",
            c="Cal Drew wrote.",
        ),
    )
    people = {
        "e1": entity("e1", "Ada Byron"),
        "e2": entity("e2", "Bea Cross"),
        "e3": entity("e3", "Cal Drew"),
    }
    regime = FakeRegime(cost=1.0)

    written, spent = generated(cfg, people, regime, max_spend_usd=2.0)

    assert (written, spent) == (2, 2.0)
    assert regime.calls == ["Ada Byron", "Bea Cross"]


def test_a_regime_that_cannot_price_itself_refuses_the_ceiling(
    tmp_path: Path,
) -> None:
    """An unenforceable ceiling is the same risk with a false assurance on it."""
    cfg = config(tmp_path, source_docs_dir=corpus_with(tmp_path, a="Ada Byron wrote."))
    regime = FakeRegime(cost=None, prices_itself=False)

    with pytest.raises(ValueError, match="cannot bound"):
        generated(cfg, {"e1": entity("e1", "Ada Byron")}, regime, max_spend_usd=5.0)

    assert regime.calls == []


def test_the_command_line_refuses_an_unenforceable_ceiling_before_anything(
    tmp_path: Path,
) -> None:
    """Refused before the bundle is loaded, so nothing needs to exist yet."""
    code = main(
        [
            "--regime",
            "in-context",
            "--limit",
            "1",
            "--max-spend-usd",
            "5",
            "--entities-bundle",
            str(tmp_path / "bundle.json"),
            "--source-docs",
            str(tmp_path / "source_docs"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert not (tmp_path / "out").exists()


def test_make_regime_dispatches_on_the_configured_regime(tmp_path: Path) -> None:
    """One flag decides which implementation runs; the loop never asks again."""
    addendum = tmp_path / "addendum.jinja"
    addendum.write_text("Extra rules.", encoding="utf-8")

    agentic_regime = make_regime(config(tmp_path, regime="agentic", addendum=addendum))
    assert isinstance(agentic_regime, AgenticRegime)
    assert agentic_regime.name == "agentic"
    assert agentic_regime.prices_itself
    # The base agent rules wrap the addendum here, not in the loop.
    assert "Extra rules." in agentic_regime.system_prompt

    in_context_regime = make_regime(
        config(tmp_path, regime="in-context", model="gpt-5.6-luna", addendum=addendum)
    )
    assert isinstance(in_context_regime, InContextRegime)
    assert in_context_regime.name == "in-context"
    assert not in_context_regime.prices_itself
    # The in-context prompt has its own rules section and takes it raw.
    assert in_context_regime.addendum == "Extra rules."


def test_the_checkpoint_records_the_regime_that_ran(tmp_path: Path) -> None:
    cfg = config(tmp_path, source_docs_dir=corpus_with(tmp_path, a="Ada Byron wrote."))
    generated(cfg, {"e1": entity("e1", "Ada Byron")}, FakeRegime())
    (row,) = checkpoints(cfg)
    assert row["regime"] == "fake"
    assert row["cost_usd"] == 1.0
    assert row["usage"] == {"prompt_tokens": 10}


def test_a_narrowed_resume_counts_only_its_own_entities(tmp_path: Path) -> None:
    """The log outlives the run that wrote it; its size is not this run's total.

    Reporting every id in the log would tell a --limit 5 resume it produced the
    hundreds of profiles an earlier, wider run had bought.
    """
    cfg = config(tmp_path, source_docs_dir=corpus_with(tmp_path, a="Ada Byron wrote."))
    write_checkpoints(
        cfg,
        {"entity_id": "e1", "profile": {"full_name": "Ada"}},
        {"entity_id": "elsewhere", "profile": {"full_name": "Someone"}},
    )

    written, spent = generated(cfg, {"e1": entity("e1", "Ada Byron")}, FakeRegime())

    assert (written, spent) == (1, 0.0)


def test_an_unreadable_checkpoint_line_does_not_lose_the_log(tmp_path: Path) -> None:
    """A crash mid-write must cost the last record, not every profile above it."""
    cfg = config(tmp_path)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "checkpoints.jsonl").write_text(
        json.dumps({"entity_id": "e1", "profile": {"full_name": "Ada"}})
        + "\n"
        + "null\n"  # well-formed JSON, not a record
        + json.dumps({"profile": {"full_name": "Nameless"}})
        + "\n"  # no entity id
        + '{"entity_id": "e2", "prof',  # truncated by the crash
        encoding="utf-8",
    )
    assert list(read_checkpoints(cfg.work_dir / "checkpoints.jsonl")) == ["e1"]
